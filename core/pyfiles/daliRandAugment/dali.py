import torch
import math
import time
import time
import numpy as np
import torch
import random
from itertools import combinations

import torch.multiprocessing as multiprocessing
from torch._six import queue
from torch.utils.data import RandomSampler
from posix_ipc import O_CREAT, SharedMemory, unlink_shared_memory
from .sampler import FinegrainedBatchSampler, BatchSampler
import ctypes

try:
    from nvidia import dali
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.backend import TensorGPU, TensorListGPU, TensorListCPU
    from nvidia.dali.plugin.base_iterator import _DaliBaseIterator
    from nvidia.dali.plugin.base_iterator import LastBatchPolicy
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
    import nvidia.dali.backend as backend
except ImportError:
    raise ImportError(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run.")
    
if __debug__:
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger(__name__)


class ExternalInputGpuIterator(object):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset = dataset
        self.data_path_list = None
        self.data_label_list = None
        self.files = []
        self.index = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        batches = []
        labels = []
        
        for _ in range(self.batch_size):
            if self.index >= len(self.dataset):
                self.index = 0
            jpeg_file, label = self.dataset.samples[self.index]
            self.index += 1
            
            f = open(jpeg_file, 'rb')
            batches.append(np.frombuffer(f.read(), dtype=np.uint8))
            labels.append(np.array([label], dtype=np.uint8))
        
        return (batches, labels)

class DaliIterator():
    """
    Wrapper class to decode the DALI iterator output & provide iterator that functions the same as torchvision
    pipelines (Pipeline): DALI pipelines
    size (int): Number of examples in set
    Note: allow extra inputs to keep compatibility with CPU iterator
    """

    def __init__(self, pipelines, size, **kwargs):
        self._dali_iterator = DALIClassificationIterator(pipelines=pipelines, size=size)

    def __iter__(self):
        return self

    def __len__(self):
        return int(math.ceil(self._dali_iterator._size / self._dali_iterator.batch_size))


class DaliIteratorGPU(DaliIterator):
    """
    Wrapper class to decode the DALI iterator output & provide iterator that functions the same as torchvision
    pipelines (Pipeline): DALI pipelines
    size (int): Number of examples in set
    Note: allow extra inputs to keep compatibility with CPU iterator
    """

    def __next__(self):
        try:
            data = next(self._dali_iterator)
        except StopIteration:
            print('Resetting DALI loader')
            self._dali_iterator.reset()
            raise StopIteration

        # Decode the data output
        input = data[0]['data']
        target = data[0]['label'].squeeze().long()

        return input, target

# Background task send process
def task_sender(in_queue, sampler, done_event, stop_iteration_event):
    sampler_iter = iter(sampler)
    try:
        while not done_event.is_set():
            if in_queue.qsize() < 2:
                try:
                    index = next(sampler_iter)
                except StopIteration:
                    stop_iteration_event.set()
                    in_queue.put([0])
                    in_queue.put([0])
                    sampler_iter = iter(sampler)
                    continue
                    # break
                in_queue.put(index)
                # in_queue.put([0])
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass

to_torch_type = {
    types.DALIDataType.FLOAT:   torch.float32,
    types.DALIDataType.FLOAT64: torch.float64,
    types.DALIDataType.FLOAT16: torch.float16,
    types.DALIDataType.UINT8:   torch.uint8,
    types.DALIDataType.INT8:    torch.int8,
    types.DALIDataType.INT16:   torch.int16,
    types.DALIDataType.INT32:   torch.int32,
    types.DALIDataType.INT64:   torch.int64
}


def feed_ndarray(dali_tensor, arr, cuda_stream=None):
    """
    Copy contents of DALI tensor to PyTorch's Tensor.

    Parameters
    ----------
    `dali_tensor` : nvidia.dali.backend.TensorCPU or nvidia.dali.backend.TensorGPU
                    Tensor from which to copy
    `arr` : torch.Tensor
            Destination of the copy
    `cuda_stream` : torch.cuda.Stream, cudaStream_t or any value that can be cast to cudaStream_t.
                    CUDA stream to be used for the copy
                    (if not provided, an internal user stream will be selected)
                    In most cases, using pytorch's current stream is expected (for example,
                    if we are copying to a tensor allocated with torch.zeros(...))
    """
    dali_type = to_torch_type[dali_tensor.dtype]

    assert dali_type == arr.dtype, ("The element type of DALI Tensor/TensorList"
                                    " doesn't match the element type of the target PyTorch Tensor: "
                                    "{} vs {}".format(dali_type, arr.dtype))
    assert dali_tensor.shape() == list(arr.size()), \
        ("Shapes do not match: DALI tensor has size {0}, but PyTorch Tensor has size {1}".
            format(dali_tensor.shape(), list(arr.size())))
    cuda_stream = types._raw_cuda_stream(cuda_stream)

    # turn raw int to a c void pointer
    c_type_pointer = ctypes.c_void_p(arr.data_ptr())
    if isinstance(dali_tensor, (TensorGPU, TensorListGPU)):
        stream = None if cuda_stream is None else ctypes.c_void_p(cuda_stream)
        dali_tensor.copy_to_external(c_type_pointer, stream, non_blocking=True)
    else:
        dali_tensor.copy_to_external(c_type_pointer)
    return arr


class DALIGenericIterator(_DaliBaseIterator):
    """
    General DALI iterator for PyTorch. It can return any number of
    outputs from the DALI pipeline in the form of PyTorch's Tensors.

    Parameters
    ----------
    pipelines : list of nvidia.dali.Pipeline
                List of pipelines to use
    output_map : list of str
                List of strings which maps consecutive outputs
                of DALI pipelines to user specified name.
                Outputs will be returned from iterator as dictionary
                of those names.
                Each name should be distinct
    size : int, default = -1
                Number of samples in the shard for the wrapped pipeline (if there is more than
                one it is a sum)
                Providing -1 means that the iterator will work until StopIteration is raised
                from the inside of iter_setup(). The options `last_batch_policy` and
                `last_batch_padded` don't work in such case. It works with only one pipeline inside
                the iterator.
                Mutually exclusive with `reader_name` argument
    reader_name : str, default = None
                Name of the reader which will be queried to the shard size, number of shards and
                all other properties necessary to count properly the number of relevant and padded
                samples that iterator needs to deal with. It automatically sets `last_batch_policy`
                to PARTIAL when the FILL is used, and `last_batch_padded` accordingly to match
                the reader's configuration
    auto_reset : string or bool, optional, default = False
                Whether the iterator resets itself for the next epoch or it requires reset() to be
                called explicitly.

                It can be one of the following values:

                * ``"no"``, ``False`` or ``None`` - at the end of epoch StopIteration is raised
                  and reset() needs to be called
                * ``"yes"`` or ``"True"``- at the end of epoch StopIteration is raised but reset()
                  is called internally automatically

    dynamic_shape : any, optional,
                Parameter used only for backward compatibility.
    fill_last_batch : bool, optional, default = None
                **Deprecated** Please use ``last_batch_policy`` instead

                Whether to fill the last batch with data up to 'self.batch_size'.
                The iterator would return the first integer multiple
                of self._num_gpus * self.batch_size entries which exceeds 'size'.
                Setting this flag to False will cause the iterator to return
                exactly 'size' entries.
    last_batch_policy: optional, default = LastBatchPolicy.FILL
                What to do with the last batch when there are not enough samples in the epoch
                to fully fill it. See :meth:`nvidia.dali.plugin.base_iterator.LastBatchPolicy`
    last_batch_padded : bool, optional, default = False
                Whether the last batch provided by DALI is padded with the last sample
                or it just wraps up. In the conjunction with ``last_batch_policy`` it tells
                if the iterator returning last batch with data only partially filled with
                data from the current epoch is dropping padding samples or samples from
                the next epoch. If set to ``False`` next
                epoch will end sooner as data from it was consumed but dropped. If set to
                True next epoch would be the same length as the first one. For this to happen,
                the option `pad_last_batch` in the reader needs to be set to True as well.
                It is overwritten when `reader_name` argument is provided
    prepare_first_batch : bool, optional, default = True
                Whether DALI should buffer the first batch right after the creation of the iterator,
                so one batch is already prepared when the iterator is prompted for the data

    Example
    -------
    With the data set ``[1,2,3,4,5,6,7]`` and the batch size 2:

    last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = True  -> last batch = ``[7]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = False -> last batch = ``[7]``,
    next iteration will return ``[2, 3]``

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = True   -> last batch = ``[7, 7]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = False  -> last batch = ``[7, 1]``,
    next iteration will return ``[2, 3]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = True   -> last batch = ``[5, 6]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = False  -> last batch = ``[5, 6]``,
    next iteration will return ``[2, 3]``
    """

    def __init__(self,
                 pipelines,
                 output_map,
                 size=-1,
                 reader_name=None,
                 auto_reset=False,
                 fill_last_batch=None,
                 dynamic_shape=False,
                 last_batch_padded=False,
                 last_batch_policy=LastBatchPolicy.FILL,
                 prepare_first_batch=True):

        # check the assert first as _DaliBaseIterator would run the prefetch
        assert len(set(output_map)) == len(output_map), "output_map names should be distinct"
        self._output_categories = set(output_map)
        self.output_map = output_map

        _DaliBaseIterator.__init__(self,
                                   pipelines,
                                   size,
                                   reader_name,
                                   auto_reset,
                                   fill_last_batch,
                                   last_batch_padded,
                                   last_batch_policy,
                                   prepare_first_batch=prepare_first_batch)

        self._first_batch = None
        if self._prepare_first_batch:
            try:
                self._first_batch = DALIGenericIterator.__next__(self)
                # call to `next` sets _ever_consumed to True but if we are just calling it from
                # here we should set if to False again
                self._ever_consumed = False
            except StopIteration:
                assert False, "It seems that there is no data in the pipeline. This may happen " \
                       "if `last_batch_policy` is set to PARTIAL and the requested batch size is " \
                       "greater than the shard size."

    def __next__(self):
        self._ever_consumed = True
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch

        # Gather outputs
        outputs = self._get_outputs()

        data_batches = [None for i in range(self._num_gpus)]
        for i in range(self._num_gpus):
            dev_id = self._pipes[i].device_id
            # initialize dict for all output categories
            category_outputs = dict()
            # segregate outputs into categories
            for j, out in enumerate(outputs[i]):
                category_outputs[self.output_map[j]] = out

            # Change DALI TensorLists into Tensors
            category_tensors = dict()
            category_shapes = dict()
            for category, out in category_outputs.items():
                category_tensors[category] = out.as_tensor()
                category_shapes[category] = category_tensors[category].shape()

            category_torch_type = dict()
            category_device = dict()
            torch_gpu_device = None
            torch_cpu_device = torch.device('cpu')
            # check category and device
            for category in self._output_categories:
                category_torch_type[category] = to_torch_type[category_tensors[category].dtype]
                if type(category_tensors[category]) is TensorGPU:
                    if not torch_gpu_device:
                        torch_gpu_device = torch.device('cuda', dev_id)
                    category_device[category] = torch_gpu_device
                else:
                    category_device[category] = torch_cpu_device

            pyt_tensors = dict()
            for category in self._output_categories:
                pyt_tensors[category] = torch.empty(category_shapes[category],
                                                    dtype=category_torch_type[category],
                                                    device=category_device[category])

            data_batches[i] = pyt_tensors

            # Copy data from DALI Tensors to torch tensors
            for category, tensor in category_tensors.items():
                if isinstance(tensor, (TensorGPU, TensorListGPU)):
                    # Using same cuda_stream used by torch.zeros to set the memory
                    stream = torch.cuda.current_stream(device=pyt_tensors[category].device)
                    feed_ndarray(tensor, pyt_tensors[category], cuda_stream=stream)
                else:
                    feed_ndarray(tensor, pyt_tensors[category])

        self._schedule_runs()

        self._advance_and_check_drop_last()

        if self._reader_name:
            if_drop, left = self._remove_padded()
            if np.any(if_drop):
                output = []
                for batch, to_copy in zip(data_batches, left):
                    batch = batch.copy()
                    for category in self._output_categories:
                        batch[category] = batch[category][0:to_copy]
                    output.append(batch)
                return output

        else:
            if self._last_batch_policy == LastBatchPolicy.PARTIAL and (
                                          self._counter > self._size) and self._size > 0:
                # First calculate how much data is required to return exactly self._size entries.
                diff = self._num_gpus * self.batch_size - (self._counter - self._size)
                # Figure out how many GPUs to grab from.
                numGPUs_tograb = int(np.ceil(diff / self.batch_size))
                # Figure out how many results to grab from the last GPU
                # (as a fractional GPU batch may be required to bring us
                # right up to self._size).
                mod_diff = diff % self.batch_size
                data_fromlastGPU = mod_diff if mod_diff else self.batch_size

                # Grab the relevant data.
                # 1) Grab everything from the relevant GPUs.
                # 2) Grab the right data from the last GPU.
                # 3) Append data together correctly and return.
                output = data_batches[0:numGPUs_tograb]
                output[-1] = output[-1].copy()
                for category in self._output_categories:
                    output[-1][category] = output[-1][category][0:data_fromlastGPU]
                return output

        return data_batches


class DALIClassificationIterator(DALIGenericIterator):
    """
    DALI iterator for classification tasks for PyTorch. It returns 2 outputs
    (data and label) in the form of PyTorch's Tensor.

    Calling

    .. code-block:: python

       DALIClassificationIterator(pipelines, reader_name)

    is equivalent to calling

    .. code-block:: python

       DALIGenericIterator(pipelines, ["data", "label"], reader_name)

    Parameters
    ----------
    pipelines : list of nvidia.dali.Pipeline
                List of pipelines to use
    size : int, default = -1
                Number of samples in the shard for the wrapped pipeline (if there is more than
                one it is a sum)
                Providing -1 means that the iterator will work until StopIteration is raised
                from the inside of iter_setup(). The options `last_batch_policy` and
                `last_batch_padded` don't work in such case. It works with only one pipeline inside
                the iterator.
                Mutually exclusive with `reader_name` argument
    reader_name : str, default = None
                Name of the reader which will be queried to the shard size, number of shards and
                all other properties necessary to count properly the number of relevant and padded
                samples that iterator needs to deal with. It automatically sets `last_batch_policy`
                to PARTIAL when the FILL is used, and `last_batch_padded` accordingly to match
                the reader's configuration
    auto_reset : string or bool, optional, default = False
                Whether the iterator resets itself for the next epoch or it requires reset() to be
                called explicitly.

                It can be one of the following values:

                * ``"no"``, ``False`` or ``None`` - at the end of epoch StopIteration is raised
                  and reset() needs to be called
                * ``"yes"`` or ``"True"``- at the end of epoch StopIteration is raised but reset()
                  is called internally automatically

    dynamic_shape : any, optional,
                Parameter used only for backward compatibility.
    fill_last_batch : bool, optional, default = None
                **Deprecated** Please use ``last_batch_policy`` instead

                Whether to fill the last batch with data up to 'self.batch_size'.
                The iterator would return the first integer multiple
                of self._num_gpus * self.batch_size entries which exceeds 'size'.
                Setting this flag to False will cause the iterator to return
                exactly 'size' entries.
    last_batch_policy: optional, default = LastBatchPolicy.FILL
                What to do with the last batch when there are not enough samples in the epoch
                to fully fill it. See :meth:`nvidia.dali.plugin.base_iterator.LastBatchPolicy`
    last_batch_padded : bool, optional, default = False
                Whether the last batch provided by DALI is padded with the last sample
                or it just wraps up. In the conjunction with ``last_batch_policy`` it tells
                if the iterator returning last batch with data only partially filled with
                data from the current epoch is dropping padding samples or samples from
                the next epoch. If set to ``False`` next
                epoch will end sooner as data from it was consumed but dropped. If set to
                True next epoch would be the same length as the first one. For this to happen,
                the option `pad_last_batch` in the reader needs to be set to True as well.
                It is overwritten when `reader_name` argument is provided
    prepare_first_batch : bool, optional, default = True
                Whether DALI should buffer the first batch right after the creation of the iterator,
                so one batch is already prepared when the iterator is prompted for the data

    Example
    -------
    With the data set ``[1,2,3,4,5,6,7]`` and the batch size 2:

    last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = True  -> last batch = ``[7]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = False -> last batch = ``[7]``,
    next iteration will return ``[2, 3]``

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = True   -> last batch = ``[7, 7]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = False  -> last batch = ``[7, 1]``,
    next iteration will return ``[2, 3]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = True   -> last batch = ``[5, 6]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = False  -> last batch = ``[5, 6]``,
    next iteration will return ``[2, 3]``
    """

    def __init__(self,
                 pipelines,
                 size=-1,
                 reader_name=None,
                 auto_reset=True,
                 fill_last_batch=None,
                 dynamic_shape=False,
                 last_batch_padded=False,
                 last_batch_policy=LastBatchPolicy.FILL,
                 prepare_first_batch=True):
        super(DALIClassificationIterator, self).__init__(pipelines, ["data", "label"],
                                                         size,
                                                         reader_name=reader_name,
                                                         auto_reset=auto_reset,
                                                         fill_last_batch=fill_last_batch,
                                                         dynamic_shape=dynamic_shape,
                                                         last_batch_padded=last_batch_padded,
                                                         last_batch_policy=last_batch_policy,
                                                         prepare_first_batch=prepare_first_batch)




class DaliMicroBatchLoader():
    """
    Only for classification tasks
    Wrapper class to load the data with microbatch for supporting RandAugment
    pipelines (Pipeline): DALI pipelines
    size (int): Number of examples in set
    Note: allow extra inputs to keep compatibility with CPU iterator
    """

    def __init__(self, dataset, device_id, template_shape, microbatch_size, minibatch_size,
                 size=224,
                 val_batch_size=None,
                 val_size=256,
                 min_crop_size=0.08,
                 workers=1,
                 world_size=1,
                 cuda=True,
                 use_dali=True,
                 dali_cpu=False,
                 fp16=False,
                 mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
                 std=(0.229 * 255, 0.224 * 255, 0.225 * 255),
                 pin_memory=True,
                 pin_memory_dali=False,
                 rand_factor=[2, 9],
                 aug_type = "default",
                 generator=None,
                 sampler = None,
                 path=None):

        self.size = size
        self.min_crop_size = min_crop_size
        self.workers = workers
        self.world_size = world_size
        self.cuda = cuda
        self.use_dali = use_dali
        self.dali_cpu = dali_cpu
        self.fp16 = fp16
        self.mean = mean
        self.std = std
        self.pin_memory = pin_memory
        self.pin_memory_dali = pin_memory_dali
        self.num_of_ops = rand_factor[0]
        self.degree_of_ops = rand_factor[1]
        self.device_id = device_id
        self.path = path
        print(f"device_id: {device_id}")

        self.augment_type = aug_type
        self.in_queue = multiprocessing.Queue()
        self.dataset = dataset
        if sampler == None:
            self.sampler = RandomSampler(self.dataset)
        else:
            self.sampler = sampler
        self.microbatch_size = microbatch_size
        self.minibatch_size = minibatch_size
        self.micro_to_mini = self.minibatch_size // self.microbatch_size
        if __debug__:
            log.debug(f"Minibatch: {self.micro_to_mini}")
        if self.augment_type == "randaugment":
            self.batch_sampler = FinegrainedBatchSampler(batch_size = microbatch_size, sampler = self.sampler, actual_batch_size = minibatch_size, drop_last = True)
        else:
            self.batch_sampler = BatchSampler(batch_size = minibatch_size, sampler = self.sampler, drop_last = True)
        self.done_event = multiprocessing.Event()
        self.stop_iteration_event = multiprocessing.Event()

        self.task_sender_process = multiprocessing.Process(
            target=task_sender,
            args=(self.in_queue, self.batch_sampler, self.done_event, self.stop_iteration_event)
         )
        self.task_sender_process.daemon = True
        self.task_sender_process.start()

        if __debug__:
            log.debug(f"workers: {self.workers}")

        self.template_shape = template_shape
        self.input_template = self.template_shape
        self.label_template = (minibatch_size)
        self.iter_counter = 0
        if self.augment_type == "randaugment":
            # self.ops_combinations=list(combinations(DALI_RANDAUGMENT_LIST, self.num_of_ops))[0]
            self.ops_combinations=list(combinations(DALI_RANDAUGMENT_LIST, self.num_of_ops))
        else:
            self.ops_combinations=["default"]
        if __debug__:
            log.debug(f"DALI GPU PRocessor: aug type {self.augment_type}")
        self._build_dali_pipeline()
        
        # FIXME: hardcode type
        self.data_type = torch.float32
        self.label_type = torch.uint8
        self.final_type = torch.half if self.fp16 else torch.long
        # self.batch_tensor = torch.empty((self.microbatch_size, self.template_shape[1], self.template_shape[2]), dtype=self.data_type)
        # self.label_tensor = torch.empty((self.microbatch_size), dtype=self.label_type)

    def __len__(self):
        return len(self.batch_sampler)

    def feed_ndarray(self, dali_tensor, arr, cuda_stream = None, arr_data_ptr = None):
        """
        Snippet from https://github.com/NVIDIA/DALI/blob/main/dali/python/nvidia/dali/plugin/pytorch.py
        Modify for aggregation 

        Copy contents of DALI tensor to PyTorch's Tensor.
        Parameters
        ----------
        `dali_tensor` : nvidia.dali.backend.TensorCPU or nvidia.dali.backend.TensorGPU
                        Tensor from which to copy
        `arr` : torch.Tensor
                Destination of the copy
        `cuda_stream` : torch.cuda.Stream, cudaStream_t or any value that can be cast to cudaStream_t.
                        CUDA stream to be used for the copy
                        (if not provided, an internal user stream will be selected)
                        In most cases, using pytorch's current stream is expected (for example,
                        if we are copying to a tensor allocated with torch.zeros(...))
        """
        if isinstance(dali_tensor, (TensorListCPU, TensorListGPU)):
            dali_type = dali_tensor[0].dtype()
        else:
            dali_type = dali_tensor.dtype()
        # dali_type = np.dtype(dali_type)
        if __debug__:
            log.debug(f"Aggregator: type DALI tensor {dali_type} Arr {arr.dtype}")
        # assert to_torch_type[dali_type] == arr.dtype, ("The element type of DALI Tensor/TensorList"
                # " doesn't match the element type of the target PyTorch Tensor: {} vs {}".format(to_torch_type[dali_type], arr.dtype))
        if __debug__:
            log.debug(f"Aggregator: shape DALI tensor  {dali_tensor.shape()} Arr {arr.size()}")
        # assert dali_tensor.shape() == list(arr.size()), \
                # ("Shapes do not match: DALI tensor has size {0}"
                # ", but PyTorch Tensor has size {1}".format(dali_tensor.shape(), list(arr.size())))
        cuda_stream = types._raw_cuda_stream(cuda_stream)

        # turn raw int to a c void pointer
        if arr_data_ptr == None:
            arr_data_ptr = arr.data_ptr()
        

        c_type_pointer = ctypes.c_void_p(arr_data_ptr)
        if __debug__:
            log.debug(f"Aggregator: arr ptr: {arr_data_ptr}")
        if isinstance(dali_tensor, (TensorGPU, TensorListGPU)):
            dali_tensor.copy_to_external(c_type_pointer, None if cuda_stream is None else ctypes.c_void_p(cuda_stream))
        else:
            dali_tensor.copy_to_external(c_type_pointer)
        return arr
            
    def __next__(self):
        torch.cuda.set_device(self.device_id)
        # Hardcoded data type as float32 for dali datatype
        stream = torch.cuda.current_stream(device=self.device_id)
        if self.augment_type == "randaugment":
            data_images = torch.empty(self.input_template,
                        dtype=self.data_type,
                        device=self.device_id)
            
            # Hardcoded data type as uint8 for dali datatype
            data_labels = torch.empty(self.label_template,
                        dtype=self.label_type,
                        device=self.device_id)
            self.iter_counter = 0

            for _ in range(self.micro_to_mini):
                if self.stop_iteration_event.is_set():
                    self.stop_iteration_event.clear()
                    raise StopIteration
                
                pipe_num = random.randrange(self.get_max_select())
                one_time_gpu_pipe = self.get_pipe(pipe_num)

                while True:
                    if __debug__:
                        # log.debug("Randaug build start")
                        start = time.perf_counter()
                    _input, _target = one_time_gpu_pipe.run()
                    if __debug__:
                        end = time.perf_counter()
                        log.debug(f"GPU worker FETCH_TIME {end - start}")
                    if len(_input) == 1:
                        continue
                    if __debug__:
                        log.debug(f"test0 {len(_input)}, {len(_target)}")
                        # log.debug(f"input: {_input}, _target: {_target}")
                    break

                if __debug__:
                    log.debug(f"iter_counter {self.iter_counter}")

                _input_tensor = _input.as_tensor()
                self.feed_ndarray(_input_tensor, data_images, stream, data_images[self.iter_counter].data_ptr())
                _target_tensor = _target.as_tensor()
                self.feed_ndarray(_target_tensor, data_labels, stream, data_labels[self.iter_counter].data_ptr())
                del _input, _target, _input_tensor, _target_tensor
                # if __debug__:
                #     log.debug(f"_input: {_input}, target: {_target}")
                self.iter_counter += self.microbatch_size
                
            if __debug__:
                log.debug(f"measurement: data_images{data_images}, data_labels{data_labels} torch allocation{torch.cuda.memory_allocated()} torch reserved{torch.cuda.memory_reserved()}")
            
            

            # data_images = data_images.type(self.final_type)
            data_labels = data_labels.type(torch.long)
            if __debug__:
                log.debug(f"measurement: data_images{data_images.device}, data_labels{data_labels.device}")
            return data_images, data_labels
        else:
            if self.stop_iteration_event.is_set():
                self.stop_iteration_event.clear()
                raise StopIteration

            while True:
                images, label = next(self.train_iter)
                if label.shape == torch.Size([]):
                    del images, label
                    continue
                break
            return images, label
    
        

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.batch_sampler)
    
    def _shutdown(self):
        self.done_event.set()
        self.task_sender_process.join(timeout=5)
        if self.task_sender_process.is_alive():
            self.task_sender_process.terminate()

        self.in_queue.cancel_join_thread()
        self.in_queue.close()
        
    def __del__(self):
        self._shutdown()

import torch.utils.data as data

class DALIDataLoader(data.DataLoader):
    def __init__(self, pipeline, batch_size, num_threads, device_id):
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.device_id = device_id
        self.iterator = None
        super().__init__(dataset=None, batch_size=None, shuffle=False)

    def __iter__(self):
        self.iterator = self.pipeline.run()
        return self

    def __next__(self):
        output = self.iterator.next()
        output = [torch.from_numpy(out) for out in output]
        return output

