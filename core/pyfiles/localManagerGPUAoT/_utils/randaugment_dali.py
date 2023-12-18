from . import operation_offloading as CO
from . import USE_MONOLITH_PIPE, USE_SHARED_MEMORY, DEALLOC_GPU_MEMORY_AFTER_OP

# import pynvml
import torch
import time
import numpy as np
import torch
import os
import sys
from itertools import combinations
from torch.multiprocessing import Event
# from torch._six import queue
# PyTorch 2.0 Fix
import queue
from posix_ipc import O_CREAT, SharedMemory, unlink_shared_memory
import mmap
from nvidia.dali.backend import TensorGPU, TensorListGPU, TensorListCPU
import ctypes
from random import shuffle

from time import perf_counter as timer_func
import psutil
import random

try:
    from nvidia import dali
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn

    from nvidia.dali.plugin.base_iterator import _DaliBaseIterator
    from nvidia.dali.plugin.base_iterator import LastBatchPolicy

    import torch
    import torch.utils.dlpack as torch_dlpack
    from nvidia.dali.auto_aug import rand_augment
    from nvidia.dali.pipeline import experimental      

except ImportError:
    raise ImportError(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run.")

if __debug__:
    import logging
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger(__name__)
   

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

class ManagerWatchdog(object):  # type: ignore[no-redef]
    def __init__(self):
        self.manager_pid = os.getppid()
        self.manager_dead = False

    def is_alive(self):
        if not self.manager_dead:
            self.manager_dead = os.getppid() != self.manager_pid
        return not self.manager_dead


class ExternalInputGpuIterator(object):
    '''
    The external source operator can also accept GPU data from CuPy or any other data source
    that supports the cuda array interface. For this example, we create the ExternalInputGpuIterator
    that returns data on the GPU. Since decoders.image does not accept data on the GPU we need to
    decode it outside DALI on the CPU and then move it to the GPU. Typically, because of the
    operation of another library, the image; or other data will already be on the GPU.
    '''

    def __init__(self, dataset, batch_size, from_file=True):
        self.batch_size = batch_size
        self.dataset = dataset
        self.in_queue = []
        self.data_path_list = None
        self.data_label_list = None
        self.files = []
        self.from_file = from_file

    def append(self, files):
        self.in_queue.append(files)

    def set_files(self):
        if __debug__:
            log.debug(f"ExternalInput: set_Files wait for queue")
        data_list = self.in_queue.pop(0)
        if __debug__:
            log.debug(f"ExternalInput: set_dataset as {data_list}")
        self.files = data_list

    def get_gpu_batches(self):
        if __debug__:
            log.debug(f"ExternalInput: get_gpu_batch wait for queue")
        batches, labels = self.in_queue.pop(0)
        if __debug__:
            log.debug(f"ExternalInput: get labels")
        return batches, labels

    def __iter__(self):
        return self

    def __next__(self):
        if self.from_file:
            self.set_files()

            batches = []
            labels = []

            if __debug__:
                log.debug(f"ExternalInput: run pipe {self.files}")

            for index in self.files:
                jpeg_file, label = self.dataset.samples[index]
                f = open(jpeg_file, 'rb')
                batches.append(np.frombuffer(f.read(), dtype=np.uint8))
                labels.append(np.array([label], dtype=np.uint8))
        else:
            batches, labels = self.get_gpu_batches()

        return (batches, labels)


class ExternalPrepQueueIterator(object):
    '''
    External source will be Prep Queue.
    '''

    def __init__(self):
        self.in_queue = []

    def append(self, images, labels):
        self.in_queue.append((images, labels))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            # images, labels = self.in_queue.get()
            images, labels = self.in_queue.pop(0)
        except BaseException as e:
            print("Impossible situation!", flush=True)
            print("This is going to cause error in program", flush=True)
            while True:
                i = 12
                j = 132
                i = i ^ j
                j = j ^ i
                i = i ^ j 
        return images, labels


def _try_get_queue(in_queue):
    try:
        if in_queue.empty():
            return -1
        idx = in_queue.get()
    except queue.Empty:
        idx = -1
    return idx

def _load_np_arr(type, rank, worker_id, frame, dtype=np.int32):
    shm_name = f'{type}{rank}_{int(worker_id)}'
    shape = frame.shape
    size = frame.nbytes
    if __debug__:
        log.debug(f"Worker load as {shm_name} size: {size}")
    print(f"Worker load as {shm_name} size: {size}")
    shm = SharedMemory(name=shm_name)
    shm_buf = mmap.mmap(shm.fd, size)
    shm.close_fd()
    shm_np_arr = np.ndarray(
        shape=shape, dtype=dtype, buffer=shm_buf)

    return shm_buf, shm_np_arr


def feed_ndarray(dali_tensor, arr, cuda_stream = None, arr_data_ptr = None):
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
    # dali_type = to_torch_type[dali_tensor.dtype]
    if isinstance(dali_tensor, (TensorListCPU, TensorListGPU)):
        dali_type = dali_tensor[0].dtype()
    else:
        dali_type = dali_tensor.dtype
    # dali_type = np.dtype(dali_type)
    # dali_type = np.asarray(dali_type, dtype=np.float64)
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
    # assert dali_type == arr.dtype, ("The element type of DALI Tensor/TensorList"
    #                                 " doesn't match the element type of the target PyTorch Tensor: "
    #                                 "{} vs {}".format(dali_type, arr.dtype))
    # assert dali_tensor.shape() == list(arr.size()), \
    #     ("Shapes do not match: DALI tensor has size {0}, but PyTorch Tensor has size {1}".
    #         format(dali_tensor.shape(), list(arr.size())))
    # cuda_stream = types._raw_cuda_stream(cuda_stream)

    # turn raw int to a c void pointer
    c_type_pointer = ctypes.c_void_p(arr.data_ptr())
    if __debug__:
        log.debug(f"Aggregator: arr ptr: {arr_data_ptr}")
    if isinstance(dali_tensor, (TensorGPU, TensorListGPU)):
        stream = None if cuda_stream is None else ctypes.c_void_p(cuda_stream)
        dali_tensor.copy_to_external(c_type_pointer, stream, non_blocking=True)
    else:
        dali_tensor.copy_to_external(c_type_pointer)
    return arr


def make_pipe(op_name):
    func = None
    if op_name == "Decode":
        @experimental.pipeline_def()    
        def decode_pipeline(eii_source):
            jpegs, labels = dali.fn.external_source(
                    source=eii_source, num_outputs=2, device="cpu")
            images = fn.decoders.image(jpegs, device="mixed")
            return images, labels.gpu()
        return decode_pipeline
    elif op_name == "RandAugment_Auto":
        @experimental.pipeline_def(enable_conditionals=False)            
        def randaug_pipeline(eii_source):
            images, labels = dali.fn.external_source(
                    source=eii_source, num_outputs=2, device="gpu", layout=["HWC", None])
            images = rand_augment.rand_augment(images, n=2, m=9)
            return images, labels
        return randaug_pipeline
    elif op_name == "RandAugment_Manual":
        @experimental.pipeline_def(enable_conditionals=False)            
        def aug_pipeline(eii_source, aug):
            images, labels = dali.fn.external_source(
                    source=eii_source, num_outputs=2, device="gpu", layout=["HWC", None])
            images = aug(images)
            images = dali.fn.cast(images, dtype=types.DALIDataType.UINT8)
            return images, labels
        return aug_pipeline
    elif op_name == "AutoAugment_Auto":
        @experimental.pipeline_def(enable_conditionals=True)            
        def aug_pipeline(eii_source, augs):
            images, labels = dali.fn.external_source(
                    source=eii_source, num_outputs=2, device="gpu", layout=["HWC", None])
            for prob, aug in augs:
                should_apply = fn.random.coin_flip(probability=prob, seed=42, dtype=types.BOOL)
                if should_apply:
                    images = aug(images)
                    images = dali.fn.cast(images, dtype=types.DALIDataType.UINT8)
            return images, labels
        return aug_pipeline
    elif op_name == "RandomCrop":
        @experimental.pipeline_def(enable_conditionals=False)
        def resized_crop_pipeline(eii_source, crop=224):
            images, labels = dali.fn.external_source(
                    source=eii_source, num_outputs=2, device="gpu", layout=["HWC", None])
            images = fn.random_resized_crop(images, size=[crop, crop], minibatch_size=1)
            return images, labels
        return resized_crop_pipeline
    elif op_name == "FlipAndNormalize":
        mean=(0.485 * 255, 0.456 * 255, 0.406 * 255)
        std=(0.229 * 255, 0.224 * 255, 0.225 * 255)
        @experimental.pipeline_def(enable_conditionals=False)
        def mirror_norm_pipeline(eii_source):
            rng = fn.random.coin_flip(probability=0.5)
            images, labels = dali.fn.external_source(
                    source=eii_source, num_outputs=2, device="gpu", layout=["HWC", None])
            images = fn.crop_mirror_normalize(images,
                                            mirror=rng,
                                            mean=mean,
                                            std=std,
                                            dtype=types.DALIDataType.FLOAT,
                                            output_layout=types.NCHW)
            return images, labels
        return mirror_norm_pipeline
    return func

def make_monolith_pipe(augname):
    mean=(0.485 * 255, 0.456 * 255, 0.406 * 255)
    std=(0.229 * 255, 0.224 * 255, 0.225 * 255)
    print("Choosing", augname, "for GPU data pipeline", flush=True)
    if augname == 'randaugment':
        @experimental.pipeline_def(enable_conditionals=True)            
        def monolith_pipeline(eii_source, crop=224):
            rng = fn.random.coin_flip(probability=0.5)

            jpegs, labels = dali.fn.external_source(
                    source=eii_source, num_outputs=2, device="cpu")
            images = fn.decoders.image(jpegs, device="mixed")
            images = rand_augment.rand_augment(images, n=2, m=9)
            images = fn.random_resized_crop(images, size=[crop, crop], minibatch_size=1)
            output = fn.crop_mirror_normalize(images,
                                                mirror=rng,
                                                mean=mean,
                                                std=std,
                                                dtype=types.DALIDataType.FLOAT,
                                                output_layout=types.NCHW)

            labels = labels.gpu()
            return output, labels
        return monolith_pipeline
    elif augname == 'resize_randaug':
        @experimental.pipeline_def(enable_conditionals=True)            
        def monolith_pipeline(eii_source, crop=224):
            rng = fn.random.coin_flip(probability=0.5)

            jpegs, labels = dali.fn.external_source(
                    source=eii_source, num_outputs=2, device="cpu")
            images = fn.decoders.image(jpegs, device="mixed")
            images = fn.random_resized_crop(images, size=[crop, crop], minibatch_size=1)
            images = rand_augment.rand_augment(images, n=2, m=9)
            output = fn.crop_mirror_normalize(images,
                                                mirror=rng,
                                                mean=mean,
                                                std=std,
                                                dtype=types.DALIDataType.FLOAT,
                                                output_layout=types.NCHW)

            labels = labels.gpu()
            return output, labels
        return monolith_pipeline

def clear_traced_pipes(trace, pipes, eii_source, _rank):
    '''Clears pipes that are in trace list!'''

    # I changed Pipeline's logic to fit the last data size
    # Small Data -> Fit the Pipe -> Small Pipe
    print("Clearing Pipelines: Start", flush=True)
    for op_name in trace:
        if type(pipes[op_name]) is not list:
            if op_name == "Decode":
                eii_source[0].append([0])
            else:
                _dummy_input = torch.zeros(1, 1, 3, dtype=torch.uint8, device=_rank)
                _dummy_target = torch.zeros(1, dtype=torch.uint8, device=_rank)
                eii_source[1].append([_dummy_input], [_dummy_target])
            pipes[op_name].run()
        else:
            for _, pipe in pipes[op_name]:
                _dummy_input = torch.zeros(1, 1, 3, dtype=torch.uint8, device=_rank)
                _dummy_target = torch.zeros(1, dtype=torch.uint8, device=_rank)
                eii_source[1].append([_dummy_input], [_dummy_target])
                pipe.run()
    dali.backend.ReleaseUnusedMemory()
    print("Clearing Pipelines: End", flush=True)

def augmentation_func(op, minval, maxval, aug_factor):
    if op == "AutoContrast":
        def auto_contrast(data):
            # assumes HWC layout
            lo, hi = fn.reductions.min(data, axes=[0, 1]), fn.reductions.max(data, axes=[0, 1])
            diff = hi - lo
            mask_scale = diff > 0
            mask_id = mask_scale ^ True
            # choose div so that scale ends up being 255 / diff if diff > 0 and 1 otherwise
            div_by = diff * mask_scale + types.Constant(255, dtype=types.UINT8) * mask_id
            scale = 255 / div_by
            scaled = (data - lo * mask_scale) * scale
            return fn.cast_like(scaled, data)
        wrap = lambda images: auto_contrast(images)
    elif op == "Equalize":
        wrap = lambda images: fn.experimental.equalize(images)
    elif op == "Posterize":
        val = (float(aug_factor) / 30) * float(maxval - minval) + minval
        val = max(int(val), 1)
        def posterize(data, mask):
            assert 0 <= mask and mask <= 4
            return data & mask
        # wrap = lambda images: dali.auto_aug.augmentations.posterize(images, mask=val)
        wrap = lambda images: posterize(images, mask=val)
    elif op == "Solarize":
        val = (float(aug_factor) / 30) * float(maxval - minval) + minval
        def solarize(data, threshold):
            sample_inv = types.Constant(255, dtype=types.UINT8) - data
            mask_unchanged = data < threshold
            mask_inverted = mask_unchanged ^ True
            return mask_unchanged * data + mask_inverted * sample_inv
        # wrap = lambda images: dali.auto_aug.augmentations.solarize(images, threshold=val)
        wrap = lambda images: solarize(images, threshold=val)
    elif op == "SolarizeAdd":
        val = (float(aug_factor) / 30) * float(maxval - minval) + minval
        def solarize_add(data, shift):
            mask_shifted = data < types.Constant(128, dtype=types.UINT8)
            mask_id = mask_shifted ^ True
            sample_shifted = data + shift
            return mask_shifted * sample_shifted + mask_id * data
        # wrap = lambda images: dali.auto_aug.augmentations.solarize_add(images, shift=val)
        wrap = lambda images: solarize_add(images, shift=val)
    elif op == "Color":
        val = (float(aug_factor) / 30) * float(maxval - minval) + minval
        wrap = lambda images: fn.saturation(images, saturation=val)
    elif op == "Sharpness":
        def sharpness_kernel(magnitude):
            # assumes magnitude: [-1, 1]
            blur = np.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=np.float32) / 13
            ident = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
            return -magnitude * blur + (1 + magnitude) * ident
        val = (float(aug_factor) / 30) * float(maxval - minval) + minval
        wrap = lambda images: fn.experimental.filter(images, sharpness_kernel(val))
    # elif op == "CutoutAbs":  # Implement
    #     ...
    elif op == "Brightness":
        val = (float(aug_factor) / 30) * float(maxval - minval) + minval
        wrap = lambda images: fn.brightness_contrast(images, brightness_shift=val)
    elif op == "Contrast":
        val = (float(aug_factor) / 30) * float(maxval - minval) + minval
        wrap = lambda images: fn.brightness_contrast(images, contrast=val)
    elif op == "Rotate":
        val = (float(aug_factor) / 30) * float(maxval - minval) + minval
        wrap = lambda images: fn.rotate(images,
                                        angle=val,
                                        interp_type=types.INTERP_LINEAR,
                                        fill_value=0)
    elif op == "Invert":
        wrap = lambda images: fn.flip(images,
                                        vertical=0,
                                        horizontal=1)
    elif op == "ShearX":
        val = (float(aug_factor) / 30) * float(maxval - minval) + minval
        wrap = lambda images: fn.warp_affine(images,
                                            matrix=[1.0, val, 0.0, 0.0, 1.0, 0.0],
                                            interp_type=types.INTERP_LINEAR)
    elif op == "ShearY":
        val = (float(aug_factor) / 30) * float(maxval - minval) + minval
        wrap = lambda images: fn.warp_affine(images,
                                            matrix=[1.0, 0.0, 0.0, val, 1.0, 0.0],
                                            interp_type=types.INTERP_LINEAR)
    elif op == "TranslateXabs" or op == "TranslateX":
        val = (float(aug_factor) / 30) * float(maxval - minval) + minval
        wrap = lambda images: fn.warp_affine(images,
                                            matrix=[1.0, 0.0, val, 0.0, 1.0, 0.0],
                                            interp_type=types.INTERP_LINEAR)
    elif op == "TranslateYabs" or op == "TranslateY":
        val = (float(aug_factor) / 30) * float(maxval - minval) + minval
        wrap = lambda images: fn.warp_affine(images,
                                            matrix=[1.0, 0.0, 0.0, 0.0, 1.0, val],
                                            interp_type=types.INTERP_LINEAR)
    elif op == 'Identity':
        def identity(data):
            return data 
        wrap = lambda images: identity(images)
    else:
        print(op, "not implemented!", flush=True)
        raise NotImplementedError()
    return wrap

def _gpu_worker_loop(augname,
                    in_queue,
                    out_queue,
                    cpu_queues,
                    cpu_workers_id,
                    dataset,
                    microbatch_size,
                    maxbatch_size,
                    _rank,
                    world_size,
                    workers,
                    done_event,
                    gpu_finish_event,
                    gpu_worker_id,
                    aggr_shape_hint,
                    dali_not_in_progress_event,
                    cpu_id,
                    use_shared_memory,
                    pr_mapping_key):
    assert not use_shared_memory or use_shared_memory == USE_SHARED_MEMORY
    OP_STAGE_ORDER = CO.OP_STAGE_ORDER[augname]
    assert augname != "randaugment" or "RandAugment_Manual" in OP_STAGE_ORDER
    assert augname != "autoaugment" or "AutoAugment_Manual" in OP_STAGE_ORDER
    
    # CPU PINNING
    p = psutil.Process()
    if cpu_id is not None:
        print("Current Pinned CPUS before RePinning:", p.cpu_affinity(), flush=True)
        p.cpu_affinity([cpu_id])
    dali_cpu_affinity = p.cpu_affinity()
    print("Current Pinned CPUS:", dali_cpu_affinity, flush=True)
    os.environ["DALI_AFFINITY_MASK"] = ','.join(map(str, dali_cpu_affinity))

    # handle_ = pynvml.nvmlDeviceGetHandleByIndex(_rank)
    # _nvml_affinity_elements = (os.cpu_count() + 63) // 64

    # affinity_string = ''
    # for j in pynvml.nvmlDeviceGetCpuAffinity(handle_, _nvml_affinity_elements):
    #     # assume nvml returns list of 64 bit ints
    #     affinity_string = '{:064b}'.format(j) + affinity_string
    # affinity_list = [int(x) for x in affinity_string]
    # affinity_list.reverse()  # so core 0 is in 0th element of list
    # print("nvml affinity list", affinity_list)

    if __debug__:
        log.debug(f"{_rank}, {gpu_worker_id}")
    shm_buf, worker_progress_info = _load_np_arr(
        "worker_progress_info", _rank, gpu_worker_id, frame=np.array([0, 0], dtype=np.int32))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_rank)
    # print(_rank)
    torch.cuda.set_device(_rank)
    torch_stream = torch.cuda.current_stream(device=_rank)
    max_batch_size = maxbatch_size
    eii_pipes = (
        ExternalInputGpuIterator(dataset, microbatch_size),
        ExternalPrepQueueIterator())
    pipes = {
        "Decode": make_pipe("Decode")(eii_source=eii_pipes[0], num_threads=workers, batch_size=max_batch_size, 
                                      device_id=_rank, prefetch_queue_depth=1, exec_async=False, exec_pipelined=False),
        
        "RandomCrop": make_pipe("RandomCrop")(eii_source=eii_pipes[1], num_threads=workers, batch_size=max_batch_size, 
                                              device_id=_rank, prefetch_queue_depth=1, exec_async=False, exec_pipelined=False),
        
        "FlipAndNormalize": make_pipe("FlipAndNormalize")(eii_source=eii_pipes[1], num_threads=workers, batch_size=max_batch_size, 
                                                          device_id=_rank, prefetch_queue_depth=1, exec_async=False, exec_pipelined=False),
        
        # "RandAugment_Auto": make_pipe("RandAugment_Auto")(eii_source=eii_pipes[1], num_threads=workers, batch_size=max_batch_size, 
        #                                                   device_id=_rank, prefetch_queue_depth=1, exec_async=False)
    }
    if "RandAugment_Auto" in OP_STAGE_ORDER:
        pipes["RandAugment_Auto"] = make_pipe("RandAugment_Auto")(eii_source=eii_pipes[1], num_threads=workers, batch_size=max_batch_size, 
                                                                  device_id=_rank, prefetch_queue_depth=1, exec_async=False, exec_pipelined=False)
    elif "RandAugment_Manual" in OP_STAGE_ORDER:
        aug_factor = CO.M
        for op, minval, maxval in CO.AUG_NAME_LIST:
            wrap = augmentation_func(op, minval, maxval, aug_factor)
            pipes[op] = make_pipe("RandAugment_Manual")(eii_source=eii_pipes[1], aug=wrap, 
                                                        num_threads=workers, batch_size=max_batch_size, 
                                                        device_id=_rank, prefetch_queue_depth=1, exec_async=False, exec_pipelined=False)
    elif "AutoAugment_Manual" in OP_STAGE_ORDER:
        for i, policy in enumerate(CO.POLICY_LIST):
            policy_name = "Policy " + str(i)
            pipes[policy_name] = []
            for aug_name, prob, aug_factor in policy:
                # Get Min-Max Value
                for op, minval, maxval in CO.AUG_NAME_LIST:
                    if op == aug_name:
                        wrap = augmentation_func(op, minval, maxval, aug_factor)
                        pipes[policy_name].append((prob, make_pipe("RandAugment_Manual")(eii_source=eii_pipes[1], aug=wrap, 
                                                        num_threads=workers, batch_size=max_batch_size, 
                                                        device_id=_rank, prefetch_queue_depth=1, exec_async=False, exec_pipelined=False)))
                        break
            # pipes[policy_name] = make_pipe("AutoAugment_Manual")(eii_source=eii_pipes[1], augs=augs, 
            #                                             num_threads=workers, batch_size=max_batch_size, 
            #                                             device_id=_rank, prefetch_queue_depth=1, exec_async=False, exec_pipelined=False)
    elif "DeepAutoAugment_Manual" in OP_STAGE_ORDER:
        aug_factor = CO.DEEP_MAGNITUTE
        for op, minval, maxval in CO.DeepAUTOAUGMENT_OPTIONS:
            wrap = augmentation_func(op, minval, maxval, aug_factor)
            pipes[op] = make_pipe("RandAugment_Manual")(eii_source=eii_pipes[1], aug=wrap, 
                                                        num_threads=workers, batch_size=max_batch_size, 
                                                        device_id=_rank, prefetch_queue_depth=1, exec_async=False, exec_pipelined=False)
        
        # policy_data = np.load('/home/mansur/Manticore/manticore/core/pyfiles/Augmentations/policy_port/policy_DeepAA.npz')
        # policy_probs = policy_data['policy_probs']
        # l_ops = policy_data['l_ops']
        # l_mags = policy_data['l_mags']
        # ops = policy_data['ops']
        # mags = policy_data['mags']
        # op_names = policy_data['op_names']
        # for i, k_policy in enumerate(policy_probs):
        #     policy_name = "Policy " + str(i)
        #     pipes[policy_name] = []
        #     k_samp = random.choices(range(len(k_policy)), weights=k_policy, k=1)[0]
        #     op, mag = np.squeeze(ops[k_samp]), np.squeeze(mags[k_samp]).astype(np.float32)/float(l_mags-1)
        #     op_name = op_names[op].split(':')[0]
    else:
        print("No RandAugment!", flush=True)
        raise NotImplementedError()
    monolith_pipeline = make_monolith_pipe(augname)
    if USE_MONOLITH_PIPE:
        monolith_pipe = monolith_pipeline(eii_source=eii_pipes[0],
                                        num_threads=workers, batch_size=max_batch_size, 
                                        device_id=_rank, prefetch_queue_depth=1, exec_async=False, exec_pipelined=False)
    
    if __debug__:
        log.debug(f"GPU {workers} on num_threads, {microbatch_size} on batch size")
    
    print(f"[gpu_finish_event GPU WORKER] WAIT", flush=True)
    is_built = False
    watchdog = ManagerWatchdog()
    while not done_event.is_set() and watchdog.is_alive():
        worker_progress_info[0] = -1
        if not gpu_finish_event.is_set():
            if USE_SHARED_MEMORY and use_shared_memory:
                print(f"[gpu_finish_event GPU WORKER] WAIT", flush=True)
                dali_not_in_progress_event.set()
                gpu_finish_event.wait()
                dali_not_in_progress_event.clear()    
                dali.backend.ReadSharedMemory(_rank, pr_mapping_key[0], pr_mapping_key[1])
            
        if not is_built:
            is_built = True
            # Monolith Pipe
            if USE_MONOLITH_PIPE:
                monolith_pipe.build()
            else:
                for op_name in pipes.keys():
                    if type(pipes[op_name]) is not list:
                        pipes[op_name].build()
                    else:
                        for prob, pipe in pipes[op_name]:
                            pipe.build()
                preallocate_cuda_vm_memory = False
                if preallocate_cuda_vm_memory:
                    for op_name in pipes.keys():
                        if op_name == 'Decode':
                            continue
                        _dummy_input = [torch.zeros(3050, 3050, 3, dtype=torch.uint8, device=_rank) for i in range(max_batch_size)]
                        _dummy_target = [torch.zeros(1, dtype=torch.uint8, device=_rank) for i in range(max_batch_size)]
                        eii_pipes[1].append(_dummy_input, _dummy_target)
                        d1_, d2_ = pipes[op_name].run()
                        del d1_, d2_, _dummy_input, _dummy_target
                        clear_traced_pipes([op_name], pipes, eii_pipes, _rank)  


        # Get Data from Queue
        _tmp = _try_get_queue(in_queue)
        if isinstance(_tmp, tuple):  # Not a signal, real data!
            op_type = _tmp[0]

            if CO.is_start(op_type):
                op_type = CO.next_op(OP_STAGE_ORDER, op_type)
                decision = CO.conditional_offloading_gpu(op_type)
                if decision != -1:
                    print(f"GPU -> CPU {decision}, OP_TYPE {op_type}", flush=True)
                    while not done_event.is_set():
                        try:
                            cpu_queues[decision].put((op_type, _tmp[1]))
                            break
                        except queue.Full:
                            continue
                    continue
            if CO.get_op_name(op_type) == "Decode":
                idx, _files = _tmp[1]
                if len(_files) == 1:
                    continue
                eii_pipes[0].append(_files)
            else:
                print("Received data from CPU", flush=True)
                idx, _input, _target = _tmp[1]
                # Send them to GPU side

                start_time = timer_func()
                
                for i in range(len(_input)):
                    _input[i] = torch.from_numpy(_input[i])
                    _input[i] = _input[i].to(device=_rank)
                for i in range(len(_target)):
                    _target[i] = torch.tensor([_target[i]], dtype=torch.uint8)
                    _target[i] = _target[i].to(device=_rank)
                torch.cuda.synchronize()
                
                end_time = timer_func()

                print(f"Received data from CPU in {end_time - start_time} s", flush=True)

                eii_pipes[1].append(_input, _target)
        elif _tmp == None:
            if __debug__:
                log.debug(f"GPU: {_rank} GPU worker Finish")
            break
        elif _tmp == -1 or _tmp == -999:
            continue
        else:
            print("What are you", _tmp, type(_tmp), flush=True)
        if isinstance(idx, int):
            idx = [idx,]
        if __debug__:
            log.debug(f"GPU: {_rank} GPU worker idx {idx}")          
        worker_progress_info[0] = idx[-1]
        worker_progress_info[1] = microbatch_size
        print("GPU Received", idx, "tinybatches", flush=True)
        
        traced_operations = []  # What kind of pipes where used
        if USE_MONOLITH_PIPE:
            _input, _target = monolith_pipe.run()
        else:
            while not done_event.is_set():
                print("GPU", op_type, flush=True)
                op_name = CO.get_op_name(op_type)
                traced_operations.append(op_name)

                start_time = timer_func()
                is_first = True
                if type(pipes[op_name]) is not list:
                    _input, _target = pipes[op_name].run()
                    is_first = False
                else:
                    for prob, pipe in pipes[op_name]:
                        if random.random() > prob:
                            continue
                        if not is_first:
                            eii_pipes[1].append(_input, _target)
                            del _input, _target
                        _input, _target = pipe.run()
                        is_first = False
                
                end_time = timer_func()
                print(f"GPU Operation, {op_name}, f{end_time - start_time} s", flush=True)
                
                op_type = CO.next_op(OP_STAGE_ORDER, op_type)
                if CO.is_finish(op_type):
                    break
                decision = CO.conditional_offloading_gpu(op_type)
                if decision == -1:
                    if not is_first:
                        eii_pipes[1].append(_input, _target)
                        del _input, _target
                else:
                    start_time = timer_func()

                    # Place them to CPU, both of them!!
                    _input = _input.as_cpu()
                    _cnt = len(_input.shape())
                    _input_tmp = []
                    for i in range(_cnt):
                        _input_tmp.append(_input.at(i))
                    _input = _input_tmp

                    _target = _target.as_cpu()
                    _target_tmp = []
                    for i in range(_cnt):
                        _target_tmp.append(_target.at(i)[0])
                    _target = _target_tmp

                    end_time = timer_func()
                    print(f"GPU -> CPU data transfer preparation {end_time - start_time} s", flush=True)

                    # Send them to Worker!
                    print(f"GPU -> CPU {decision}, OP_TYPE {op_type}", flush=True)
                    while not done_event.is_set():
                        try:
                            cpu_queues[decision].put((op_type, (idx, _input, _target)))
                            break
                        except queue.Full:
                            continue
                    break
            if done_event.is_set():
                break
            if not CO.is_finish(op_type):
                del _input, _target
                clear_traced_pipes(traced_operations, pipes, eii_pipes, _rank)
                continue

        # Data Preparation is done, now send it back!
        # Create copy of the data since DALI uses CUDA VM, which does not support sharing!
        torch_input = torch.empty(microbatch_size * len(idx),aggr_shape_hint[1],aggr_shape_hint[2],aggr_shape_hint[3], dtype=torch.float32, device=_rank)
        torch_target = torch.empty(microbatch_size * len(idx), dtype=torch.uint8, device=_rank)
        input_tensor = _input.as_tensor()
        target_tensor = _target.as_tensor()
        feed_ndarray(input_tensor, torch_input, torch_stream, torch_input[0].data_ptr())
        feed_ndarray(target_tensor, torch_target, torch_stream, torch_target[0].data_ptr())
        torch.cuda.synchronize()
        if __debug__:
            log.debug(f"GPU:{_rank} see target {target_tensor} and copyed {torch_target} torch allocation {torch.cuda.memory_allocated({_rank})} torch reserved {torch.cuda.memory_reserved({_rank})}")
        
        if DEALLOC_GPU_MEMORY_AFTER_OP:
            clear_traced_pipes(traced_operations, pipes, eii_pipes, _rank)
        del _input, _target, input_tensor, target_tensor
        
        torch_target=torch_target.type(torch.long)
        for i in range(len(idx)):
            while not done_event.is_set():
                try:
                    out_queue.put((idx[i],(torch_input[i*microbatch_size:(i+1)*microbatch_size],torch_target[i*microbatch_size:(i+1)*microbatch_size])))
                    break
                except queue.Full:
                    continue
        del torch_input, torch_target
    if done_event.is_set():
        shm_buf.close()
        out_queue.cancel_join_thread()
        if __debug__:
            log.debug(f"GPU: {_rank} worker_id: {gpu_worker_id}, cancel_join_thread data_queue:\n{out_queue}")
        out_queue.close()
