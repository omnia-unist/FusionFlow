import torch
import math
import time
import time
import numpy as np
import torch
import random
from itertools import combinations
import cupy

import threading
import torch.multiprocessing as multiprocessing
from torch._six import queue
from torch.utils.data import RandomSampler
from posix_ipc import O_CREAT, SharedMemory, unlink_shared_memory
from .sampler import FinegrainedBatchSampler, BatchSampler
import ctypes

try:
    from nvidia import dali
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.backend import TensorGPU, TensorListGPU, TensorListCPU

    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
    import nvidia.dali.backend as backend
except ImportError:
    raise ImportError(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run.")

# from https://github.com/NVIDIA/DALI/blob/main/dali/python/nvidia/dali/plugin/pytorch.py
to_torch_type = {
    np.dtype(np.float32) : torch.float32,
    np.dtype(np.float64) : torch.float64,
    np.dtype(np.float16) : torch.float16,
    np.dtype(np.uint8)   : torch.uint8,
    np.dtype(np.int8)    : torch.int8,
    np.dtype(np.int16)   : torch.int16,
    np.dtype(np.int32)   : torch.int32,
    np.dtype(np.int64)   : torch.int64
}

if __debug__:
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger(__name__)
    
DALI_RANDAUGMENT_LIST = [
    # (AutoContrast, 0, 1),
    # (Equalize, 0, 1),
    ("Invert", 0, 1),
    ("Rotate", 0, 30),
    # (Posterize, 0, 4),
    # (Solarize, 0, 256),
    # (SolarizeAdd, 0, 110),
    # (Color, 0.1, 1.9),
    ("Contrast", 0.1, 1.9),
    ("Brightness", 0.1, 1.9),
    # ("Sharpness", 0.1, 1.9),
    ("ShearX", 0., 0.3),
    ("ShearY", 0., 0.3),
    # (CutoutAbs, 0, 40),
    ("TranslateXabs", 0., 100),
    ("TranslateYabs", 0., 100),
]


def clear_memory(verbose=True):
    StartTime = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # https://forums.fast.ai/t/clearing-gpu-memory-pytorch/14637
    # gc.collect()

    if verbose:
        print('Cleared memory.  Time taken was %f secs' %
              (time.time() - StartTime))

def _preproc_worker(dali_iterator, cuda_stream, fp16, mean, std, output_queue, proc_next_input, done_event, pin_memory):
    """
    Worker function to parse DALI output & apply final pre-processing steps
    """

    while not done_event.is_set():
        # Wait until main thread signals to proc_next_input -- normally once it has taken the last processed input
        proc_next_input.wait()
        proc_next_input.clear()

        if done_event.is_set():
            print('Shutting down preproc thread')
            break

        try:
            data = next(dali_iterator)

            # Decode the data output
            input_orig = data[0]['data']
            target = data[0]['label'].squeeze().long()  # DALI should already output target on device

            # Copy to GPU and apply final processing in separate CUDA stream
            with torch.cuda.stream(cuda_stream):
                input = input_orig
                if pin_memory:
                    input = input.pin_memory()
                    del input_orig  # Save memory
                input = input.cuda(non_blocking=True)

                input = input.permute(0, 3, 1, 2)

                # Input tensor is kept as 8-bit integer for transfer to GPU, to save bandwidth
                if fp16:
                    input = input.half()
                else:
                    input = input.float()

                input = input.sub_(mean).div_(std)

            # Put the result on the queue
            output_queue.put((input, target))

        except StopIteration:
            print('Resetting DALI loader')
            dali_iterator.reset()
            output_queue.put(None)


class ExternalInputGpuIterator(object):
    def __init__(self, dataset, batch_size, in_queue):
        self.batch_size = batch_size
        self.dataset = dataset
        self.in_queue = in_queue
        self.data_path_list = None
        self.data_label_list = None
        self.files = []
        
    def set_files(self):
        if __debug__:
            log.debug(f"ExternalInput: wait for queue")
        data_list = self.in_queue.get()
        if __debug__:
            log.debug(f"ExternalInput: set_dataset as {data_list}")
        self.files = data_list
        
    def __iter__(self):
        return self
    
    def __next__(self):
        self.set_files()
        
        batches = []
        labels = []
        
        if __debug__:
            log.debug(f"ExternalInput: run pipe {self.files}")

        for index in self.files:
            jpeg_file, label = self.dataset.samples[index]
            f = open(jpeg_file, 'rb')
            batches.append(np.frombuffer(f.read(), dtype = np.uint8))
            labels.append(np.array([label], dtype = np.uint8))
        
        return (batches, labels)

class DummyExternalInputGpuIterator(ExternalInputGpuIterator):
    def __init__(self, dataset, batch_size, in_queue):
        super(DummyExternalInputGpuIterator, self).__init__(dataset, batch_size, in_queue)
        self.in_queue = in_queue
    def __next__(self):
        images, labels = self.in_queue.get()
        if __debug__:
            log.debug(f"DummyExternalInput: run pipe {labels}")
        return images, labels



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

# class HybridTrainPipe(Pipeline):
#     """
#     From https://github.com/yaysummeriscoming/DALI_pytorch_demo/blob/master/dali.py
#     Modify for dynamic input with external source

#     DALI Train Pipeline
#     Based on the official example: https://github.com/NVIDIA/DALI/blob/master/docs/examples/pytorch/resnet50/main.py
#     In comparison to the example, the CPU backend does more computation on CPU, reducing GPU load & memory use.
#     This dataloader implements ImageNet style training preprocessing, namely:
#     -random resized crop
#     -random horizontal flip
#     batch_size (int): how many samples per batch to load
#     num_threads (int): how many DALI workers to use for data loading.
#     device_id (int): GPU device ID
#     containing train & val subdirectories, with image class subfolders
#     crop (int): Image output size (typically 224 for ImageNet)
#     mean (tuple): Image mean value for each channel
#     std (tuple): Image standard deviation value for each channel
#     local_rank (int, optional, default = 0) – Id of the part to read
#     world_size (int, optional, default = 1) - Partition the data into this many parts (used for multiGPU training)
#     dali_cpu (bool, optional, default = False) - Use DALI CPU mode instead of GPU
#     shuffle (bool, optional, default = True) - Shuffle the dataset each epoch
#     fp16 (bool, optional, default = False) - Output the data in fp16 instead of fp32 (GPU mode only)
#     min_crop_size (float, optional, default = 0.08) - Minimum random crop size
#     """

#     def __init__(self, in_queue, dataset, batch_size, num_threads, device_id, crop,
#                  mean, std, local_rank=0, world_size=1, dali_cpu=False, shuffle=True, fp16=False,
#                  min_crop_size=0.08):

#         # As we're recreating the Pipeline at every epoch, the seed must be -1 (random seed)
#         super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=-1)
#         self.eii_gpu = ExternalInputGpuIterator(dataset, batch_size, in_queue)
#         # Enabling read_ahead slowed down processing ~40%

#         # Let user decide which pipeline works best with the chosen model
#         decode_device = "mixed"
#         self.dali_device = "gpu"

#         output_dtype = types.FLOAT
#         if self.dali_device == "gpu" and fp16:
#             output_dtype = types.FLOAT16

#         self.dali_device = "gpu"
#         self.crop = crop
#         self.size = crop
#         self.resize = self.size + 32
#         self.mean = mean
#         self.std = std
#         # print('DALI "{0}" variant'.format(self.dali_device))

#     def define_graph(self):
#         rng = fn.random.coin_flip(probability=0.5)
#         self.jpegs, self.labels = dali.fn.external_source(source=self.eii_gpu, num_outputs=2, device="cpu")

#         # Combined decode & random crop
#         images = fn.decoders.image(self.jpegs, device="mixed", device_memory_padding = 0)
#         # Resize as desired

#         if self.dali_device == "gpu":
#             images = fn.resize(images, resize_x=self.resize, resize_y=self.resize, minibatch_size=2)
#             output = fn.crop_mirror_normalize(images,
#                                               mirror=rng,
#                                               crop=[self.crop, self.crop],
#                                               mean=self.mean,
#                                               std=self.std,
#                                               dtype=types.FLOAT,
#                                               output_layout=types.NCHW)
#         else:
#             images = fn.resize(images, resize_x=self.resize, resize_y=self.resize, minibatch_size=2)
#             output = fn.crop_mirror_normalize(images,
#                                               mirror=rng,
#                                               crop=[self.crop, self.crop],
#                                               mean=self.mean,
#                                               std=self.std,
#                                               dtype=types.FLOAT,
#                                               output_layout=types.NCHW)
#         self.labels = self.labels.gpu()
#         return [output, self.labels]

class OriginalTrainPipe(Pipeline):
    """
    From https://github.com/yaysummeriscoming/DALI_pytorch_demo/blob/master/dali.py
    With file reader

    DALI Train Pipeline
    Based on the official example: https://github.com/NVIDIA/DALI/blob/master/docs/examples/pytorch/resnet50/main.py
    In comparison to the example, the CPU backend does more computation on CPU, reducing GPU load & memory use.
    This dataloader implements ImageNet style training preprocessing, namely:
    -random resized crop
    -random horizontal flip
    batch_size (int): how many samples per batch to load
    num_threads (int): how many DALI workers to use for data loading.
    device_id (int): GPU device ID
    containing train & val subdirectories, with image class subfolders
    crop (int): Image output size (typically 224 for ImageNet)
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    local_rank (int, optional, default = 0) – Id of the part to read
    world_size (int, optional, default = 1) - Partition the data into this many parts (used for multiGPU training)
    dali_cpu (bool, optional, default = False) - Use DALI CPU mode instead of GPU
    shuffle (bool, optional, default = True) - Shuffle the dataset each epoch
    fp16 (bool, optional, default = False) - Output the data in fp16 instead of fp32 (GPU mode only)
    min_crop_size (float, optional, default = 0.08) - Minimum random crop size
    """

    def __init__(self, path, batch_size, num_threads, device_id, crop,
                 mean, std, local_rank=0, world_size=1, dali_cpu=False, shuffle=True, fp16=False,
                 min_crop_size=0.08):

        # As we're recreating the Pipeline at every epoch, the seed must be -1 (random seed)
        super(OriginalTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=-1)
        self.path = path
        # Let user decide which pipeline works best with the chosen model
        decode_device = "mixed"
        self.dali_device = "gpu"
        self.crop = crop
        self.size = crop
        self.resize = self.size + 32
        self.mean = mean
        self.std = std
        self.shuffle = shuffle
        self.local_rank = local_rank
        self.world_size = world_size
        if __debug__:
            log.debug(f"rank = {self.local_rank}, world_size = {self.world_size}")
        output_dtype = types.FLOAT
        if self.dali_device == "gpu" and fp16:
            output_dtype = types.FLOAT16
        # print('DALI "{0}" variant'.format(self.dali_device))
        
    def define_graph(self):
        rng = fn.random.coin_flip(probability=0.5)
        self.jpegs, self.labels = fn.readers.file(file_root=self.path, num_shards=self.world_size, shard_id=self.local_rank, random_shuffle = self.shuffle, device="cpu")

        # Combined decode & random crop
        images = fn.decoders.image(self.jpegs, device="mixed", device_memory_padding = 0)
        # Resize as desired

        images = fn.random_resized_crop(images, size=[self.crop,self.crop], minibatch_size=1)
        output = fn.crop_mirror_normalize(images,
                                          mirror=rng,
                                          mean=self.mean,
                                          std=self.std,
                                          dtype=types.FLOAT,
                                          output_layout=types.NCHW)
            
        self.labels = self.labels.gpu()
        return [output, self.labels]



class RandAugmentPipe(Pipeline):
    """[summary]
    Args:
        Pipeline ([type]): [description]
    Raises:
        StopIteration: [description]
    Returns:
        [type]: [description]
    """

    def __init__(self, batch_size, device_id, crop, dataset, in_queue,
                 mean, std, local_rank, world_size, dali_cpu=False, fp16=False,
                 min_crop_size=0.08, aug_name_list=[], aug_factor=1, num_threads=2,
                 py_num_workers=2,py_start_method="fork"):
        super(RandAugmentPipe, self).__init__(batch_size, num_threads, device_id, seed=-1, prefetch_queue_depth=1)
        
        self.eii_gpu = ExternalInputGpuIterator(dataset, batch_size, in_queue)
        
        self.pipe_BatchSize = batch_size
        self.crop = crop
        self.mean = mean
        self.std = std
        self.local_rank = local_rank
        self.size = crop
        self.resize = self.size + 32
        # if dali_cpu:
        #     self.decode_device = "cpu"
        #     self.dali_device = "cpu"
        # else:
        self.decode_device = "mixed"
        self.dali_device = "gpu"
        # if self.dali_device == "gpu" and fp16:
        #     output_dtype = types.FLOAT16
        # else:
        #     output_dtype = types.FLOAT
        self.world_size = world_size
        self.augmentations = {}
        self.meta_augmentations = []
        for op, minval, maxval in aug_name_list:
            self.meta_augmentations.append(op)
            if op == "Brightness":
                val = (float(aug_factor) / 30) * \
                    float(maxval - minval) + minval
                '''
                fn.brightness_contrast
                
                out = brightness_shift * output_range + brightness \
                    * (contrast_center + contrast * (in - contrast_center))
                '''
                self.augmentations["Brightness"] = \
                    lambda images: fn.brightness_contrast(images,
                                                            brightness_shift=val)
            elif op == "Contrast":
                val = (float(aug_factor) / 30) * \
                    float(maxval - minval) + minval
                self.augmentations["Contrast"] = \
                    lambda images: fn.brightness_contrast(images,
                                                            contrast=val)

            if op == "Rotate":
                val = (float(aug_factor) / 30) * \
                    float(maxval - minval) + minval
                self.augmentations["Rotate"] = \
                    lambda images: fn.rotate(images,
                                                angle=val,
                                                interp_type=types.INTERP_LINEAR,
                                                fill_value=0)
            if op == "Invert":
                val = (float(aug_factor) / 30) * \
                    float(maxval - minval) + minval
                # Color value inverting - implement with flip for convenience
                self.augmentations["Invert"] = \
                    lambda images: fn.flip(images,
                                            vertical=0,
                                            horizontal=1)

            if op == "ShearX":
                val = (float(aug_factor) / 30) * \
                    float(maxval - minval) + minval
                # ShearX img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))
                self.augmentations["ShearX"] = \
                    lambda images: fn.warp_affine(images,
                                                    matrix=[
                                                        1.0, val, 0.0, 0.0, 1.0, 0.0],
                                                    interp_type=types.INTERP_LINEAR)
            if op == "ShearY":
                val = (float(aug_factor) / 30) * \
                    float(maxval - minval) + minval
                # ShearY img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))
                self.augmentations["ShearY"] = \
                    lambda images: fn.warp_affine(images,
                                                    matrix=[
                                                        1.0, 0.0, 0.0, val, 1.0, 0.0],
                                                    interp_type=types.INTERP_LINEAR)
            if op == "TranslateXabs":
                val = (float(aug_factor) / 30) * \
                    float(maxval - minval) + minval
                # TranslateX abs img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
                self.augmentations["TranslateXabs"] = \
                    lambda images: fn.warp_affine(images,
                                                    matrix=[
                                                        1.0, 0.0, val, 0.0, 1.0, 0.0],
                                                    interp_type=types.INTERP_LINEAR)
            if op == "TranslateYabs":
                val = (float(aug_factor) / 30) * \
                    float(maxval - minval) + minval
                # TranslateY abs img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
                self.augmentations["TranslateYabs"] = \
                    lambda images: fn.warp_affine(images,
                                                    matrix=[
                                                        1.0, 0.0, 0.0, 0.0, 1.0, val],
                                                    interp_type=types.INTERP_LINEAR)
    

        
        # # print('DALI "{0}" variant'.format(self.dali_device))
    
    def define_graph(self):
        rng = fn.random.coin_flip(probability=0.5)
        self.jpegs, self.labels = dali.fn.external_source(source=self.eii_gpu, num_outputs=2, device="cpu")

        # Combined decode & random crop
        images = fn.decoders.image(self.jpegs, device="mixed", device_memory_padding = 0)
        # Resize as desired

        if self.dali_device == "gpu":
            print(self.meta_augmentations)
            for aug in self.augmentations.values():
                images = aug(images)
            images = fn.random_resized_crop(images, size=[self.crop,self.crop], minibatch_size=1)
            output = fn.crop_mirror_normalize(images,
                                            mirror=rng,
                                            mean=self.mean,
                                            std=self.std,
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW)
        else:
            print("Working on CPU side")
            print(self.meta_augmentations)
            for aug in self.augmentations.values():
                images = aug(images)
            images = fn.random_resized_crop(images, size=[self.crop,self.crop], minibatch_size=1)
            output = fn.crop_mirror_normalize(images,
                                            mirror=rng,
                                            mean=self.mean,
                                            std=self.std,
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW)
        self.labels = self.labels.gpu()
        return [output, self.labels]


class dummyPipeline(Pipeline):
    def __init__(self, in_queue, batch_size, dataset, device_id, num_threads=2) -> None:
        super(dummyPipeline, self).__init__(batch_size, num_threads, device_id, seed=-1, prefetch_queue_depth=2)
        self.in_queue = in_queue
        self.eli = DummyExternalInputGpuIterator(dataset, batch_size, in_queue)
    

    def define_graph(self):
        images, labels = dali.fn.external_source(source=self.eli, num_outputs=2, device="cpu")
        return images, labels

# DALI Tensor to Torch Tensor process
def final_processor(in_queue, out_queue, done_event, dataset, minibatch_size, workers, device_id):
    torch.cuda.set_device(device_id)
    dummy_pipe = dummyPipeline(
        in_queue=in_queue, dataset=dataset, batch_size=minibatch_size, device_id=device_id, num_threads=workers
    )
    dummy_pipe.build()
    _dali_iterator = DaliIteratorGPU(pipelines=dummy_pipe)
    if __debug__:
        log.debug("Flush dummy")
    dummy_out = next(_dali_iterator)  
    if __debug__:
        log.debug(f"Final_processor: dummy {dummy_out}")
    del dummy_out
    try:
        while not done_event.is_set():
            data=next(_dali_iterator)
            while not done_event.is_set():
                try:
                    out_queue.put(data)
                    break
                except queue.Full:
                    continue
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass
    if done_event.is_set():
        out_queue.cancel_join_thread()
        out_queue.close()

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
                 workers=4,
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
        
    def _build_dali_pipeline(self):
        # assert self.world_size == 1, 'Distributed support not tested yet'
        self.train_loader = None
        if __debug__:
            log.debug(f"Build pipeline with device {self.device_id} augtype{self.augment_type}")
        if self.augment_type == "randaugment":
            self.train_pipes = []
            if __debug__:
                i = 0
            for _ops in self.ops_combinations:
                if __debug__:
                    log.debug(f"Build Randaug {i} Ops:{_ops}")
                self.in_queue.put([0])
                self.in_queue.put([0])
                train_pipe = RandAugmentPipe( batch_size=self.microbatch_size, num_threads=self.workers, device_id=self.device_id, in_queue=self.in_queue,
                                            dataset=self.dataset, crop=self.size, dali_cpu=self.dali_cpu,
                                            mean=self.mean, std=self.std, local_rank=self.device_id,
                                            world_size=self.world_size, fp16=self.fp16, min_crop_size=self.min_crop_size)
                if __debug__:
                    # log.debug("Randaug build start")
                    start = time.perf_counter()
                train_pipe.build()
                if __debug__:
                    end = time.perf_counter()
                    log.debug(f"Randaug build pipeline done TIME {end - start}")
                # data = train_pipe.run()
                # del data
                if __debug__:
                    log.debug(f"Randaug dali_queue qsize: {self.in_queue.qsize()}")
                    i += 1
                self.train_pipes.append(train_pipe)
            if __debug__:
                log.debug("Randaug delete queue")
            
            if __debug__:
                log.debug("Randaug build pipeline done")
        else:
            if __debug__:
                log.debug("Build pipeline with device Hybrid")
            self.in_queue.put([0])
            self.in_queue.put([0])
            # train_pipe = HybridTrainPipe(  batch_size=self.minibatch_size, num_threads=self.workers, device_id=self.device_id, in_queue=self.in_queue,
            #                                 dataset=self.dataset, crop=self.size, dali_cpu=self.dali_cpu,
            #                                 mean=self.mean, std=self.std, local_rank=self.device_id,
            #                                 world_size=self.world_size, fp16=self.fp16, min_crop_size=self.min_crop_size)
            
            train_pipe2 = OriginalTrainPipe( path=self.path ,batch_size=self.minibatch_size, num_threads=self.workers, device_id=self.device_id, crop=self.size, dali_cpu=self.dali_cpu, world_size=self.world_size,
                                            mean=self.mean, std=self.std, local_rank=self.device_id, fp16=self.fp16, min_crop_size=self.min_crop_size)
            
            # train_pipe.serialize()
            
            # train_pipe.build()
            train_pipe2.build()
            pipe_size = train_pipe2.epoch_size()
            size = int(pipe_size['__File_1']) / self.world_size
            if __debug__:
                log.debug(f"Hybrid build size {pipe_size}, {size}")
            self.train_loader = DaliIteratorGPU(pipelines=train_pipe2, size = size, fp16=self.fp16, mean=self.mean, std=self.std, pin_memory=self.pin_memory_dali)
    
    def get_default_loader(self):
        print("Get Default Augmentation Loader")
        return self.train_loader
    
    def get_max_select(self):
        return len(self.ops_combinations)
    
    def get_pipe(self, op_number):
        """
        Creates & returns an iterator for the training dataset
        :return: Dataset iterator object
        """
        # self.reset(data_list, batch_size, device_id)
        return self.train_pipes[op_number]
    
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
        dali_type = np.dtype(dali_type)
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