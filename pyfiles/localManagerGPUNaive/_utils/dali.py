
import torch
import math
import time
import time
import numpy as np
import torch
import random
from itertools import combinations

import threading
from torch.multiprocessing import Event
from torch._six import queue
from posix_ipc import O_CREAT, SharedMemory, unlink_shared_memory
import mmap
from nvidia.dali.backend import TensorGPU, TensorListGPU, TensorListCPU
import ctypes
try:
    from nvidia import dali
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run.")

if __debug__:
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger(__name__)
# ORIG_IMAGENET_AUTOAUGMENT_POLICY = [
#     [('PosterizeOriginal', 0.4, 8), ('Rotate', 0.6, 9)],
#     [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
#     [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
#     [('PosterizeOriginal', 0.6, 7), ('PosterizeOriginal', 0.6, 6)],
#     [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
#     [('Equalize', 0.4, 4), ('Rotate', 0.8, 8)],
#     [('Solarize', 0.6, 3), ('Equalize', 0.6, 7)],
#     [('PosterizeOriginal', 0.8, 5), ('Equalize', 1.0, 2)],
#     [('Rotate', 0.2, 3), ('Solarize', 0.6, 8)],
#     [('Equalize', 0.6, 8), ('PosterizeOriginal', 0.4, 6)],
#     [('Rotate', 0.8, 8), ('Color', 0.4, 0)],
#     [('Rotate', 0.4, 9), ('Equalize', 0.6, 2)],
#     [('Equalize', 0.0, 7), ('Equalize', 0.8, 8)],
#     [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
#     [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
#     [('Rotate', 0.8, 8), ('Color', 1.0, 2)],
#     [('Color', 0.8, 8), ('Solarize', 0.8, 7)],
#     [('Sharpness', 0.4, 7), ('Invert', 0.6, 8)],
#     [('ShearX', 0.6, 5), ('Equalize', 1.0, 9)],
#     [('Color', 0.4, 0), ('Equalize', 0.6, 3)],
#     [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
#     [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
#     [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
#     [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
#     [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
# ]


def minmax_list():
    minmax = {
        "AutoContrast":(0, 1),
        "Equalize":(0, 1),
        "Invert":(0, 1),
        "Rotate":(0, 30),
        "Posterize":(0, 4),
        "Solarize":(0, 256),
        "SolarizeAdd":(0, 110),
        "Color":(0.1, 1.9),
        "Contrast":(0.1, 1.9),
        "Brightness":(0.1, 1.9),
        "Sharpness":(0.1, 1.9),
        "ShearX":(0., 0.3),
        "ShearY":(0., 0.3),
        "CutoutAbs":(0, 40),
        "TranslateXabs":(0., 100),
        "TranslateYabs":(0., 100)
    }

    return minmax

# NOTE: Try to replace similar cost augmentation operators based on static profiling...
#       due to unsupported augmentation operator in NVIDIA DALI
# Replaced List:
#   Equalize == AutoContrast == Sharpness == Contrast
#   Posterize == Invert == Solarize == TranslateXabs 
#   SolarizeAdd == Rotate
def imagenet_autoaugment_policy():
    p = [
        [('TranslateXabs', 0.4, 8), ('Rotate', 0.6, 9)],
        [('TranslateXabs', 0.6, 5), ('Contrast', 0.6, 5)],
        [('Contrast', 0.8, 8), ('Contrast', 0.6, 3)],
        [('TranslateXabs', 0.6, 7), ('TranslateXabs', 0.6, 6)],
        [('Contrast', 0.4, 7), ('TranslateXabs', 0.2, 4)],
        [('Contrast', 0.4, 4), ('Rotate', 0.8, 8)],
        [('TranslateXabs', 0.6, 3), ('Contrast', 0.6, 7)],
        [('TranslateXabs', 0.8, 5), ('Contrast', 1.0, 2)],
        [('Rotate', 0.2, 3), ('TranslateXabs', 0.6, 8)],
        [('Contrast', 0.6, 8), ('TranslateXabs', 0.4, 6)],
        [('Rotate', 0.8, 8), ('Contrast', 0.4, 0)],
        [('Rotate', 0.4, 9), ('Contrast', 0.6, 2)],
        [('Contrast', 0.0, 7), ('Contrast', 0.8, 8)],
        [('TranslateXabs', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Contrast', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Rotate', 0.8, 8), ('Contrast', 1.0, 2)],
        [('Contrast', 0.8, 8), ('TranslateXabs', 0.8, 7)],
        [('Contrast', 0.4, 7), ('TranslateXabs', 0.6, 8)],
        [('ShearX', 0.6, 5), ('Contrast', 1.0, 9)],
        [('Contrast', 0.4, 0), ('Contrast', 0.6, 3)],
        [('Contrast', 0.4, 7), ('TranslateXabs', 0.2, 4)],
        [('TranslateXabs', 0.6, 5), ('Contrast', 0.6, 5)],
        [('TranslateXabs', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Contrast', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Contrast', 0.8, 8), ('Contrast', 0.6, 3)],
    ]
    return p

DALI_RANDAUGMENT_LIST = [
    # (AutoContrast, 0, 1),
    # (Equalize, 0, 1),
    # ("Invert", 0, 1),
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
    def __init__(self, dataset, batch_size, in_queue, from_file=True):
        self.batch_size = batch_size
        self.dataset = dataset
        self.in_queue = in_queue
        self.data_path_list = None
        self.data_label_list = None
        self.files = []
        self.from_file = from_file

    def set_files(self):
        if __debug__:
            log.debug(f"ExternalInput: set_Files wait for queue")
        data_list = self.in_queue.get()
        if __debug__:
            log.debug(f"ExternalInput: set_dataset as {data_list}")
        
        self.files = data_list
    
    def get_gpu_batches(self):
        if __debug__:
            log.debug(f"ExternalInput: get_gpu_batch wait for queue")
        batches, labels = self.in_queue.get()
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
                batches.append(np.frombuffer(f.read(), dtype = np.uint8))
                labels.append(np.array([label], dtype = np.uint8))
        else:
            batches, labels = self.get_gpu_batches()

        return (batches, labels)

            


class DaliIterator():
    """
    Wrapper class to decode the DALI iterator output & provide iterator that functions the same as torchvision
    pipelines (Pipeline): DALI pipelines
    size (int): Number of examples in set
    Note: allow extra inputs to keep compatibility with CPU iterator
    """

    def __init__(self, pipelines, **kwargs):
        self._dali_iterator = DALIClassificationIterator(
            pipelines=pipelines)

    def __iter__(self):
        return self

    def __len__(self):
        return int(math.ceil(self._dali_iterator._size / self._dali_iterator.batch_size))


class DaliIteratorCPU(DaliIterator):
    """
    Wrapper class to decode the DALI iterator output & provide iterator that functions the same as torchvision
    Note that permutation to channels first, converting from 8 bit to float & normalization are all performed on GPU
    pipelines (Pipeline): DALI pipelines
    size (int): Number of examples in set
    fp16 (bool): Use fp16 as output format, f32 otherwise
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    pin_memory (bool): Transfer input tensor to pinned memory, before moving to GPU
    """

    def __init__(self, fp16=False, mean=(0., 0., 0.), std=(1., 1., 1.), pin_memory=True, **kwargs):
        super().__init__(**kwargs)
        print('Using DALI CPU iterator')
        self.stream = torch.cuda.Stream()

        self.fp16 = fp16
        self.mean = torch.tensor(mean).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor(std).cuda().view(1, 3, 1, 1)
        self.pin_memory = pin_memory

        if self.fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()

        self.proc_next_input = Event()
        self.done_event = Event()
        self.output_queue = queue.Queue(maxsize=5)
        self.preproc_thread = threading.Thread(
            target=_preproc_worker,
            kwargs={'dali_iterator': self._dali_iterator, 'cuda_stream': self.stream, 'fp16': self.fp16, 'mean': self.mean, 'std': self.std, 'proc_next_input': self.proc_next_input, 'done_event': self.done_event, 'output_queue': self.output_queue, 'pin_memory': self.pin_memory})
        self.preproc_thread.daemon = True
        self.preproc_thread.start()

        self.proc_next_input.set()

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.output_queue.get()
        self.proc_next_input.set()
        if data is None:
            raise StopIteration
        return data

    def __del__(self):
        self.done_event.set()
        self.proc_next_input.set()
        torch.cuda.current_stream().wait_stream(self.stream)
        self.preproc_thread.join()


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


class HybridTrainPipe(Pipeline):
    """
    From https://github.com/yaysummeriscoming/DALI_pytorch_demo/blob/master/dali.py
    Modify for dynamic input with external source

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

    def __init__(self, in_queue, dataset, batch_size, num_threads, device_id, crop,
                 mean, std, local_rank=0, world_size=1, dali_cpu=False, shuffle=True, fp16=False,
                 min_crop_size=0.08):

        # As we're recreating the Pipeline at every epoch, the seed must be -1 (random seed)
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=-1)
        self.eii_gpu = ExternalInputGpuIterator(dataset, batch_size, in_queue)
        # Enabling read_ahead slowed down processing ~40%

        # Let user decide which pipeline works best with the chosen model
        decode_device = "mixed"
        self.dali_device = "gpu"
        self.crop = crop
        self.size = crop
        self.resize = self.size + 32
        self.mean = mean
        self.std = std
        output_dtype = types.FLOAT
        if self.dali_device == "gpu" and fp16:
            output_dtype = types.FLOAT16
        # print('DALI "{0}" variant'.format(self.dali_device))
        
    def define_graph(self):
        rng = fn.random.coin_flip(probability=0.5)
        self.jpegs, self.labels = dali.fn.external_source(source=self.eii_gpu, num_outputs=2, device="cpu")

        # Combined decode & random crop
        images = fn.decoders.image(self.jpegs, device="mixed")
        # Resize as desired

        if self.dali_device == "gpu":
            images = fn.random_resized_crop(images, size=[self.crop,self.crop], minibatch_size=1)
            output = fn.crop_mirror_normalize(images,
                                              mirror=rng,
                                              mean=self.mean,
                                              std=self.std,
                                              dtype=types.FLOAT,
                                              output_layout=types.NCHW)
        else:
            images = fn.random_resized_crop(images, size=[self.crop,self.crop], minibatch_size=1)
            output = fn.crop_mirror_normalize(images,
                                              mirror=rng,
                                              mean=self.mean,
                                              std=self.std,
                                              dtype=types.FLOAT,
                                              output_layout=types.NCHW)
        self.labels = self.labels.gpu()
        return [output, self.labels]



class AutoAugmentPipeElement(Pipeline):
    """[summary]
    Modify
    Args:
        Pipeline ([type]): [description]
    Raises:
        StopIteration: [description]
    Returns:
        [type]: [description]
    """

    def __init__(self, batch_size, device_id, crop, dataset, in_queue,
                 mean, std, local_rank, world_size, dali_cpu=False, fp16=False,
                 min_crop_size=0.08, subpolicy=None, num_threads=2,
                 py_num_workers=2,py_start_method="fork", layer_order=0):
        super(AutoAugmentPipeElement, self).__init__(batch_size, num_threads, device_id, seed=-1, prefetch_queue_depth=1)
        self.layer_order = layer_order
        if self.layer_order == 0:
            from_file=True
        else:
            from_file=False
        
        if __debug__:
            log.debug(f"layer_order {self.layer_order} from file {from_file}")
        self.eii_gpu = ExternalInputGpuIterator(dataset, batch_size, in_queue, from_file=from_file)
        self.pipe_BatchSize = batch_size
        self.crop = crop
        self.mean = mean
        self.std = std
        self.local_rank = local_rank
        self.size = crop
        self.resize = self.size + 32
            
        self.minmax = minmax_list()
        self.policy = subpolicy
        
        # if self.dali_device == "gpu" and fp16:
        #     output_dtype = types.FLOAT16
        # else:
        #     output_dtype = types.FLOAT        

        self.world_size = world_size
        self.augmentations = {}
        if self.policy:
            op, self.prob, magnitude = self.policy
            if op == "Brightness":
                val = (float(magnitude) / 30) * \
                    float(self.minmax[op][1] - self.minmax[op][0]) + self.minmax[op][0]
                '''
                fn.brightness_contrast
                
                out = brightness_shift * output_range + brightness \
                    * (contrast_center + contrast * (in - contrast_center))
                '''
                self.augmentations["Brightness"] = \
                    lambda images: fn.brightness_contrast(images,
                                                            brightness_shift=val)
            if op == "Contrast":
                val = (float(magnitude) / 30) * \
                    float(self.minmax[op][1] - self.minmax[op][0]) + self.minmax[op][0]
                self.augmentations["Contrast"] = \
                    lambda images: fn.brightness_contrast(images,
                                                            contrast=val)

            if op == "Rotate":
                val = (float(magnitude) / 30) * \
                    float(self.minmax[op][1] - self.minmax[op][0]) + self.minmax[op][0]
                self.augmentations["Rotate"] = \
                    lambda images: fn.rotate(images,
                                                angle=val,
                                                interp_type=types.INTERP_LINEAR,
                                                fill_value=0)
            if op == "Invert":
                val = (float(magnitude) / 30) * \
                    float(self.minmax[op][1] - self.minmax[op][0]) + self.minmax[op][0]
                # Color value inverting - implement with flip for convenience
                self.augmentations["Invert"] = \
                    lambda images: fn.flip(images,
                                            vertical=0,
                                            horizontal=1)

            if op == "ShearX":
                val = (float(magnitude) / 30) * \
                    float(self.minmax[op][1] - self.minmax[op][0]) + self.minmax[op][0]
                # ShearX img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))
                self.augmentations["ShearX"] = \
                    lambda images: fn.warp_affine(images,
                                                    matrix=[
                                                        1.0, val, 0.0, 0.0, 1.0, 0.0],
                                                    interp_type=types.INTERP_LINEAR)
            if op == "ShearY":
                val = (float(magnitude) / 30) * \
                    float(self.minmax[op][1] - self.minmax[op][0]) + self.minmax[op][0]
                # ShearY img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))
                self.augmentations["ShearY"] = \
                    lambda images: fn.warp_affine(images,
                                                    matrix=[
                                                        1.0, 0.0, 0.0, val, 1.0, 0.0],
                                                    interp_type=types.INTERP_LINEAR)
            if op == "TranslateXabs":
                val = (float(magnitude) / 30) * \
                    float(self.minmax[op][1] - self.minmax[op][0]) + self.minmax[op][0]
                # TranslateX abs img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
                self.augmentations["TranslateXabs"] = \
                    lambda images: fn.warp_affine(images,
                                                    matrix=[
                                                        1.0, 0.0, val, 0.0, 1.0, 0.0],
                                                    interp_type=types.INTERP_LINEAR)
            if op == "TranslateYabs":
                val = (float(magnitude) / 30) * \
                    float(self.minmax[op][1] - self.minmax[op][0]) + self.minmax[op][0]
                # TranslateY abs img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
                self.augmentations["TranslateYabs"] = \
                    lambda images: fn.warp_affine(images,
                                                    matrix=[
                                                        1.0, 0.0, 0.0, 0.0, 1.0, val],
                                                    interp_type=types.INTERP_LINEAR)
    
        # # print('DALI "{0}" variant'.format(self.dali_device))

    def define_graph(self):
        if self.layer_order != 0:
            images, self.labels = dali.fn.external_source(source=self.eii_gpu, num_outputs=2, device="gpu")
        else:
            images, self.labels = dali.fn.external_source(source=self.eii_gpu, num_outputs=2, device="cpu")

        if self.layer_order == 0:
            images = fn.decoders.image(images, device="mixed")
            self.labels = self.labels.gpu()
        # Combined decode & random crop
        
        if self.layer_order != 3:
            # Resize as desired
            if __debug__:
                log.debug(f"AutoAugment: {self.augmentations}")
                
            for i, aug in enumerate(self.augmentations):
                if __debug__:
                    log.debug(f"AutoAugment: aug {aug}, prob{self.prob}")
                # rng_aug = fn.random.coin_flip(probability=self.prob, dtype=dali.types.BOOL)
                images = self.augmentations[aug](images)
        elif self.layer_order == 3:
            rng = fn.random.coin_flip(probability=0.5)
            images = fn.resize(images, resize_x=self.resize, resize_y=self.resize, minibatch_size=4)
            images = fn.crop_mirror_normalize(images,
                                                mirror=rng,
                                                crop=[self.crop, self.crop],
                                                mean=self.mean,
                                                std=self.std,
                                                dtype=types.FLOAT,
                                                output_layout=types.NCHW)
        else:
            raise NotImplementedError("Implementation Error")
            
        return [images, self.labels]

class AutoAugmentWrapper():
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
                 min_crop_size=0.08, subpolicy=None, num_threads=2, first_layer=None, last_layer=None,
                 py_num_workers=2,py_start_method="fork", out_queue = None):
        self.batch_size = batch_size
        self.crop = crop
        self.mean = mean
        self.std = std
        self.local_rank = local_rank
        self.size = crop
        self.resize = self.size + 32
        self.workers=num_threads
        self.device_id = device_id
        self.dataset = dataset
        self.dali_cpu = dali_cpu
        self.fp16 = fp16,
        self.min_crop_size = min_crop_size
        if dali_cpu:
            self.decode_device = "cpu"
            self.dali_device = "cpu"
        else:
            self.decode_device = "mixed"
            self.dali_device = "gpu"
        if subpolicy == None:
            raise NotImplementedError("Unsupported policy")
        
        self.minmax = minmax_list()
        self.out_queue = out_queue
        self.policy = subpolicy
        self.last_layer = last_layer
        self.first_layer = first_layer

        self.world_size = world_size
        self.subpolicy = subpolicy
        self.in_queue = in_queue
        self.out_queue = out_queue
        if out_queue == None:
            raise ValueError("Cannot find out_queue param")
        self.wire_queues = [self.in_queue]
        
        for i, policy in enumerate(self.subpolicy):
            self.wire_queues.append(queue.Queue())
        
        self.wire_queues.append(self.out_queue)

    def reset(self):
        self.cleanup()
        self.train_pipes = [self.first_layer]
        i = 1
        for policy in self.subpolicy:
            train_pipe = AutoAugmentPipeElement( batch_size=self.batch_size, num_threads=self.workers, device_id=self.device_id, 
                                            dataset=self.dataset, crop=self.size, dali_cpu=self.dali_cpu, in_queue=self.wire_queues[i],
                                            mean=self.mean, std=self.std, local_rank=self.device_id, subpolicy=policy, layer_order=i,
                                            world_size=self.world_size, fp16=self.fp16, min_crop_size=self.min_crop_size)

            self.train_pipes.append(train_pipe)
            i += 1
        self.train_pipes.append(self.last_layer)
        if __debug__:
            log.debug("AutoAug wrapper, reset pipe")
    
    def run(self):
        if __debug__:
            log.debug(f"AutoAug wrapper, run train_pipe 0 layer, in_queue_size: {self.in_queue.qsize()}")
        out = self.train_pipes[0].run()
        
        for i in range(1,len(self.train_pipes)):
            self.wire_queues[i].put(out)
            if __debug__:
                log.debug(f"AutoAug wrapper, run train_pipe {i} layer, in_queue_size: {self.wire_queues[i].qsize()}")
            out = self.train_pipes[i].run()
        if __debug__:
            log.debug(f"AutoAug wrapper done")
        
        return out

    def build(self):
        self.reset()
        for train_pipe in self.train_pipes:
            train_pipe.build()
    
    def cleanup(self):
        if hasattr(self, 'train_pipes'):
            for train_pipe in self.train_pipes:
                del train_pipe
            del self.first_layer, self.last_layer
            self.first_layer = None
            self.last_layer = None
            self.train_pipes = None
    
    def __del__(self):
        self.cleanup()

class RandAugmentPipe(Pipeline):
    """[summary]
    Modify from Esthesia (Taeyoon Kim)
    https://github.com/Esthesia/DALI_pytorch_demo
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
        # if self.dali_device == "gpu" and fp16:
        #     output_dtype = types.FLOAT16
        # else:
        #     output_dtype = types.FLOAT        
        # self.flip = ops.Flip(device=self.dali_device)
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
            if op == "Contrast":
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
    

        
        # print('DALI "{0}" variant'.format(self.dali_device))
    
    def define_graph(self):
        rng = fn.random.coin_flip(probability=0.5)
        self.jpegs, self.labels = dali.fn.external_source(source=self.eii_gpu, num_outputs=2, device="cpu")

        # Combined decode & random crop
        images = fn.decoders.image(self.jpegs, device="mixed")
        # Resize as desired

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


class DALI_GPU_Processor():
    """
    NVIDIA DALI CPU/GPU pipelines.
    This dataloader implements ImageNet style training preprocessing, namely:
    -random resized crop
    -random horizontal flip
    And ImageNet style validation preprocessing, namely:
    -resize to specified size
    -center crop to desired size
    data_dir (str): Directory to dataset.  Format should be the same as torchvision dataloader,
    batch_size (int): how many samples per batch to load
    size (int): Output size (typically 224 for ImageNet)
    val_size (int): Validation pipeline resize size (typically 256 for ImageNet)
    workers (int): how many workers to use for data loading
    world_size (int, optional, default = 1) - Partition the data into this many parts (used for multiGPU training)
    cuda (bool): Output tensors on CUDA, CPU otherwise
    use_dali (bool): Use Nvidia DALI backend, torchvision otherwise
    dali_cpu (bool): Use Nvidia DALI cpu backend, GPU backend otherwise
    fp16 (bool, optional, default = False) - Output the data in fp16 instead of fp32
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    pin_memory (bool): Transfer CPU tensor to pinned memory before transfer to GPU (torchvision only)
    pin_memory_dali (bool): Transfer CPU tensor to pinned memory before transfer to GPU (dali only)
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 device_id,
                 in_queue,
                 size=224,
                 val_batch_size=None,
                 val_size=256,
                 min_crop_size=0.08,
                 workers=2,
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
                 augment_type="randaugment"
                 ):

        self.size = size
        # self.val_batch_size = val_batch_size
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
        self.dataset = dataset
        self.train_loader = None
        self.train_pipe = None
        self.val_loader = None
        self.val_pipe = None
        self.in_queue = in_queue
        self.device_id = device_id
        self.batch_size = batch_size
        self.augment_type = augment_type
        self.out_queue = None
        # self.val_size = val_size
        # if self.val_size is None:
        #     self.val_size = self.size

        # if self.val_batch_size is None:
        #     self.val_batch_size = self.batch_size
        # DALI Dataloader
        if self.use_dali:
            print('Using Nvidia DALI dataloader')
            # assert len(datasets.ImageFolder(
            #     self.valdir)) % self.val_batch_size == 0, 'Validation batch size must divide validation dataset size cleanly...  DALI has problems otherwise.'
            
        # Standard torchvision dataloader
        else:
            raise NotImplementedError("Not support without dali")
        if self.augment_type == "randaugment":
            self.ops_combinations=list(combinations(DALI_RANDAUGMENT_LIST, self.num_of_ops))
        elif self.augment_type == "autoaugment":
            self.ops_combinations=imagenet_autoaugment_policy()
            self.out_queue = queue.Queue()
        else:
            self.ops_combinations=["default"]
        if __debug__:
            log.debug(f"DALI GPU PRocessor: aug type {self.augment_type}")
    def get_max_select(self):
        return len(self.ops_combinations)
    
    def get_loader(self, op_number):
        """
        Creates & returns an iterator for the training dataset
        :return: Dataset iterator object
        """
        if self.augment_type == "randaugment":
            _ops = self.ops_combinations[op_number]
            if __debug__:
                log.debug(f"Build Randaug Ops:{_ops}")
            self.train_pipe = RandAugmentPipe( batch_size=self.batch_size, num_threads=self.workers, device_id=self.device_id, in_queue=self.in_queue,
                                        dataset=self.dataset, crop=self.size, dali_cpu=self.dali_cpu,
                                        mean=self.mean, std=self.std, local_rank=self.device_id,
                                        world_size=self.world_size, fp16=self.fp16, min_crop_size=self.min_crop_size, aug_name_list=_ops, aug_factor=self.degree_of_ops)
            # self.reset(data_list, batch_size, device_id)
            self.train_pipe.build()
        elif self.augment_type == "autoaugment":
            _ops = self.ops_combinations[op_number]
            
            if __debug__:
                log.debug(f"Build AutoAug Ops:{_ops}")
            if hasattr(self, 'first_layer'):
                del self.first_layer
            if hasattr(self, 'last_layer'):
                del self.last_layer
            self.first_layer = AutoAugmentPipeElement( batch_size=self.batch_size, num_threads=self.workers, device_id=self.device_id, 
                                            dataset=self.dataset, crop=self.size, dali_cpu=self.dali_cpu, in_queue=self.in_queue,
                                            mean=self.mean, std=self.std, local_rank=self.device_id, subpolicy=[], layer_order=0,
                                            world_size=self.world_size, fp16=self.fp16, min_crop_size=self.min_crop_size)
            
            self.last_layer = AutoAugmentPipeElement( batch_size=self.batch_size, num_threads=self.workers, device_id=self.device_id, 
                                            dataset=self.dataset, crop=self.size, dali_cpu=self.dali_cpu, in_queue=self.out_queue,
                                            mean=self.mean, std=self.std, local_rank=self.device_id, subpolicy=[], layer_order=3,
                                            world_size=self.world_size, fp16=self.fp16, min_crop_size=self.min_crop_size)
            
            self.train_pipe = AutoAugmentWrapper( batch_size=self.batch_size, num_threads=self.workers, device_id=self.device_id, 
                                            dataset=self.dataset, crop=self.size, dali_cpu=self.dali_cpu, in_queue=self.in_queue,
                                            mean=self.mean, std=self.std, local_rank=self.device_id, subpolicy=_ops,
                                            world_size=self.world_size, fp16=self.fp16, min_crop_size=self.min_crop_size, out_queue=self.out_queue,
                                            first_layer=self.first_layer, last_layer=self.last_layer)
            self.train_pipe.build()
        else:
            pass
        return self.train_pipe

    def _build_dali_pipeline(self):
        # assert self.world_size == 1, 'Distributed support not tested yet'
        if self.augment_type == "randaugment":
            pass
        elif self.augment_type == "autoaugment":
            pass
            # self.first_layer = AutoAugmentPipeElement( batch_size=self.batch_size, num_threads=self.workers, device_id=self.device_id, 
            #                                 dataset=self.dataset, crop=self.size, dali_cpu=self.dali_cpu, in_queue=self.in_queue,
            #                                 mean=self.mean, std=self.std, local_rank=self.device_id, subpolicy=[], layer_order=0,
            #                                 world_size=self.world_size, fp16=self.fp16, min_crop_size=self.min_crop_size)
            
            # self.last_layer = AutoAugmentPipeElement( batch_size=self.batch_size, num_threads=self.workers, device_id=self.device_id, 
            #                                 dataset=self.dataset, crop=self.size, dali_cpu=self.dali_cpu, in_queue=self.out_queue,
            #                                 mean=self.mean, std=self.std, local_rank=self.device_id, subpolicy=[], layer_order=3,
            #                                 world_size=self.world_size, fp16=self.fp16, min_crop_size=self.min_crop_size)

            # self.train_pipe = AutoAugmentWrapper( batch_size=self.batch_size, num_threads=self.workers, device_id=self.device_id, 
            #                                 dataset=self.dataset, crop=self.size, dali_cpu=self.dali_cpu, in_queue=self.in_queue,
            #                                 mean=self.mean, std=self.std, local_rank=self.device_id, subpolicy=policy,
            #                                 world_size=self.world_size, fp16=self.fp16, min_crop_size=self.min_crop_size, out_queue=self.out_queue,
            #                                 first_layer=self.first_layer, last_layer=self.last_layer))

        else:
            self.in_queue.put([0])
            self.in_queue.put([0])
            self.train_pipe = HybridTrainPipe( batch_size=self.batch_size, num_threads=self.workers, device_id=self.device_id, in_queue=self.in_queue,
                                            dataset=self.dataset, crop=self.size, dali_cpu=self.dali_cpu,
                                            mean=self.mean, std=self.std, local_rank=self.device_id,
                                            world_size=self.world_size, fp16=self.fp16, min_crop_size=self.min_crop_size)
            # train_pipe.serialize()
            # self.train_pipe.build()
            
            if __debug__:
                log.debug(f"Default dali_queue qsize: {self.in_queue.qsize()}")

    def reset(self):
        # clear_memory()

        # Currently we need to delete & rebuild the dali pipeline every epoch,
        # due to a memory leak somewhere in DALI
        if __debug__:
            log.debug('Recreating DALI dataloaders to reduce memory usage')
            
        if self.train_loader == None:  
            pass
        else:
            del self.train_loader, self.train_pipe
        self.val_loader = None
        self.val_pipe = None
        # del self.val_loader, self.val_pipe
        # clear_memory()

        # taken from: https://stackoverflow.com/questions/1254370/reimport-a-module-in-python-while-interactive
        # importlib.reload(dali)
        # from dali import RandAugment
        self._build_dali_pipeline()
    
    def cleanup(self):
        if __debug__:
            log.debug('Cleanup DALI dataloaders to reduce memory usage')
        del self.train_pipe
        self.train_pipe = None

def _try_get_queue(in_queue):
    try:
        idx = in_queue.get()
    except queue.Empty:
        idx = -1
    return idx

def _load_np_arr(type, rank, worker_id, frame, dtype=np.int):
    shm_name = f'{type}{rank}_{int(worker_id)}'
    shape = frame.shape
    size = frame.nbytes
    if __debug__:
        log.debug(f"Worker load as {shm_name} size: {size}")
    shm = SharedMemory(name=shm_name)
    shm_buf = mmap.mmap(shm.fd, size)
    shm.close_fd()
    shm_np_arr = np.ndarray(
        shape=shape, dtype=dtype, buffer=shm_buf)
    
    return shm_buf, shm_np_arr

def feed_ndarray(dali_tensor, arr, cuda_stream = None, arr_data_ptr = None):
        """
        Snippet from https://github.com/NVIDIA/DALI/blob/main/dali/python/nvidia/dali/plugin/pytorch.py
        Modify for aggregation 

        Copy contents of DALI tensor to PyTorch's Tensor.
        Parameters
        ----------
        `dali_tensor` : nvidia.dali.backend.TensorCPU or nvidia.dali.backend.TensorGPU
                        Tensor from which to copy
        `arr` : torch.Tensor
                Destination of the copyz
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

def _gpu_worker_loop(in_queue, dali_queue, out_queue, dataset, microbatch_size, _rank, world_size, workers, done_event, gpu_finish_event, gpu_worker_id, augmentation_type, aggr_shape_hint):
    shm_buf, worker_progress_info = _load_np_arr("worker_progress_info", _rank, gpu_worker_id, frame=np.array([0, 0], dtype=np.int))
    dali_processor = DALI_GPU_Processor(dataset, microbatch_size, _rank, world_size=world_size, workers=workers, in_queue=dali_queue, size = aggr_shape_hint[2], augment_type= augmentation_type)
    if __debug__:
        log.debug(f"GPU: {_rank} GPU worker dali_queue qsize: {dali_queue.qsize()}")
    
    dali_processor.reset()
    if __debug__:
        log.debug(f"GPU: {_rank} GPU worker reset: {dali_queue.qsize()}")
    
    torch_stream = torch.cuda.current_stream(device=_rank)
    while not done_event.is_set():
        worker_progress_info[0] = -1
        gpu_finish_event.wait()
        
        if __debug__:
            start = time.perf_counter()
        pipe_num = random.randrange(dali_processor.get_max_select())
        one_time_gpu_pipe = dali_processor.get_loader(pipe_num)

        idx = _try_get_queue(in_queue)
        if idx == None:
            break
        elif idx == -1:
            continue

        if __debug__:
            log.debug(f"GPU: {_rank} GPU worker idx {idx}")
        if idx != -999:
            worker_progress_info[0] = idx
            worker_progress_info[1] = 999

        while not done_event.is_set():
            _input, _target = one_time_gpu_pipe.run()
            if __debug__:
                log.debug(f"GPU:{_rank} GPU workers check dummy {_target[0].shape()} {_target[0].layout()}")
            
            if idx == -999 or len(_input) == 1:
                del _input, _target
                continue

            if __debug__:
                end = time.perf_counter()
                log.debug(f"GPU:{_rank} GPU workers done END FETCH_TIME {end-start} idx{idx}")
            break

        if idx == None:
            if __debug__:
                log.debug(f"GPU: {_rank} GPU worker Finish")
            break
        input_tensor = _input.as_tensor()
        target_tensor = _target.as_tensor()
        torch_input = torch.empty(microbatch_size,aggr_shape_hint[1],aggr_shape_hint[2],aggr_shape_hint[3], dtype=torch.float32, device=_rank)
        torch_target = torch.empty(microbatch_size, dtype=torch.uint8, device=_rank)
        feed_ndarray(input_tensor, torch_input, torch_stream, torch_input.data_ptr())
        feed_ndarray(target_tensor, torch_target, torch_stream, torch_target.data_ptr())
        torch_target=torch_target.type(torch.long)
        if __debug__:
            log.debug(f"GPU:{_rank} see target {target_tensor} and copyed {torch_target} torch allocation{torch.cuda.memory_allocated()} torch reserved{torch.cuda.memory_reserved()}")
        del _input, _target, input_tensor, target_tensor
        if __debug__:
            copy_end = time.perf_counter()
            log.debug(f"GPU:{_rank} GPU workers done END COPY_TIME {copy_end-end} idx{idx}")
        while not done_event.is_set():
            try:
                out_queue.put((idx,(torch_input,torch_target)))
                break
            except queue.Full:
                continue
        del torch_input, torch_target
        dali_processor.cleanup()
        
        
    if done_event.is_set():
        shm_buf.close()
        out_queue.cancel_join_thread()
        # if __debug__:
        #     log.debug(f"GPU: {rank} worker_id: {worker_id}, cancel_join_thread data_queue:\n{data_queue}")
        out_queue.close()