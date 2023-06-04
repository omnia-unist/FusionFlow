#!python
import sys
from os import path
sys.path.append(path.dirname(path.abspath("../pyfiles/RandAugment")))
import io
import pathlib
import logging
import time
import customUtils
import torch
import torchvision.transforms as transforms
from PIL import Image
from multiprocessing import Process, Barrier


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

num_p = 1
rep = 1
batch_size = 64
trace_compose = False
synthetic = False
decoding = True
dir_path = '/data/sampled_imagenet/'
image_sizes = [11, 100, 200, 314, 914, 1200]

if trace_compose:
    ComposeTrace = customUtils.FullTraceCompose
else:
    ComposeTrace = transforms.Compose

augmentations = [
    # transforms.Resize(256),
    # transforms.CenterCrop(size),
    transforms.RandomResizedCrop(size),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    normalize
]
single_composed = ComposeTrace(augmentations)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def memloader(byte_data):
    img = Image.open(byte_data)
    return img.convert('RGB')


def aug(imgd, barrier, num, preloading, num_p, rep, aug_composed, batch_size):
    torch.set_num_threads(1)
    import psutil
    p = psutil.Process()
    p.cpu_affinity([num])

    for i in range(rep):
        arr = []

        barrier.wait()
        log.debug(f"Start Process{num}")
        start = time.perf_counter()

        for i in range(batch_size):
            start_load = time.perf_counter()
            if preloading:
                img = memloader(imgd[i])
            else:
                img = imgd[i]
            start_aug = time.perf_counter()
            
            log.debug(f"Decode: {start_aug-start_load}")
        end = time.perf_counter()
        elapsed_time = end-start
        log.debug(f"End Process{num} Process time: {elapsed_time}")

    return arr


barrier = Barrier(num_p)
for image_size in image_sizes:
    log.info(f"image_size: {image_size}")
    d = []

    for i in range(batch_size):
        if synthetic:
            d.append(torch.rand((3, 224, 224)))
        elif decoding:
            path_raw = dir_path+f'/{image_size}k.JPEG'

            path = pathlib.Path(path_raw)
            start = time.perf_counter()
            with open(path, 'rb') as f:
                img_byte = f.read()
                img_byte_mem = io.BytesIO(img_byte)
            end = time.perf_counter()
            # log.debug(f"{path_raw} I/O_time: {end - start}")
            d.append(img_byte_mem)
        else:
            path_raw = dir_path+f'/{image_size}k.JPEG'
            print(path_raw)
            path = pathlib.Path(path_raw)
            d.append(pil_loader(path))
    print(f'length {len(d)}')

    ps = []
    for i in range(num_p):
        p = Process(target=aug, args=(d, barrier, i, decoding,
                    num_p, rep, single_composed, batch_size))
        ps.append(p)

    for p in ps:
        p.start()

    for p in ps:
        p.join()
