#!python
import io
import pathlib
import logging
import time
import customUtils
from RandAugment.augmentations import augment_list
import torch
import torchvision.transforms as transforms
from PIL import Image
from multiprocessing import Process, Manager, Barrier
import itertools
import sys
from os import path
sys.path.append(path.dirname(path.abspath("../pyfiles/RandAugment")))
del sys.path[0]
# print(sys.path)


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

aug_list = augment_list()
num_p = 1
rep = 1
batch_size = 64
trace_compose = True
synthetic = False
decoding = True
dir_path = '/data/sampled_imagenet/'
image_sizes = [11, 100, 200, 314, 914, 1200]


class RandAug:
    def __init__(self, ops):
        self.m = 9
        self.ops = ops

    def __call__(self, img):
        for op, minval, maxval in self.ops:
            start = time.perf_counter()
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)
            end = time.perf_counter()
            log.debug(f"Transform {op.__name__} END at_time {end-start}")
        return img


if trace_compose:
    ComposeTrace = customUtils.FullTraceCompose
else:
    ComposeTrace = transforms.Compose

single_composeds = []
double_composeds = []
for ops in itertools.combinations(aug_list, 1):
    augmentations = [
        # transforms.Resize(256),
        # transforms.CenterCrop(size),
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        normalize
    ]
    augmentations.insert(0, RandAug(ops))
    single_composeds.append(ComposeTrace(augmentations))


for ops in itertools.combinations(aug_list, 2):
    augmentations = [
        # transforms.Resize(256),
        # transforms.CenterCrop(size),
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        normalize
    ]
    augmentations.insert(0, RandAug(ops))
    double_composeds.append(customUtils.FullTraceCompose(augmentations))

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
            sample = aug_composed(img)
            end_aug = time.perf_counter()
            arr.append(sample)
        end = time.perf_counter()

        elapsed_time = end-start
        log.debug(
            f"Load: {start_aug-start_load} Aug: {end_aug-start_aug} End Process{num} Process time: {elapsed_time}")
#         result.append(elapsed_time)

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
    #         print(f"path size: {os.stat(path).st_size}")
            start = time.perf_counter()
            with open(path, 'rb') as f:
                img_byte = f.read()
                img_byte_mem = io.BytesIO(img_byte)
            end = time.perf_counter()
            log.debug(f"{path_raw} I/O_time: {end - start}")
    #         print(f"img_byte_mem size: get_size of {sys.getsizeof(img_byte_mem.tell())} bytes")
            d.append(img_byte_mem)
        else:
            path_raw = dir_path+f'/{image_size}k.JPEG'
            print(path_raw)
            path = pathlib.Path(path_raw)
            d.append(pil_loader(path))
    print(f'length {len(d)}')

    for single_composed in single_composeds:
        print(f"Transform: {single_composed.transforms[0].ops}")
        ps = []
        for i in range(num_p):
            #         p = Process(target=shdf, args=(smd[i % 4],barrier,i))
            p = Process(target=aug, args=(d, barrier, i, decoding,
                        num_p, rep, single_composed, batch_size))
            ps.append(p)

        for p in ps:
            p.start()

        for p in ps:
            p.join()

# for single_d in d:
#     print(d)
