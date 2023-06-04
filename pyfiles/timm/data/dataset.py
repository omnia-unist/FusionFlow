""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch.utils.data as data
import os
import io
import torch
import logging
import time

from PIL import Image

from .parsers import create_parser

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s:\t\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
_logger = logging.getLogger(__name__)


_ERROR_RETRY = 50


class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            parser=None,
            class_map='',
            load_bytes=False,
            transform=None,
    ):
        if parser is None or isinstance(parser, str):
            parser = create_parser(parser or '', root=root, class_map=class_map)
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.parser[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.parser))
            else:
                raise e
        self._consecutive_errors = 0
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.tensor(-1, dtype=torch.long)
        return img, target

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            parser=None,
            split='train',
            is_training=False,
            batch_size=None,
            class_map='',
            load_bytes=False,
            repeats=0,
            transform=None,
    ):
        assert parser is not None
        if isinstance(parser, str):
            self.parser = create_parser(
                parser, root=root, split=split, is_training=is_training, batch_size=batch_size, repeats=repeats)
        else:
            self.parser = parser
        self.transform = transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.parser:
            if self.transform is not None:
                img = self.transform(img)
            if target is None:
                target = torch.tensor(-1, dtype=torch.long)
            yield img, target

    def __len__(self):
        if hasattr(self.parser, '__len__'):
            return len(self.parser)
        else:
            return 0

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)

class SyntheticDataset(ImageDataset):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(SyntheticDataset, self).__init__(*args,**kwargs)
        self.inmemory=[]
        
        print(f"len: {len(self.parser)}")
        for i in range(len(self.parser)):
            print(f"index:{i}")
            _, target = self.parser[i]
            synth_img = torch.rand((3, 224, 224))
            possible_img = synth_img
            self.inmemory.append((possible_img, target))

    def __getitem__(self, index):
        sample, target = self.inmemory[index]
        return sample, target


class CopyedDistMemDataset(ImageDataset):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(CopyedDistMemDataset, self).__init__(*args,**kwargs)
        # Store with key value
        self.inmemory={}

    def inMem(self, indices):
        for index in indices:
            possible_img, _ = self.parser[index]

            with open(possible_img, 'rb') as f:
                img_byte = f.read()
                img_byte_mem = io.BytesIO(img_byte)
            self.inmemory[index] = img_byte_mem

        self.load_bytes = True

    def __getitem__(self, index):
        assert self.load_bytes is True
        _, target = self.parser[index]
        possible_img = self.inmemory[index]
        possible_img = self.memloader(possible_img)

        if self.transform is not None:
            sample = self.transform(possible_img)
        if target is None:
            target = torch.tensor(-1, dtype=torch.long)
        return sample, target

    def memloader(self, byte_data):
        img = Image.open(byte_data)
        return img.convert('RGB')


class FullTraceCopyedDistMemDataset(CopyedDistMemDataset):
    def __init__(
        self,
        cur_rank,
        *args,
        **kwargs
    ):
        super(FullTraceCopyedDistMemDataset, self).__init__(*args,**kwargs)
        self.cur_rank = cur_rank
            
    def __getitem__(self, index):
        assert self.load_bytes is True
        if __debug__:
            start = time.perf_counter()
        _, target = self.parser[index]
        if __debug__:
            get_tar_end = time.perf_counter()
        possible_img = self.inmemory[index]
        if __debug__:
            get_img_end = time.perf_counter()
        possible_img = self.memloader(possible_img)
        if __debug__:
            load_end = time.perf_counter()
        if self.transform is not None:
            sample = self.transform(possible_img)
            if __debug__:
                trans_end = time.perf_counter()
        if target is None:
            target = torch.tensor(-1, dtype=torch.long)
            if __debug__:
                _logger.debug(f"TARGET TRANSFORM IS NOT COLLECTED!!")
        if __debug__:
            _logger.debug(f"GPU: {self.cur_rank} Dataset GET_tar\t{get_tar_end-start}\t\tGET_img\t{get_img_end-get_tar_end}\t\tLOAD\t{load_end-get_img_end}\t\tAUG\t{trans_end-load_end}\t\tINDEX\t{index}")

        return sample, target
