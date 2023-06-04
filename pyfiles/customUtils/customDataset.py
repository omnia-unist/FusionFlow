# pip install opencv-python
import numpy as np
import io
from PIL import Image
import gzip
import torch
from multiprocessing import Manager, Array
from ctypes import c_wchar_p, c_int
from torchvision.datasets import ImageFolder 
from typing import Any, Callable, Iterable, cast, Dict, List, Optional, Tuple
# import cv2


if __debug__:
    import logging
    import time
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s:\t\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger(__name__)
    
def imagenet_21k_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        if f.read(2) == b'\x1f\x8b':
            with gzip.open(path, 'rb') as f2:
                    img = Image.open(f2)
                    return img.convert('RGB')
        else:
            img = Image.open(f)
            return img.convert('RGB')
        
def fetch_decode_loader(path: str) -> Image.Image:
    # if __debug__:
    load_start = time.perf_counter()
    with open(path, 'rb') as f:
        img_byte = f.read()
        img_byte_mem = io.BytesIO(img_byte)
    # if __debug__:
    load_end = time.perf_counter()
        
    img = Image.open(img_byte_mem)
    img = img.convert("RGB")
    # if __debug__:
    decode_end = time.perf_counter()
    fetch = load_end-load_start
    decode = decode_end-load_end
        # log.debug(f"Dataset FETCH\t{load_end-load_start}\t\tDECODE\t{decode_end-load_end}")
    return img, fetch, decode
    
class RdonlyDataset(ImageFolder):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(RdonlyDataset, self).__init__(*args,**kwargs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # assert self.isMem is True
        possible_img, target = self.samples[index]
        imagenet_21k_loader(possible_img)

        return [0], [0]

    def memloader(self, byte_data):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(byte_data, 'rb') as f:
            img_byte = f.read()





    


class CopyedDistMemDataset(ImageFolder):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(CopyedDistMemDataset, self).__init__(*args,**kwargs)
        # Store with key value
        self.inmemory={}
        self.isMem=False       

    def inMem(self, indices):
        for index in indices:
            possible_img, _ = self.samples[index]

            with open(possible_img, 'rb') as f:
                img_byte = f.read()
                img_byte_mem = io.BytesIO(img_byte)
            self.inmemory[index] = img_byte_mem

        self.isMem = True

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        assert self.isMem is True
        _, target = self.samples[index]
        possible_img = self.inmemory[index]
        possible_img = self.memloader(possible_img)

        if self.transform is not None:
            sample = self.transform(possible_img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def memloader(self, byte_data):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        img = Image.open(byte_data)
        return img.convert('RGB')

class NoDecodeCopyedDistMemDataset(ImageFolder):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(NoDecodeCopyedDistMemDataset, self).__init__(*args,**kwargs)
        # Store with key value
        self.inmemory={}
        self.isMem=False        

    def inMem(self, indices):
        for index in indices:
            possible_img, _ = self.samples[index]

            with open(possible_img, 'rb') as f:
                img = Image.open(f)
                self.inmemory[index] = img.convert('RGB')

        self.isMem = True

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        assert self.isMem is True
        _, target = self.samples[index]
        possible_img = self.inmemory[index]
        if self.transform is not None:
            sample = self.transform(possible_img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

class FullTraceImageFolder(ImageFolder): 
    def __init__(
            self,
            cur_rank,
            *args,
            **kwargs
        ):
            super(FullTraceImageFolder, self).__init__(*args,**kwargs)
            self.cur_rank = cur_rank
            self.loader = fetch_decode_loader
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        
        if __debug__:
            start = time.perf_counter()
        path, target = self.samples[index]
        if __debug__:
            get_tar_end = time.perf_counter()
        sample, fetch, decode = self.loader(path)
        if __debug__:
            get_img_end = time.perf_counter()
        if self.transform is not None:
            sample = self.transform(sample)
            if __debug__:
                trans_end = time.perf_counter()
        if self.target_transform is not None:
            target = self.target_transform(target)
            if __debug__:
                log.debug(f"TARGET TRANSFORM IS NOT COLLECTED!!!")
        
        if __debug__:
            log.debug(f"GPU: {self.cur_rank} Dataset GET_tar\t0\t\tGET_img\t{get_tar_end-start}\t\tFETCH\t{fetch}\t\tDECODE\t{decode}\t\tLOAD\t{get_img_end-get_tar_end}\t\tAUG\t{trans_end-get_img_end}\t\tINDEX\t{index}")
        return sample, target

class FullTraceNoDecodeCopyedDistMemDataset(NoDecodeCopyedDistMemDataset):     
    def __init__(
        self,
        cur_rank,
        *args,
        **kwargs
    ):
        super(FullTraceNoDecodeCopyedDistMemDataset, self).__init__(*args,**kwargs)
        self.cur_rank = cur_rank
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        assert self.isMem is True
        if __debug__:
            start = time.perf_counter()
        _, target = self.samples[index]
        if __debug__:
            get_tar_end = time.perf_counter()
        possible_img = self.inmemory[index]
        if __debug__:
            get_img_end = time.perf_counter()
        if self.transform is not None:
            sample = self.transform(possible_img)
            if __debug__:
                trans_end = time.perf_counter()
        if self.target_transform is not None:
            target = self.target_transform(target)
            if __debug__:
                log.debug(f"TARGET TRANSFORM IS NOT COLLECTED!!!")
        if __debug__:
            log.debug(f"GPU: {self.cur_rank} Dataset GET_tar\t{get_tar_end-start}\t\tGET_img\t{get_img_end-get_tar_end}\t\tLOAD\t0\t\tAUG\t{trans_end-get_img_end}\t\tINDEX\t{index}")
        return sample, target

class NoDecodeDataset(ImageFolder):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(NoDecodeDataset, self).__init__(*args,**kwargs)
        # Store with key value
        manager = Manager()
        self.inmemory=manager.dict()
        self.isMem=False        

    def inMem(self, indices):
        for index in indices:
            possible_img, _ = self.samples[index]

            with open(possible_img, 'rb') as f:
                img_byte = f.read()
                img_byte_mem = io.BytesIO(img_byte)
            self.inmemory[index] = img_byte_mem

        self.isMem = True


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        assert self.isMem is True

        _, target = self.samples[index]

        possible_img = self.inmemory[index]
        if self.transform is not None:
            sample = self.transform(possible_img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return sample, target

    def memloader(self, byte_data):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
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
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        assert self.isMem is True
        if __debug__:
            start = time.perf_counter()
        _, target = self.samples[index]
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
        if self.target_transform is not None:
            target = self.target_transform(target)
            if __debug__:
                log.debug(f"TARGET TRANSFORM IS NOT COLLECTED!!!")
        if __debug__:
            log.debug(f"GPU: {self.cur_rank} Dataset GET_tar\t{get_tar_end-start}\t\tGET_img\t{get_img_end-get_tar_end}\t\tLOAD\t{load_end-get_img_end}\t\tAUG\t{trans_end-load_end}\t\tINDEX\t{index}")

        return sample, target

class SyntheticDistMemDataset(ImageFolder):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(SyntheticDistMemDataset, self).__init__(*args,**kwargs)
        # Store with key value
        self.inmemory={}
        self.isMem=False         

    def inMem(self, indices):
        self.isMem = True

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        assert self.isMem == True
        if __debug__:
            start = time.perf_counter()
        _, target = self.samples[index]
        if __debug__:
            get_tar_end = time.perf_counter()
        sample = torch.rand((3, 224, 224))
        return sample, target

class SyntheticDataset(ImageFolder):
    def __init__(
            self,
            size,
            *args,
            **kwargs
    ):
        super(SyntheticDataset, self).__init__(*args,**kwargs)

        self.inmemory=[]

        self.synth_img = torch.rand((3, size, size))
        self.isMem=False         

    def inMem(self, indices):
        self.isMem = True


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = self.synth_img
        target = self.samples[index][1]
        return sample, target

class PrepDataset(ImageFolder):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(PrepDataset, self).__init__(*args,**kwargs)
        self.inmemory=[]
        print(f"len: {len(self.samples)}")

        for i in range(len(self.samples)):
            print(f"index:{i}")
            sample, target = self.samples[i]
            loaded_sample=self.loader(sample)
            if self.transform is not None:
                sample = self.transform(loaded_sample)
            else:
                sample = loaded_sample
            if self.target_transform is not None:
                target = self.target_transform(target)
            self.inmemory.append((sample,target))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample, target = self.inmemory[index]
        return sample, target

        
class SharedMemoryDistMemDataset(ImageFolder):
    def __init__(
            self,
            *args,
            shm_mngr=None,
            **kwargs
    ):
        if shm_mngr is None:
            raise Exception("Cannot start dataset without multiprocessing.managers.SharedMemoryManager().")
        super(SharedMemoryDistMemDataset, self).__init__(*args,**kwargs)
        # Store with key value
        self.shm_mngr= shm_mngr # managers.SharedMemoryManager()
        self.isMem=False
        self.inmemory=None

    def inMem(self, indices):
        encoded_images = [i for i in range(len(self.samples))]
        for index in indices:
            possible_img, _ = self.samples[index]
            
            with open(possible_img, 'rb') as f:
                img_byte = f.read()
            img_byte_mem = io.BytesIO(img_byte)
            # print(f"getbuffer nbytes: {img_byte_mem.getbuffer().nbytes}")
            # print(f"sizeof: {img_byte_mem.getbuffer().nbytes}")
            read_img = img_byte_mem.read()
            # print(f"tobytes: {sys.getsizeof(read_img)}")
            encoded_images[index] = read_img
        print("done")
        # self.inmemory=self.shm_mngr.ShareableList(encoded_images)
        self.inmemory=self.shm_mngr.ShareableList([])
        print(self.inmemory)
        self.isMem = True
        self.shm_mngr = None

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        assert self.isMem is True
        _, target = self.samples[index]
        possible_img = self.loop.run_until_complete(self.inmemory.get(index))
        possible_img = self.memloader(possible_img)
        if self.transform is not None:
            sample = self.transform(possible_img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def memloader(self, byte_data):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        img = Image.open(byte_data)
        return img.convert('RGB')
    
    # def __del__(self):
    #     self.shm_mngr.shutdown()
    

####################
# Work In Progress #
####################


class SharedImageFolder(ImageFolder):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(SharedImageFolder, self).__init__(*args,**kwargs)
        self.imgs = None
        samples = self.samples
        
        self.samples = Array(c_wchar_p, len(samples))
        self.targets = Array(c_int, len(samples))
        
        for i, s in enumerate(samples):
            self.samples[i] = s[0]
            self.targets[i] = s[1]
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        target = self.targets[index]
        
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return sample, target



class MemDataset(ImageFolder):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(MemDataset, self).__init__(*args,**kwargs)
        self.inmemory=[]
        # print(f"len: {len(self.samples)}")
        ##
        # debug_size=0
        ##
        for i in range(len(self.samples)):
            # print(f"index:{i}")
            possible_img, _ = self.samples[i]
            with open(possible_img, 'rb') as f:
                img_byte = f.read()
                img_byte_mem = io.BytesIO(img_byte)

                # # For debug
                # debug_size+=sys.getsizeof(img_byte_mem)
                # # # # #
            self.inmemory.append(img_byte_mem)
        
        # print(f"allocated_size: {debug_size}")

        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        _, target = self.samples[index]
        possible_img = self.inmemory[index]
        possible_img = self.memloader(possible_img)
        if self.transform is not None:
            sample = self.transform(possible_img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def memloader(self, byte_data):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        img = Image.open(byte_data)
        return img.convert('RGB')

class DistMemDataset(ImageFolder):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(DistMemDataset, self).__init__(*args,**kwargs)
        # Store with key value
        manager = Manager()
        self.inmemory=manager.dict()
        self.isMem=False        

    def inMem(self, indices):
        for index in indices:
            possible_img, _ = self.samples[index]

            with open(possible_img, 'rb') as f:
                img_byte = f.read()
                img_byte_mem = io.BytesIO(img_byte)
            self.inmemory[index] = img_byte_mem

        self.isMem = True

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        assert self.isMem is True
        _, target = self.samples[index]
        possible_img = self.inmemory[index]
        possible_img = self.memloader(possible_img)
        if self.transform is not None:
            sample = self.transform(possible_img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def memloader(self, byte_data):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        img = Image.open(byte_data)
        return img.convert('RGB')

# class cvDistMemDataset(ImageFolder):
#     def __init__(
#             self,
#             *args,
#             **kwargs
#     ):
#         super(cvDistMemDataset, self).__init__(*args,**kwargs)
#         # Store with key value
#         manager = Manager()
#         self.inmemory=manager.dict()
#         self.isMem=False        

#     def inMem(self, indices):
#         for index in indices:
#             possible_img, _ = self.samples[index]

#             with open(possible_img, 'rb') as f:
#                 img_byte_mem = f.read()

#             self.inmemory[index] = img_byte_mem

#         self.isMem = True

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (sample, target) where target is class_index of the target class.
#         """
#         assert self.isMem is True
#         _, target = self.samples[index]
#         possible_img = self.inmemory[index]
#         possible_img = self.memloader(possible_img)
#         if self.transform is not None:
#             sample = self.transform(possible_img)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         return sample, target

#     def memloader(self, byte_data):
#         # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
#         # Load raw file to memory
#         # Decoding
#         img = cv2.imdecode(np.fromstring(byte_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
#         # Convert RGB, PIL: img.convert('RGB')
#         RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
#         # return to PIL for compatibility with data loader because of transformation with PIL
#         return Image.fromarray(RGB)


##############
# DEPRECATED #
##############
