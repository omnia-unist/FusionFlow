import numpy as np
from PIL import Image
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder 
import time
import random
import pandas as pd

if __debug__:
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s:\t\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger(__name__)

class RefurbishDataset(ImageFolder):
    def __init__(
            self,
            *args,
            cur_rank=None,
            cache_ratio=0.2708,
            batch_size = None,
            crop_size,
            **kwargs
    ):
        super(RefurbishDataset, self).__init__(*args,**kwargs)
        self.cur_rank = cur_rank
        # Store with key value
        self.partial_ratio = 1-cache_ratio
        self.origin_ratio = cache_ratio
        
        self.inmemory = None
        if batch_size == None:
            raise RuntimeError(f"{batch_size}")
        # NOTE: Read and Decode the Cached files
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        self.final_transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        self.weights = [self.partial_ratio, self.origin_ratio]
        self.population = [self.final_transform, self.transform]
        self.transforms = random.choices(population=self.population, weights=self.weights, k=batch_size)
        # print(self.transforms)s
        self.counter = 0
        self.batch_size = batch_size
        
    def partial_loader(self, index):
        if __debug__:
            start = time.perf_counter()
        path, target = self.samples[index]
        
        if __debug__:
            get_tar_end = time.perf_counter()
        possible_img = self.loader(path)
        if __debug__: 
            get_img_end = time.perf_counter()
        if self.final_transform is not None:
            sample = self.final_transform(possible_img)
                # log.debug(transform)
            if __debug__:
                trans_end = time.perf_counter()
        if self.target_transform is not None:
            target = self.target_transform(target)
            if __debug__:
                log.debug(f"TARGET TRANSFORM IS NOT COLLECTED!!!")
        if __debug__:
            log.debug(f"GPU: {self.cur_rank} Dataset GET_tar\t{get_tar_end-start}\t\tGET_img\t{get_img_end-get_tar_end}\t\tLOAD\t0\t\tAUG\t{trans_end-get_img_end}\t\tINDEX\t{index}")
            
        return sample, target
    
    def full_loader(self, index):
        if __debug__:
            start = time.perf_counter()
        path, target = self.samples[index]
        if __debug__:
            get_tar_end = time.perf_counter()
        possible_img = self.loader(path)
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
        possible_img = self.loader(path)
        if __debug__: 
            get_img_end = time.perf_counter()
        if self.transform is not None:
            transform = self.transforms[self.counter]
            sample = transform(possible_img)
            self.counter+=1
            if self.counter >= self.batch_size:
                self.counter=0
                # log.debug(transform)
            if __debug__:
                trans_end = time.perf_counter()
        if self.target_transform is not None:
            target = self.target_transform(target)
            if __debug__:
                log.debug(f"TARGET TRANSFORM IS NOT COLLECTED!!!")
        if __debug__:
            log.debug(f"GPU: {self.cur_rank} Dataset GET_tar\t{get_tar_end-start}\t\tGET_img\t{get_img_end-get_tar_end}\t\tLOAD\t0\t\tAUG\t{trans_end-get_img_end}\t\tINDEX\t{index}")
        return sample, target


class RefurbishencodeDataset(ImageFolder):
    def __init__(
            self,
            *args,
            cur_rank=None,
            cache_ratio=0.2708,
            batch_size = None,
            crop_size = 224,
            **kwargs
    ):
        super(RefurbishDataset, self).__init__(*args,**kwargs)
        self.cur_rank = cur_rank
        # Store with key value
        self.partial_ratio = 1-cache_ratio
        self.origin_ratio = cache_ratio
        
        self.inmemory = None
        if batch_size == None:
            raise RuntimeError(f"{batch_size}")
        # NOTE: Read and Decode the Cached files
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        self.final_transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        
        self.weights = [self.partial_ratio, self.origin_ratio]
        self.population = [self.final_transform, self.transform]
        self.transforms = random.choices(population=self.population, weights=self.weights, k=batch_size)
        # print(self.transforms)s
        self.counter = 0
        self.batch_size = batch_size


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
        possible_img = self.loader(path)
        if __debug__: 
            get_img_end = time.perf_counter()
        if self.transform is not None:
            transform = self.transforms[self.counter]
            sample = transform(possible_img)
            self.counter+=1
            if self.counter >= self.batch_size:
                self.counter=0
                # log.debug(transform)
            if __debug__:
                trans_end = time.perf_counter()
        if self.target_transform is not None:
            target = self.target_transform(target)
            if __debug__:
                log.debug(f"TARGET TRANSFORM IS NOT COLLECTED!!!")
        if __debug__:
            log.debug(f"GPU: {self.cur_rank} Dataset GET_tar\t{get_tar_end-start}\t\tGET_img\t{get_img_end-get_tar_end}\t\tLOAD\t0\t\tAUG\t{trans_end-get_img_end}\t\tINDEX\t{index}")
        return sample, target

class RefurbishMimicDataset(ImageFolder):
    def __init__(
            self,
            *args,
            cur_rank=None,
            cache_ratio=0.2708,
            # FIXME: Related to dataset
            batch_size = None,
            crop_size,
            **kwargs
    ):
        super(RefurbishMimicDataset, self).__init__(*args,**kwargs)
        self.cur_rank = cur_rank
        # Store with key value
        self.partial_ratio = cache_ratio
        self.origin_ratio = 1.0-cache_ratio
        
        self.inmemory = None
        if batch_size == None:
            raise RuntimeError(f"{batch_size}")
        # NOTE: Read and Decode the Cached files
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        self.final_transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        # FIXME: Hardcode with getting final augmentation time without decoding from empirical result due to the implementation limitation
        parse_dir = "/home/chanho/reimplement-ds-analyzer/dsanalyzer_parsed/DDP6GPUInterFULLTRACEFETCH/{suffix}/{filename}.csv"
        filename = "origin_main_pin_openimage_default_resnet50_epoch1_b1440_worker24_thread4_simp_split"
        dataload_split_file = parse_dir.format(suffix="simp", filename= filename)
        dataload_split_df = pd.read_csv(dataload_split_file, index_col=None)
        dataload_split_df.set_index('Image number', inplace=True)
        aug_list = dataload_split_df["Augmentation (sec)"].to_dict()
        self.aug_list = aug_list
        # print(self.aug_list)
        
        # NOTE: To ensure the ratio of refurbish caching and non-caching in one batch,
        # Pre-choice partial or full images as the number of images in batch
        self.weights = [self.partial_ratio, self.origin_ratio]
        self.population = [self.final_item, self.full_item]
        self.get_items = random.choices(population=self.population, weights=self.weights, k=batch_size)
        print("Cache ratio: ",cache_ratio)
        # if cache_ratio > memory_ratio:
        #     self.memory_ratio =  memory_ratio/ cache_ratio
        #     self.io_ratio = 1.0 - self.memory_ratio
        # else:
        #     self.memory_ratio = 1.0
        #     self.io_ratio = 1.0 - self.memory_ratio

        # NOTE: To ensure the disk and memory ratio in one batch,
        # Pre-choice memory or disk images as the number of images in batch
        # self.dataload_population = [self.memory_cost, self.io_cost]
        # self.dataload_weights = [self.memory_ratio, self.io_ratio]
        # print(self.weights)
        # print(self.dataload_weights)
        # self.dataloads = random.choices(population=self.dataload_population, weights=self.dataload_weights, k=batch_size)
        self.counter=0
        self.batch_size = batch_size
        self.aug_list_values = list(self.aug_list.values())
    # FIXME: Hardcode with estimated memory cost
    # TODO: Works with input path ???
    def memory_cost(self, path=None):
        time.sleep(0.0000390625)

    def io_cost(self,path):
        with open(path, 'rb') as f:
            img_bytes = f.read()
        return img_bytes

    def final_aug_cost(self,index):
        try:
            time.sleep(self.aug_list[index])
        except:
            time.sleep(random.choice(self.aug_list_values))
        return torch.rand((3, 224, 224))

    # Original image pipeline in Refurbish
    def full_item(self, index):
        if __debug__:
            start = time.perf_counter()
        path, target = self.samples[index]
        if __debug__:
            get_tar_end = time.perf_counter()
        possible_img = self.loader(path)
        if __debug__: 
            get_img_end = time.perf_counter()
        if self.transform is not None:
            sample = self.transform(possible_img)
                # log.debug(transform)
            if __debug__:
                trans_end = time.perf_counter()
        if self.target_transform is not None:
            target = self.target_transform(target)
            if __debug__:
                log.debug(f"TARGET TRANSFORM IS NOT COLLECTED!!!")
        if __debug__:
            log.debug(f"full_item GPU: {self.cur_rank} Dataset GET_tar\t{get_tar_end-start}\t\tGET_img\t{get_img_end-get_tar_end}\t\tLOAD\t0\t\tAUG\t{trans_end-get_img_end}\t\tINDEX\t{index}")
        return sample, target

    # Partial image pipeline in Refurbish
    def final_item(self, index):
        if __debug__:
            start = time.perf_counter()
        path, target = self.samples[index]
        if __debug__:
            get_tar_end = time.perf_counter()
        # dataloads_func = 
        # self.memory_cost(path)
        if __debug__: 
            get_img_end = time.perf_counter()
        if self.transform is not None:
            sample = self.final_aug_cost(index)
            if __debug__:
                trans_end = time.perf_counter()
        if self.target_transform is not None:
            target = self.target_transform(target)
            if __debug__:
                log.debug(f"TARGET TRANSFORM IS NOT COLLECTED!!!")
        if __debug__:
            log.debug(f"final_item GPU: {self.cur_rank} Dataset GET_tar\t{get_tar_end-start}\t\tGET_img\t{get_img_end-get_tar_end}\t\tLOAD\t0\t\tAUG\t{trans_end-get_img_end}\t\tINDEX\t{index}")
        return sample, target
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        getitem_func = self.get_items[self.counter]
        sample, target = getitem_func(index)
        self.counter+=1
        if self.counter >= self.batch_size:
            self.counter=0
        return sample, target
