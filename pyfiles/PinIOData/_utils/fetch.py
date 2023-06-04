r""""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch
data from an iterable-style or map-style dataset. This logic is shared in both
single- and multi-processing data loading.
"""
from concurrent.futures import ThreadPoolExecutor
from . import worker
from functools import partial
from itertools import cycle
import threading
import psutil

import logging  
import time
logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)
    
class _BaseDatasetFetcher(object):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index):
        raise NotImplementedError()

    def ioLoad(self, possibly_batched_index):
        raise NotImplementedError()
    
class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_IterableDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            data = []
            for _ in possibly_batched_index:
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    break
            if len(data) == 0 or (self.drop_last and len(data) < len(possibly_batched_index)):
                raise StopIteration
        else:
            data = next(self.dataset_iter)
        
        if __debug__:
            start = time.perf_counter()
        collate_data = self.collate_fn(data)
        if __debug__:
            end = time.perf_counter()
            log.info(f"Collate END at_time {end-start}")
        return collate_data
    
    def ioLoad(self, possibly_batched_index):
            if self.auto_collation:
                for idx in possibly_batched_index:
                    possible_img, _ = self.dataset.samples[idx]
                    with open(possible_img, 'rb') as f:
                        f.read()
            else:
                possible_img, _ = self.dataset.samples[possibly_batched_index]
                with open(possible_img, 'rb') as f:
                    f.read()
            return
        
class _MapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_MapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            if __debug__:
                collist_start = time.perf_counter()    
            data = [self.dataset[idx] for idx in possibly_batched_index]
            if __debug__:
                collist_end = time.perf_counter()
                log.info(f"Createing list for Collate END at_time {collist_end-collist_start}")
        else:
            if __debug__:
                collist_start = time.perf_counter()    
            data = self.dataset[possibly_batched_index]
            if __debug__:
                collist_end = time.perf_counter()
                log.info(f"Createing list for Collate END at_time {collist_end-collist_start}")
        if __debug__:
            start = time.perf_counter()
        collate_data = self.collate_fn(data)
        if __debug__:
            end = time.perf_counter()
            log.info(f"Collate END at_time {end-start}")
        return collate_data

    def ioLoad(self, possibly_batched_index):
        if self.auto_collation:
            for idx in possibly_batched_index:
                possible_img, _ = self.dataset.samples[idx]
                with open(possible_img, 'rb') as f:
                    f.read()
        else:
            possible_img, _ = self.dataset.samples[possibly_batched_index]
            with open(possible_img, 'rb') as f:
                f.read()
        return
    
class _MultiThreadMapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last, max_threads=4):
        super(_MultiThreadMapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
        self.max_threads=max_threads
        self.thread_ids = []
        for i in range(max_threads):
            self.thread_ids.append(i)

    def _refer(self, index):
        
        threading.get_ident()
        
        p = psutil.Process()
        p.cpu_affinity([cpu_id])

        return self.dataset[index]
    
    # MultiThread
    def fetch(self, possibly_batched_index):
        with ThreadPoolExecutor(max_workers=self.max_threads) as thread_pools:
            # print("batch index: ", possibly_batched_index, flush=True)
            if __debug__:
                collist_start = time.perf_counter()    
            data = []
            for single_point in thread_pools.map(self._refer, cycle(self.thread_ids, possibly_batched_index)):
                data.append(single_point)
            if __debug__:
                collist_end = time.perf_counter()
                log.info(f"Createing list for Collate END at_time {collist_end-collist_start}")
            collate_data = self.collate_fn(data)
            if __debug__:
                end = time.perf_counter()
                log.info(f"Collate END at_time {end-collist_end}")
            return collate_data
        
    def ioLoad(self, possibly_batched_index):
            if self.auto_collation:
                for idx in possibly_batched_index:
                    possible_img, _ = self.dataset.samples[idx]
                    with open(possible_img, 'rb') as f:
                        f.read()
            else:
                possible_img, _ = self.dataset.samples[possibly_batched_index]
                with open(possible_img, 'rb') as f:
                    f.read()
            return