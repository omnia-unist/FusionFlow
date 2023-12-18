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


class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_IterableDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)

    def fetch(self, possibly_batched_index):
        # print("I WAS HERE", flush=True)
        # if self.auto_collation:
        #     # data = []
        #     _input, _target = [], []
        #     for _ in possibly_batched_index:
        #         try:
        #             data = next(self.dataset_iter)
        #             _input.append(data[0])
        #             _target.append(data[1])
        #             # data.append(next(self.dataset_iter))
        #         except StopIteration:
        #             break
        #     if len(data) == 0 or (self.drop_last and len(data) < len(possibly_batched_index)):
        #         raise StopIteration
        #     return _input, _target
        # else:
        #     data = next(self.dataset_iter)
        #     return data
        # if __debug__:
        #     start = time.perf_counter()
        # collate_data = self.collate_fn(data)
        # if __debug__:
        #     end = time.perf_counter()
        #     log.info(f"Collate END at_time {end-start}")
        # return collate_data
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

class _ProgressInfoMapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last, worker_progress_info):
        super(_ProgressInfoMapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
        self.worker_progress_info=worker_progress_info

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            _input, _target = [], []
            for i, idx in enumerate(possibly_batched_index):
                data = self.dataset[idx]
                _input.append(data[0])
                _target.append(data[1])
            return _input, _target
        else:
            data = self.dataset[possibly_batched_index]
            return data
        # Old Code
        
        # if self.auto_collation:
        #     data = []
        #     for i, idx in enumerate(possibly_batched_index):
        #         data.append(self.dataset[idx])
        #         self.worker_progress_info[1] = i
        # else:
        #     data = self.dataset[possibly_batched_index]
        # collate_data = self.collate_fn(data)
        # return collate_data


class _MonolithProgressInfoMapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last, worker_progress_info):
        super(_MonolithProgressInfoMapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
        self.worker_progress_info=worker_progress_info

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            data = []
            for i, idx in enumerate(possibly_batched_index):
                data.append(self.dataset._get_transformed_item(idx))
                self.worker_progress_info[1] = i
        else:
            data = self.dataset._get_transformed_item(possibly_batched_index)
        collate_data = self.collate_fn(data)
        return collate_data