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
import os, sys
sys.path.append(os.path.dirname(os.path.abspath("../")))
import time

if __debug__:
    import logging  
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger(__name__)
    
class _BaseDatasetFetcher(object):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last, rank, worker_id):
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.worker_id = worker_id
        self.rank = rank

        
    def fetch(self, possibly_batched_index, idx):
        raise NotImplementedError()


class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last, rank, worker_id):
        super(_IterableDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last, rank, worker_id)
        self.dataset_iter = iter(dataset)

    def fetch(self, possibly_batched_index, idx):
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
            log.debug(f"Collate END at_time {end-start}")
        return collate_data


class _MapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last, rank, worker_id):
        super(_MapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last, rank, worker_id)

    def fetch(self, possibly_batched_index, index):
        if self.auto_collation:
            if __debug__:
                collist_start = time.perf_counter()    
            data = []
            req = None
            heartbeat = time.perf_counter()
            for idx in possibly_batched_index:
                # Send info to server.
                if (time.perf_counter() - heartbeat) > 0.100:
                    if req is not None:
                        print("Waiting for req done",flush=True)
                        req.wait()
                    heartbeat = time.perf_counter()
                    print("Send data with gcHelper",flush=True)
                    req = self.gcHelper.Send({self.rank:{self.worker_id:(index,idx)}})
                data.append(self.dataset[idx])
                
            if __debug__:
                collist_end = time.perf_counter()
                log.debug(f"Createing list for Collate END at_time {collist_end-collist_start}")
        else:
            if __debug__:
                collist_start = time.perf_counter()    
            data = self.dataset[possibly_batched_index]
            if __debug__:
                collist_end = time.perf_counter()
                log.debug(f"Createing list for Collate END at_time {collist_end-collist_start}")
        if __debug__:
            start = time.perf_counter()
        collate_data = self.collate_fn(data)
        if __debug__:
            end = time.perf_counter()
            log.debug(f"Collate END at_time {end-start}")
        return collate_data
