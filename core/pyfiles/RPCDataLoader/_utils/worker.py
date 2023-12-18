""""Contains definitions of the methods used by the _BaseDataLoaderIter workers.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import torch
import random
import os
from collections import namedtuple
# from torch._six import queue
import queue
from torch._utils import ExceptionWrapper
from typing import Union
from . import signal_handling, MP_STATUS_CHECK_INTERVAL, IS_WINDOWS
from .DatasetKind import _DatasetKind
from . import REMOTE_DECODE_PREP_COLLATE_PIPE, REMOTE_DECODE_PREP_PIPE, REMOTE_PREP_PIPE, COLLATE_ONLY
import time

if __debug__:
    import logging
    import time
    # import os
    # import psutil

    # def _check_usage_memory():
    #     pid = os.getpid()
    #     py  = psutil.Process(pid)
    #     memory_usage  = round(py.memory_info()[0] /2.**30, 2)
        
    #     log.debug(f"memory usage\t\t: {memory_usage}%")
        
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger(__name__)

if IS_WINDOWS:
    import ctypes
    from ctypes.wintypes import DWORD, BOOL, HANDLE

    # On Windows, the parent ID of the worker process remains unchanged when the manager process
    # is gone, and the only way to check it through OS is to let the worker have a process handle
    # of the manager and ask if the process status has changed.
    class ManagerWatchdog(object):
        def __init__(self):
            self.manager_pid = os.getppid()

            # mypy cannot detect this code is windows only
            self.kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)  # type: ignore
            self.kernel32.OpenProcess.argtypes = (DWORD, BOOL, DWORD)
            self.kernel32.OpenProcess.restype = HANDLE
            self.kernel32.WaitForSingleObject.argtypes = (HANDLE, DWORD)
            self.kernel32.WaitForSingleObject.restype = DWORD

            # Value obtained from https://msdn.microsoft.com/en-us/library/ms684880.aspx
            SYNCHRONIZE = 0x00100000
            self.manager_handle = self.kernel32.OpenProcess(SYNCHRONIZE, 0, self.manager_pid)

            if not self.manager_handle:
                raise ctypes.WinError(ctypes.get_last_error())  # type: ignore

            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                # Value obtained from https://msdn.microsoft.com/en-us/library/windows/desktop/ms687032.aspx
                self.manager_dead = self.kernel32.WaitForSingleObject(self.manager_handle, 0) == 0
            return not self.manager_dead
else:
    class ManagerWatchdog(object):  # type: ignore[no-redef]
        def __init__(self):
            self.manager_pid = os.getppid()
            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                self.manager_dead = os.getppid() != self.manager_pid
            return not self.manager_dead

_worker_info = None


class WorkerInfo(object):
    __initialized = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__keys = tuple(kwargs.keys())
        self.__initialized = True

    def __setattr__(self, key, val):
        if self.__initialized:
            raise RuntimeError("Cannot assign attributes to {} objects".format(self.__class__.__name__))
        return super(WorkerInfo, self).__setattr__(key, val)

    def __repr__(self):
        items = []
        for k in self.__keys:
            items.append('{}={}'.format(k, getattr(self, k)))
        return '{}({})'.format(self.__class__.__name__, ', '.join(items))


def get_worker_info():
    r"""Returns the information about the current
    :class:`~torch.utils.data.DataLoader` iterator worker process.

    When called in a worker, this returns an object guaranteed to have the
    following attributes:

    * :attr:`id`: the current worker id.
    * :attr:`num_workers`: the total number of workers.
    * :attr:`seed`: the random seed set for the current worker. This value is
      determined by main process RNG and the worker id. See
      :class:`~torch.utils.data.DataLoader`'s documentation for more details.
    * :attr:`dataset`: the copy of the dataset object in **this** process. Note
      that this will be a different object in a different process than the one
      in the main process.

    When called in the main process, this returns ``None``.

    .. note::
       When used in a :attr:`worker_init_fn` passed over to
       :class:`~torch.utils.data.DataLoader`, this method can be useful to
       set up each worker process differently, for instance, using ``worker_id``
       to configure the ``dataset`` object to only read a specific fraction of a
       sharded dataset, or use ``seed`` to seed other libraries used in dataset
       code (e.g., NumPy).
    """
    return _worker_info


def get_raw_images(samples, indexes):
    raw_data = []
    for i in indexes:
        jpeg_file, label = samples[i]
        f = open(jpeg_file, 'rb')
        raw_data.append((f.read(), label))
    return raw_data


def get_decoded_images(dataset, indexes):
    data = []
    for i in indexes:
        path, label = dataset.samples[i]
        decoded_image = dataset.loader(path)
        data.append((decoded_image, label))
    return data


r"""Dummy class used to signal the end of an IterableDataset"""
_IterableDatasetStopIteration = namedtuple('_IterableDatasetStopIteration', ['worker_id'])

r"""Dummy class used to resume the fetching when worker reuse is enabled"""
_ResumeIteration = namedtuple('_ResumeIteration', [])

def _worker_loop(dataset_kind, dataset, index_queue, data_queue, done_event,
                 auto_collation, collate_fn, drop_last, seed, init_fn, worker_id,
                 num_workers, persistent_workers, rank):
    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    import psutil
    import sys
    sys.path.append("..")

    import globalManager as gc
    cpu_id = worker_id + gc.CPU_IDS_PER_GPU * rank
    p = psutil.Process()
    p.cpu_affinity([cpu_id])

    
    if __debug__:
        log.debug(f"GPU: {rank} worker_id: {worker_id}, CPU affinity: {p.cpu_affinity()}")
    try:
        # Initialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
        # module's handlers are executed after Python returns from C low-level
        # handlers, likely when the same fatal signal had already happened
        # again.
        # https://docs.python.org/3/library/signal.html#execution-of-python-signal-handlers
        signal_handling._set_worker_signal_handlers()

        torch.set_num_threads(1)
        random.seed(seed)
        torch.manual_seed(seed)

        global _worker_info
        _worker_info = WorkerInfo(id=worker_id, num_workers=num_workers,
                                  seed=seed, dataset=dataset)



        init_exception = None

        try:
            if init_fn is not None:
                init_fn(worker_id)

            fetcher = _DatasetKind.create_fetcher(dataset_kind, dataset, auto_collation, collate_fn, drop_last)
        except Exception:
            init_exception = ExceptionWrapper(
                where="in DataLoader worker process {}".format(worker_id))

        # When using Iterable mode, some worker can exit earlier than others due
        # to the IterableDataset behaving differently for different workers.
        # When such things happen, an `_IterableDatasetStopIteration` object is
        # sent over to the main process with the ID of this worker, so that the
        # main process won't send more tasks to this worker, and will send
        # `None` to this worker to properly exit it.
        #
        # Note that we cannot set `done_event` from a worker as it is shared
        # among all processes. Instead, we set the `iteration_end` flag to
        # signify that the iterator is exhausted. When either `done_event` or
        # `iteration_end` is set, we skip all processing step and just wait for
        # `None`.
        iteration_end = False

        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            # Add event driven wait
                
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
                # if __debug__:
                #     log.debug(f"GPU: {rank} worker_id: {worker_id}, GET {r} index_queue:\n{index_queue}")
            except queue.Empty:
                continue
            if isinstance(r, _ResumeIteration):
                # Acknowledge the main process
                data_queue.put((r, None))
                
                # # Old
                # data_queue.put(r)
                
                iteration_end = False
                # Recreate the fetcher for worker-reuse policy
                fetcher = _DatasetKind.create_fetcher(
                    dataset_kind, dataset, auto_collation, collate_fn, drop_last)
                continue
            elif r is None:
                # Received the final signal
                assert done_event.is_set() or iteration_end
                break
            elif done_event.is_set() or iteration_end:
                # `done_event` is set. But I haven't received the final signal
                # (None) yet. I will keep continuing until get it, and skip the
                # processing steps.
                continue
            idx, is_remote, index = r
            data: Union[_IterableDatasetStopIteration, ExceptionWrapper]
            if init_exception is not None:
                data = init_exception
                init_exception = None
            else:
                try:
                    if __debug__:
                        # log.debug(f"GPU: {rank} worker_id: {worker_id}, START FETCH data idx{idx}")
                        start_time = time.perf_counter()
                    if is_remote is None:  # This is local task
                        # only for duplicate dataset
                        data = fetcher.fetch(index) 
                    elif is_remote == COLLATE_ONLY:  # This is collation task
                        start_time = time.perf_counter()
                        data = collate_fn(index)
                        end_time = time.perf_counter()
                        print("Collation ", end_time - start_time, flush=True)
                    else:  # This is remote task
                        if is_remote == REMOTE_DECODE_PREP_COLLATE_PIPE or is_remote == REMOTE_DECODE_PREP_PIPE:
                            # Get decode file 
                            data = get_raw_images(dataset.samples, index)
                        elif is_remote == REMOTE_PREP_PIPE:
                            # Do decoding here
                            data = get_decoded_images(dataset, index)
                        else:
                            raise NotImplementedError(f"Operation {is_remote} does not exist")
                    if __debug__:
                        end_time = time.perf_counter()
                        log.debug(f"GPU: {rank} worker_id: {worker_id}, END FETCH at_time {end_time-start_time} data idx{idx}")
                        
                        # _check_usage_memory()
                except Exception as e:
                    print("EXCEPTION", e, flush=True)
                    if isinstance(e, StopIteration) and dataset_kind == _DatasetKind.Iterable:
                        data = _IterableDatasetStopIteration(worker_id)
                        # Set `iteration_end`
                        #   (1) to save future `next(...)` calls, and
                        #   (2) to avoid sending multiple `_IterableDatasetStopIteration`s.
                        iteration_end = True
                    else:
                        # It is important that we don't store exc_info in a variable.
                        # `ExceptionWrapper` does the correct thing.
                        # See NOTE [ Python Traceback Reference Cycle Problem ]
                        data = ExceptionWrapper(
                            where="in DataLoader worker process {}".format(worker_id))
            start_time = time.perf_counter()
            data_queue.put((idx, data))
            end_time = time.perf_counter()
            if is_remote == REMOTE_DECODE_PREP_COLLATE_PIPE or is_remote == REMOTE_DECODE_PREP_PIPE:
                print("Time to put raw data:", end_time - start_time, flush=True)
            if __debug__:
                log.debug(f"GPU: {rank} worker_id: {worker_id}, PUT ({idx}, data object at {hex(id(data))}) data_queue:\n{data_queue}")
            # _check_usage_memory()
            del data, idx, index, r  # save memory
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass
    if done_event.is_set():
        data_queue.cancel_join_thread()
        # if __debug__:
        #     log.debug(f"GPU: {rank} worker_id: {worker_id}, cancel_join_thread data_queue:\n{data_queue}")
        data_queue.close()
        # if __debug__:
        #     log.debug(f"GPU: {rank} worker_id: {worker_id}, close data_queue:\n{data_queue}")