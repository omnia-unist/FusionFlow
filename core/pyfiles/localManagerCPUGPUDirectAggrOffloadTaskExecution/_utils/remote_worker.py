""""Contains definitions of the methods used by the _BaseDataLoaderIter workers.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

from . import operation_offloading as CO

import multiprocessing
import torch
import random
import os
from collections import namedtuple
# from torch._six import queue
# PyTorch 2.0 Fix
import queue
from torch._utils import ExceptionWrapper
from typing import Union
from . import signal_handling, MP_STATUS_CHECK_INTERVAL, IS_WINDOWS
from .DatasetKind import _DatasetKind
from posix_ipc import O_CREAT, SharedMemory
import mmap
import numpy as np
import psutil
import PIL
import time

from time import perf_counter as timer_func

if __debug__:
    import logging
    import sys
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


class FunctionWrapper(object):
    def __init__(self, func, v):
        self.func = func
        self.v = v

    def __call__(self, input_):
        return self.func(input_, v)


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


def _load_np_arr(type, rank, worker_id, frame, dtype=np.int32):
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

r"""Dummy class used to signal the end of an IterableDataset"""
_IterableDatasetStopIteration = namedtuple('_IterableDatasetStopIteration', ['worker_id'])

r"""Dummy class used to resume the fetching when worker reuse is enabled"""
_ResumeIteration = namedtuple('_ResumeIteration', [])

def _worker_loop(dataset_kind, dataset, index_queue, data_queue, done_event,
                 auto_collation, collate_fn, drop_last, seed, init_fn, worker_id,
                 num_workers, persistent_workers, control_event, status, cur_cpus, 
                 dataloader_processes, rank, gpu_queue):
    try:
        # Initialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
        # module's handlers are executed after Python returns from C low-level
        # handlers, likely when the same fatal signal had already happened
        # again.
        # https://docs.python.org/3/library/signal.html#execution-of-python-signal-handlers
        signal_handling._set_worker_signal_handlers()

        input_transform = dataset.transform
        target_transform = dataset.target_transform
        iteration_end = False
        watchdog = ManagerWatchdog()
        status.value = 1
        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                status.value = 0
                control_event.wait()
                status.value = 1
                continue
            if isinstance(r, _ResumeIteration):
                # Acknowledge the main process
                iteration_end = False
                # Recreate the fetcher for worker-reuse policy
                continue
            elif r is None:
                # Received the final signal
                assert done_event.is_set() or iteration_end
                break
            elif done_event.is_set() or iteration_end:
                continue
            if __debug__:
                log.debug(f"worker: value {r}")
            op_type = r[0]
            # Decision 
            if CO.is_start(op_type):
                op_type = CO.next_op(op_type)
                decision = CO.conditional_offloading_cpu(op_type)
            idx, index = r[1]
            try:    
                # Start Putting them here!
                data_queue.put((idx, [_input, _target]))
            except Exception as e:
                if isinstance(e, StopIteration) and dataset_kind == _DatasetKind.Iterable:
                    data = _IterableDatasetStopIteration(worker_id)
                    if __debug__:
                        log.debug(f"GPU: {rank} worker_id: {worker_id}, Stop iteration due to {e}")
                    # print(f"GPU: {rank} worker_id: {worker_id}, Stop iteration due to {e}")
                    # Set `iteration_end`
                    #   (1) to save future `next(...)` calls, and
                    #   (2) to avoid sending multiple `_IterableDatasetStopIteration`s.
                    iteration_end = True
                else:
                    if __debug__:
                        log.debug(f"GPU: {rank} worker_id: {worker_id}, Stop worker process due to {e}")
                    # print(f"GPU: {rank} worker_id: {worker_id}, Stop worker process due to {e}")
                    # It is important that we don't store exc_info in a variable.
                    # `ExceptionWrapper` does the correct thing.
                    # See NOTE [ Python Traceback Reference Cycle Problem ]
                    data = ExceptionWrapper(
                        where="in DataLoader worker process {}".format(worker_id))
                data_queue.put((idx, data))
                del data, idx, r  # save memory
            if __debug__:
                log.debug(f"GPU: {rank} worker_id: {worker_id}, PUT {idx}")
            # _check_usage_memory()
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass
    if done_event.is_set():
        shm_buf.close()
        data_queue.cancel_join_thread()
        # if __debug__:
        #     log.debug(f"GPU: {rank} worker_id: {worker_id}, cancel_join_thread data_queue:\n{data_queue}")
        data_queue.close()
        # if __debug__:
        #     log.debug(f"GPU: {rank} worker_id: {worker_id}, close data_queue:\n{data_queue}")

