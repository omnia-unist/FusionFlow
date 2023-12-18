"""Definition of the DataLoader and associated iterators that subclass _BaseDataLoaderIter

To support these two classes, in `./_utils` we define many utility methods and
functions to be run in multiprocessing. E.g., the data loading worker loop is
in `./_utils/worker.py`.
"""

from logging import debug
import threading
import itertools
import warnings
from typing import Any, Callable, TypeVar, Generic, Sequence, List, Optional
from collections import deque

import multiprocessing as python_multiprocessing
import torch
import torch.multiprocessing as multiprocessing
import torch.distributed as dist
torch.multiprocessing.set_sharing_strategy('file_system')
from torch._utils import ExceptionWrapper
# from torch._six import queue, string_classes
# PyTorch 2.0 Fix
import queue
string_classes = (str, bytes)
import numa
from ctypes import c_bool
from posix_ipc import O_CREAT, SharedMemory, unlink_shared_memory
import mmap
import numpy as np

from . import IterableDataset, Sampler, SequentialSampler, RandomSampler, BatchSampler, FinegrainedBatchSampler, Dataset
from . import _utils

import sys
sys.path.append("..")

import globalManager as gc
from copy import deepcopy

torch.set_num_threads(1)
import time
import psutil

if __debug__:
    import logging
    # import os
    # import psutil

    # def _check_usage_memory():
    #     pid = os.getpid()
    #     py  = psutil.Process(pid)
    #     memory_usage  = round(py.memory_info()[0] /2.**30, 2)
        
    #     log.debug(f"memory usage\t\t: {memory_usage}%")
        
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger(__name__)
    

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
_worker_init_fn_t = Callable[[int], None]

# Ideally we would parameterize `DataLoader` by the return type of `collate_fn`, but there is currently no way to have that
# type parameter set to a default value if the user doesn't pass in a custom 'collate_fn'.
# See https://github.com/python/mypy/issues/3737.
_collate_fn_t = Callable[[List[T]], Any]


# This function used to be defined in this file. However, it was moved to
# _utils/collate.py. Although it is rather hard to access this from user land
# (one has to explicitly directly `import torch.utils.data.dataloader`), there
# probably is user code out there using it. This aliasing maintains BC in this
# aspect.
default_collate: _collate_fn_t = _utils.collate.default_collate

get_worker_info = _utils.worker.get_worker_info

from ._utils.DatasetKind import _DatasetKind

class _InfiniteConstantSampler(Sampler):
    r"""Analogous to ``itertools.repeat(None, None)``.
    Used as sampler for :class:`~torch.utils.data.IterableDataset`.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self):
        super(_InfiniteConstantSampler, self).__init__(None)

    def __iter__(self):
        while True:
            yield None

class DataLoader(Generic[T_co]):
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.

    The :class:`~torch.utils.data.DataLoader` supports both map-style and
    iterable-style datasets with single- or multi-process loading, customizing
    loading order and optional automatic batching (collation) and memory pinning.

    See :py:mod:`torch.utils.data` documentation page for more details.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        sampler (Sampler or Iterable, optional): defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented. If specified, :attr:`shuffle` must not be specified.
        batch_sampler (Sampler or Iterable, optional): like :attr:`sampler`, but
            returns a batch of indices at a time. Mutually exclusive with
            :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
            and :attr:`drop_last`.
        num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
            into CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        worker_init_fn (callable, optional): If not ``None``, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: ``None``)
        prefetch_factor (int, optional, keyword-only arg): Number of sample loaded
            in advance by each worker. ``2`` means there will be a total of
            2 * num_workers samples prefetched across all workers. (default: ``2``)
        persistent_workers (bool, optional): If ``True``, the data loader will not shutdown
            the worker processes after a dataset has been consumed once. This allows to 
            maintain the workers `Dataset` instances alive. (default: ``False``)


    .. warning:: If the ``spawn`` start method is used, :attr:`worker_init_fn`
                 cannot be an unpicklable object, e.g., a lambda function. See
                 :ref:`multiprocessing-best-practices` on more details related
                 to multiprocessing in PyTorch.

    .. warning:: ``len(dataloader)`` heuristic is based on the length of the sampler used.
                 When :attr:`dataset` is an :class:`~torch.utils.data.IterableDataset`,
                 it instead returns an estimate based on ``len(dataset) / batch_size``, with proper
                 rounding depending on :attr:`drop_last`, regardless of multi-process loading
                 configurations. This represents the best guess PyTorch can make because PyTorch
                 trusts user :attr:`dataset` code in correctly handling multi-process
                 loading to avoid duplicate data.

                 However, if sharding results in multiple workers having incomplete last batches,
                 this estimate can still be inaccurate, because (1) an otherwise complete batch can
                 be broken into multiple ones and (2) more than one batch worth of samples can be
                 dropped when :attr:`drop_last` is set. Unfortunately, PyTorch can not detect such
                 cases in general.

                 See `Dataset Types`_ for more details on these two types of datasets and how
                 :class:`~torch.utils.data.IterableDataset` interacts with
                 `Multi-process data loading`_.
    """
    dataset: Dataset[T_co]
    batch_size: Optional[int]
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float
    sampler: Sampler
    prefetch_factor: int
    _iterator : Optional['_BaseDataLoaderIter']
    __initialized = False

    def __init__(self, dataset: Dataset[T_co], remote_dataset: Dataset[T_co] = None, batch_size: Optional[int] = 1,
                 shuffle: bool = False, sampler: Optional[Sampler[int]] = None,
                 batch_sampler: Optional[Sampler[Sequence[int]]] = None,
                 num_workers: int = 0, collate_fn: _collate_fn_t = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: _worker_init_fn_t = None,
                 multiprocessing_context=None, generator=None,
                 *, prefetch_factor: int = 1,
                 persistent_workers: bool = False,
                 rank: Optional[int] = 0,
                 worker_batch_size: Optional[int] = 4,
                 aggr_shape_hint: Optional[tuple] = (None, 3, 224, 224),
                 max_workers: Optional[int] = None,
                 max_gpu: Optional[int] = None,
                 aug_type: Optional[str] = "default",
                 single_sample: Optional[bool] = False,
                 control_queue = None,
                 ):
        torch._C._log_api_usage_once("python.data_loader")  # type: ignore
        
        if num_workers < 0:
            raise ValueError('num_workers option should be non-negative; '
                             'use num_workers=0 to disable multiprocessing.')

        if timeout < 0:
            raise ValueError('timeout option should be non-negative')

        # if num_workers == 0 and prefetch_factor != 1:
        #     raise ValueError('prefetch_factor option could only be specified in multiprocessing.'
        #                      'let num_workers > 0 to enable multiprocessing.')
        assert prefetch_factor > 0

        if persistent_workers and num_workers == 0:
            raise ValueError('persistent_workers option needs num_workers > 0')

        self.prefetch_factor = prefetch_factor
        self.dataset = dataset
        self.remote_dataset = remote_dataset
        self.num_workers = num_workers
        
        self.pin_memory = pin_memory
        self.single_sample = single_sample
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.rank = rank
        self.control_queue = control_queue

        if max_gpu == None:
            self.max_gpu = gc.NUM_GPU
        else:
            self.max_gpu = max_gpu

        self.node_max = numa.get_max_node()
        self.gpu_per_node = self.max_gpu // (self.node_max+1)
        if gc.NUM_GPU == 1:
            self.gpu_per_node = 1

        self.cur_node = self.rank // self.gpu_per_node
        self.cur_cpus = list(numa.node_to_cpus(self.cur_node))
        self.cpu_per_gpu =psutil.cpu_count(logical = False) // self.max_gpu
        self.dataloader_processes = [ self.cpu_per_gpu * gpu_id + self.num_workers for gpu_id in range(self.gpu_per_node * self.cur_node, self.gpu_per_node * (self.cur_node+1)) ]
        self.dataloader_processes = [4, 9, 14, 20, 25, 30]
        print(f"In data loader {self.dataloader_processes}")
        self.aggr_shape_hint = aggr_shape_hint

        if max_workers == None:
            self.max_workers = psutil.cpu_count(logical = False)
        else:
            self.max_workers = max_workers
        self.gc_helper = gc.globalManagerHelper(rank = self.rank, max_workers=max_workers)
        if __debug__:
            log.debug(f"cur_node: {self.cur_node}, self.cur_cpus{self.cur_cpus}, gpus:{self.gpu_per_node}, dataloader process {self.dataloader_processes}")
        # print(f"cur_node: {self.cur_node}, self.cur_cpus{self.cur_cpus}, gpus:{self.gpu_per_node}, dataloader process {self.dataloader_processes}")
        
        self.aug_type = aug_type
            
        self.cpu_id_start = self.cpu_per_gpu * self.rank# - self.cur_cpus[0]
        
        if __debug__:
            log.debug(f"GPU: {self.rank} Local Manager, cpu_id_start: {self.cpu_id_start}, cpu_per_gpu: {self.cpu_per_gpu}, max_workers: {self.max_workers}")
        # print(f"GPU: {self.rank} Local Manager, cpu_id_start: {self.cpu_id_start}, cpu_per_gpu: {self.cpu_per_gpu}, max_workers: {self.max_workers}")
        # Arg-check dataset related before checking samplers because we want to
        # tell users that iterable-style datasets are incompatible with custom
        # samplers first, so that they don't learn that this combo doesn't work
        # after spending time fixing the custom sampler errors.
        if isinstance(dataset, IterableDataset):
            self._dataset_kind = _DatasetKind.Iterable
            # NOTE [ Custom Samplers and IterableDataset ]
            #
            # `IterableDataset` does not support custom `batch_sampler` or
            # `sampler` since the key is irrelevant (unless we support
            # generator-style dataset one day...).
            #
            # For `sampler`, we always create a dummy sampler. This is an
            # infinite sampler even when the dataset may have an implemented
            # finite `__len__` because in multi-process data loading, naive
            # settings will return duplicated data (which may be desired), and
            # thus using a sampler with length matching that of dataset will
            # cause data lost (you may have duplicates of the first couple
            # batches, but never see anything afterwards). Therefore,
            # `Iterabledataset` always uses an infinite sampler, an instance of
            # `_InfiniteConstantSampler` defined above.
            #
            # A custom `batch_sampler` essentially only controls the batch size.
            # However, it is unclear how useful it would be since an iterable-style
            # dataset can handle that within itself. Moreover, it is pointless
            # in multi-process data loading as the assignment order of batches
            # to workers is an implementation detail so users can not control
            # how to batchify each worker's iterable. Thus, we disable this
            # option. If this turns out to be useful in future, we can re-enable
            # this, and support custom samplers that specify the assignments to
            # specific workers.
            if shuffle is not False:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "shuffle option, but got shuffle={}".format(shuffle))
            elif sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "sampler option, but got sampler={}".format(sampler))
            elif batch_sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "batch_sampler option, but got batch_sampler={}".format(batch_sampler))
        else:
            # self._dataset_kind = _DatasetKind.Map
            self._dataset_kind = _DatasetKind.ProgressInfo

        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')

        if batch_sampler is not None:
            # auto_collation with custom batch_sampler
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            batch_size = None
            drop_last = False
        elif batch_size is None:
            # no auto_collation
            if drop_last:
                raise ValueError('batch_size=None option disables auto-batching '
                                 'and is mutually exclusive with drop_last')

        if sampler is None:  # give default samplers
            if self._dataset_kind == _DatasetKind.Iterable:
                # See NOTE [ Custom Samplers and IterableDataset ]
                sampler = _InfiniteConstantSampler()
            else:  # map-style
                if shuffle:
                    # Cannot statically verify that dataset is Sized
                    # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
                    sampler = RandomSampler(dataset, generator=generator)  # type: ignore
                else:
                    sampler = SequentialSampler(dataset)


        if batch_size is not None and batch_sampler is None:
            # auto_collation without custom batch_sampler
            # batch_sampler = BatchSampler(sampler, batch_size, drop_last)
            batch_sampler = FinegrainedBatchSampler(
                batch_size = worker_batch_size, sampler = sampler, actual_batch_size = batch_size, drop_last = drop_last)


        self.worker_batch_size = worker_batch_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.generator = generator

        if collate_fn is None:
            if self._auto_collation:
                collate_fn = _utils.collate.default_collate
            else:
                collate_fn = _utils.collate.default_convert

        self.collate_fn = collate_fn
        self.persistent_workers = persistent_workers

        self.__initialized = True
        # See NOTE [ IterableDataset and __len__ ]
        self._IterableDataset_len_called = None

        self._iterator = None

    def manager_call(self):
        return self.gc_helper

    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)

    @property
    def multiprocessing_context(self):
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context):
        if multiprocessing_context is not None:
            if self.num_workers > 0:
                if not multiprocessing._supports_context:
                    raise ValueError('multiprocessing_context relies on Python >= 3.4, with '
                                     'support for different start methods')

                if isinstance(multiprocessing_context, string_classes):
                    valid_start_methods = multiprocessing.get_all_start_methods()
                    if multiprocessing_context not in valid_start_methods:
                        raise ValueError(
                            ('multiprocessing_context option '
                             'should specify a valid start method in {!r}, but got '
                             'multiprocessing_context={!r}').format(valid_start_methods, multiprocessing_context))
                    # error: Argument 1 to "get_context" has incompatible type "Union[str, bytes]"; expected "str"  [arg-type]
                    multiprocessing_context = multiprocessing.get_context(multiprocessing_context)  # type: ignore

                if not isinstance(multiprocessing_context, python_multiprocessing.context.BaseContext):
                    raise TypeError(('multiprocessing_context option should be a valid context '
                                     'object or a string specifying the start method, but got '
                                     'multiprocessing_context={}').format(multiprocessing_context))
            else:
                raise ValueError(('multiprocessing_context can only be used with '
                                  'multi-process loading (num_workers > 0), but got '
                                  'num_workers={}').format(self.num_workers))

        self.__multiprocessing_context = multiprocessing_context

    def __setattr__(self, attr, val):
        if self.__initialized and attr in (
                'batch_size', 'batch_sampler', 'sampler', 'drop_last', 'dataset', 'persistent_workers'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super(DataLoader, self).__setattr__(attr, val)

    # We quote '_BaseDataLoaderIter' since it isn't defined yet and the definition can't be moved up
    # since '_BaseDataLoaderIter' references 'DataLoader'.
    def __iter__(self) -> '_BaseDataLoaderIter':
        # When using a single worker the returned iterator should be
        # created everytime to avoid reseting its state
        # However, in the case of a multiple workers iterator
        # the iterator is only created once in the lifetime of the
        # DataLoader object so that workers can be reused
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            return self._get_iterator()

    @property
    def _auto_collation(self):
        return self.batch_sampler is not None

    @property
    def _index_sampler(self):
        # The actual sampler used for generating indices for `_DatasetFetcher`
        # (see _utils/fetch.py) to read data at each time. This would be
        # `.batch_sampler` if in auto-collation mode, and `.sampler` otherwise.
        # We can't change `.sampler` and `.batch_sampler` attributes for BC
        # reasons.
        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.sampler

    def __len__(self) -> int:
        if self._dataset_kind == _DatasetKind.Iterable:
            # NOTE [ IterableDataset and __len__ ]
            #
            # For `IterableDataset`, `__len__` could be inaccurate when one naively
            # does multi-processing data loading, since the samples will be duplicated.
            # However, no real use case should be actually using that behavior, so
            # it should count as a user error. We should generally trust user
            # code to do the proper thing (e.g., configure each replica differently
            # in `__iter__`), and give us the correct `__len__` if they choose to
            # implement it (this will still throw if the dataset does not implement
            # a `__len__`).
            #
            # To provide a further warning, we track if `__len__` was called on the
            # `DataLoader`, save the returned value in `self._len_called`, and warn
            # if the iterator ends up yielding more than this number of samples.

            # Cannot statically verify that dataset is Sized
            length = self._IterableDataset_len_called = len(self.dataset)  # type: ignore
            if self.batch_size is not None:  # IterableDataset doesn't allow custom sampler or batch_sampler
                from math import ceil
                if self.drop_last:
                    length = length // self.batch_size
                else:
                    length = ceil(length / self.batch_size)
            return length
        else:
            return len(self._index_sampler)


class _BaseDataLoaderIter(object):
    def __init__(self, loader: DataLoader) -> None:
        self._gc_helper = loader.gc_helper
        self._dataset = loader.dataset
        self._remote_dataset = loader.remote_dataset
        self._dataset_kind = loader._dataset_kind
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._num_workers = loader.num_workers
        self._prefetch_factor = loader.prefetch_factor
        # NOTE: Currently pin_memory does not help to ours
        self._pin_memory = False #loader.pin_memory and torch.cuda.is_available()
        self._single_sample = loader.single_sample
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        self._base_seed = torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()
        self._persistent_workers = loader.persistent_workers
        self._num_yielded = 0
        self._rank = loader.rank
        self._batch_size = loader.batch_size
        self._microbatch_size = loader.worker_batch_size
        self._aug_type = loader.aug_type
        self._tail_cpu_id = loader.cpu_id_start
        self.cur_cpus = loader.cur_cpus
        self._control_queue = loader.control_queue

    def __iter__(self) -> '_BaseDataLoaderIter':
        return self

    def _reset(self, loader, first_iter=False):
        # if __debug__:
        #     start = time.perf_counter()
        self._sampler_iter = iter(self._index_sampler)
        if __debug__:
            # end = time.perf_counter()
            log.debug(f"GPU: {self._rank} Local Manager Create batch sample")
        # if self._rank == 0: 
        #     print(f"GPU: {self._rank} Local Manager Create batch sample")
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        
    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

    def _next_data(self):
        raise NotImplementedError

    def __next__(self) -> Any:
        if self._sampler_iter is None:
            self._reset()
        # if __debug__:
        #     log.debug(f"GPU: {self._rank} Local Manager Get data object")
        data = self._next_data()

        self._num_yielded += 1
        if self._dataset_kind == _DatasetKind.Iterable and \
                self._IterableDataset_len_called is not None and \
                self._num_yielded > self._IterableDataset_len_called:
            warn_msg = ("Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
                        "samples have been fetched. ").format(self._dataset, self._IterableDataset_len_called,
                                                              self._num_yielded)
            if self._num_workers > 0:
                warn_msg += ("For multiprocessing data-loading, this could be caused by not properly configuring the "
                             "IterableDataset replica at each worker. Please see "
                             "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples.")
            warnings.warn(warn_msg)
       
        return data

    next = __next__  # Python 2 compatibility

    def __len__(self) -> int:
        return len(self._index_sampler)

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("{} cannot be pickled", self.__class__.__name__)


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessDataLoaderIter, self).__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last)

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data)
        return data


class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    r"""Iterates once over the DataLoader's dataset, as specified by the sampler"""

    # NOTE [ Data Loader Multiprocessing Shutdown Logic ]
    #
    # Preliminary:
    #
    # Our data model looks like this (queues are indicated with curly brackets):
    #
    #                main process                              ||
    #                     |                                    ||
    #               {index_queue}                              ||
    #                     |                                    ||
    #              worker processes                            ||     DATA
    #                     |                                    ||
    #            {worker_result_queue}                         ||     FLOW
    #                     |                                    ||
    #      pin_memory_thread of main process                   ||   DIRECTION
    #                     |                                    ||
    #               {data_queue}                               ||
    #                     |                                    ||
    #                data output                               \/
    #
    # P.S. `worker_result_queue` and `pin_memory_thread` part may be omitted if
    #      `pin_memory=True`.
    #
    #
    # Terminating multiprocessing logic requires very careful design. In
    # particular, we need to make sure that
    #
    #   1. The iterator gracefully exits the workers when its last reference is
    #      gone or it is depleted.
    #
    #      In this case, the workers should be gracefully exited because the
    #      main process may still need to continue to run, and we want cleaning
    #      up code in the workers to be executed (e.g., releasing GPU memory).
    #      Naturally, we implement the shutdown logic in `__del__` of
    #      DataLoaderIterator.
    #
    #      We delay the discussion on the logic in this case until later.
    #
    #   2. The iterator exits the workers when the loader process and/or worker
    #      processes exits normally or with error.
    #
    #      We set all workers and `pin_memory_thread` to have `daemon=True`.
    #
    #      You may ask, why can't we make the workers non-daemonic, and
    #      gracefully exit using the same logic as we have in `__del__` when the
    #      iterator gets deleted (see 1 above)?
    #
    #      First of all, `__del__` is **not** guaranteed to be called when
    #      interpreter exits. Even if it is called, by the time it executes,
    #      many Python core library resources may alreay be freed, and even
    #      simple things like acquiring an internal lock of a queue may hang.
    #      Therefore, in this case, we actually need to prevent `__del__` from
    #      being executed, and rely on the automatic termination of daemonic
    #      children. Thus, we register an `atexit` hook that sets a global flag
    #      `_utils.python_exit_status`. Since `atexit` hooks are executed in the
    #      reverse order of registration, we are guaranteed that this flag is
    #      set before library resources we use are freed. (Hooks freeing those
    #      resources are registered at importing the Python core libraries at
    #      the top of this file.) So in `__del__`, we check if
    #      `_utils.python_exit_status` is set or `None` (freed), and perform
    #      no-op if so.
    #
    #      Another problem with `__del__` is also related to the library cleanup
    #      calls. When a process ends, it shuts the all its daemonic children
    #      down with a SIGTERM (instead of joining them without a timeout).
    #      Simiarly for threads, but by a different mechanism. This fact,
    #      together with a few implementation details of multiprocessing, forces
    #      us to make workers daemonic. All of our problems arise when a
    #      DataLoader is used in a subprocess, and are caused by multiprocessing
    #      code which looks more or less like this:
    #
    #          try:
    #              your_function_using_a_dataloader()
    #          finally:
    #              multiprocessing.util._exit_function()
    #
    #      The joining/termination mentioned above happens inside
    #      `_exit_function()`. Now, if `your_function_using_a_dataloader()`
    #      throws, the stack trace stored in the exception will prevent the
    #      frame which uses `DataLoaderIter` to be freed. If the frame has any
    #      reference to the `DataLoaderIter` (e.g., in a method of the iter),
    #      its  `__del__`, which starts the shutdown procedure, will not be
    #      called. That, in turn, means that workers aren't notified. Attempting
    #      to join in `_exit_function` will then result in a hang.
    #
    #      For context, `_exit_function` is also registered as an `atexit` call.
    #      So it is unclear to me (@ssnl) why this is needed in a finally block.
    #      The code dates back to 2008 and there is no comment on the original
    #      PEP 371 or patch https://bugs.python.org/issue3050 (containing both
    #      the finally block and the `atexit` registration) that explains this.
    #
    #      Another choice is to just shutdown workers with logic in 1 above
    #      whenever we see an error in `next`. This isn't ideal because
    #        a. It prevents users from using try-catch to resume data loading.
    #        b. It doesn't prevent hanging if users have references to the
    #           iterator.
    #
    #   3. All processes exit if any of them die unexpectedly by fatal signals.
    #
    #      As shown above, the workers are set as daemonic children of the main
    #      process. However, automatic cleaning-up of such child processes only
    #      happens if the parent process exits gracefully (e.g., not via fatal
    #      signals like SIGKILL). So we must ensure that each process will exit
    #      even the process that should send/receive data to/from it were
    #      killed, i.e.,
    #
    #        a. A process won't hang when getting from a queue.
    #
    #           Even with carefully designed data dependencies (i.e., a `put()`
    #           always corresponding to a `get()`), hanging on `get()` can still
    #           happen when data in queue is corrupted (e.g., due to
    #           `cancel_join_thread` or unexpected exit).
    #
    #           For child exit, we set a timeout whenever we try to get data
    #           from `data_queue`, and check the workers' status on each timeout
    #           and error.
    #           See `_DataLoaderiter._get_batch()` and
    #           `_DataLoaderiter._try_get_data()` for details.
    #
    #           Additionally, for child exit on non-Windows platforms, we also
    #           register a SIGCHLD handler (which is supported on Windows) on
    #           the main process, which checks if any of the workers fail in the
    #           (Python) handler. This is more efficient and faster in detecting
    #           worker failures, compared to only using the above mechanism.
    #           See `DataLoader.cpp` and `_utils/signal_handling.py` for details.
    #
    #           For `.get()` calls where the sender(s) is not the workers, we
    #           guard them with timeouts, and check the status of the sender
    #           when timeout happens:
    #             + in the workers, the `_utils.worker.ManagerWatchdog` class
    #               checks the status of the main process.
    #             + if `pin_memory=True`, when getting from `pin_memory_thread`,
    #               check `pin_memory_thread` status periodically until `.get()`
    #               returns or see that `pin_memory_thread` died.
    #
    #        b. A process won't hang when putting into a queue;
    #
    #           We use `mp.Queue` which has a separate background thread to put
    #           objects from an unbounded buffer array. The background thread is
    #           daemonic and usually automatically joined when the process
    #           exits.
    #
    #           However, in case that the receiver has ended abruptly while
    #           reading from the pipe, the join will hang forever. Therefore,
    #           for both `worker_result_queue` (worker -> main process/pin_memory_thread)
    #           and each `index_queue` (main process -> worker), we use
    #           `q.cancel_join_thread()` in sender process before any `q.put` to
    #           prevent this automatic join.
    #
    #           Moreover, having all queues called `cancel_join_thread` makes
    #           implementing graceful shutdown logic in `__del__` much easier.
    #           It won't need to get from any queue, which would also need to be
    #           guarded by periodic status checks.
    #
    #           Nonetheless, `cancel_join_thread` must only be called when the
    #           queue is **not** going to be read from or write into by another
    #           process, because it may hold onto a lock or leave corrupted data
    #           in the queue, leading other readers/writers to hang.
    #
    #           `pin_memory_thread`'s `data_queue` is a `queue.Queue` that does
    #           a blocking `put` if the queue is full. So there is no above
    #           problem, but we do need to wrap the `put` in a loop that breaks
    #           not only upon success, but also when the main process stops
    #           reading, i.e., is shutting down.
    #
    #
    # Now let's get back to 1:
    #   how we gracefully exit the workers when the last reference to the
    #   iterator is gone.
    #
    # To achieve this, we implement the following logic along with the design
    # choices mentioned above:
    #
    # `workers_done_event`:
    #   A `multiprocessing.Event` shared among the main process and all worker
    #   processes. This is used to signal the workers that the iterator is
    #   shutting down. After it is set, they will not send processed data to
    #   queues anymore, and only wait for the final `None` before exiting.
    #   `done_event` isn't strictly needed. I.e., we can just check for `None`
    #   from the input queue, but it allows us to skip wasting resources
    #   processing data if we are already shutting down.
    #
    # `pin_memory_thread_done_event`:
    #   A `threading.Event` for a similar purpose to that of
    #   `workers_done_event`, but is for the `pin_memory_thread`. The reason
    #   that separate events are needed is that `pin_memory_thread` reads from
    #   the output queue of the workers. But the workers, upon seeing that
    #   `workers_done_event` is set, only wants to see the final `None`, and is
    #   not required to flush all data in the output queue (e.g., it may call
    #   `cancel_join_thread` on that queue if its `IterableDataset` iterator
    #   happens to exhaust coincidentally, which is out of the control of the
    #   main process). Thus, since we will exit `pin_memory_thread` before the
    #   workers (see below), two separete events are used.
    #
    # NOTE: In short, the protocol is that the main process will set these
    #       `done_event`s and then the corresponding processes/threads a `None`,
    #       and that they may exit at any time after receiving the `None`.
    #
    # NOTE: Using `None` as the final signal is valid, since normal data will
    #       always be a 2-tuple with the 1st element being the index of the data
    #       transferred (different from dataset index/key), and the 2nd being
    #       either the dataset key or the data sample (depending on which part
    #       of the data model the queue is at).
    #
    # [ worker processes ]
    #   While loader process is alive:
    #     Get from `index_queue`.
    #       If get anything else,
    #          Check `workers_done_event`.
    #            If set, continue to next iteration
    #                    i.e., keep getting until see the `None`, then exit.
    #            Otherwise, process data:
    #                If is fetching from an `IterableDataset` and the iterator
    #                    is exhausted, send an `_IterableDatasetStopIteration`
    #                    object to signal iteration end. The main process, upon
    #                    receiving such an object, will send `None` to this
    #                    worker and not use the corresponding `index_queue`
    #                    anymore.
    #       If timed out,
    #          No matter `workers_done_event` is set (still need to see `None`)
    #          or not, must continue to next iteration.
    #   (outside loop)
    #   If `workers_done_event` is set,  (this can be False with `IterableDataset`)
    #     `data_queue.cancel_join_thread()`.  (Everything is ending here:
    #                                          main process won't read from it;
    #                                          other workers will also call
    #                                          `cancel_join_thread`.)
    #
    # [ pin_memory_thread ]
    #   # No need to check main thread. If this thread is alive, the main loader
    #   # thread must be alive, because this thread is set as daemonic.
    #   While `pin_memory_thread_done_event` is not set:
    #     Get from `index_queue`.
    #       If timed out, continue to get in the next iteration.
    #       Otherwise, process data.
    #       While `pin_memory_thread_done_event` is not set:
    #         Put processed data to `data_queue` (a `queue.Queue` with blocking put)
    #         If timed out, continue to put in the next iteration.
    #         Otherwise, break, i.e., continuing to the out loop.
    #
    #   NOTE: we don't check the status of the main thread because
    #           1. if the process is killed by fatal signal, `pin_memory_thread`
    #              ends.
    #           2. in other cases, either the cleaning-up in __del__ or the
    #              automatic exit of daemonic thread will take care of it.
    #              This won't busy-wait either because `.get(timeout)` does not
    #              busy-wait.
    #
    # [ main process ]
    #   In the DataLoader Iter's `__del__`
    #     b. Exit `pin_memory_thread`
    #          i.   Set `pin_memory_thread_done_event`.
    #          ii   Put `None` in `worker_result_queue`.
    #          iii. Join the `pin_memory_thread`.
    #          iv.  `worker_result_queue.cancel_join_thread()`.
    #
    #     c. Exit the workers.
    #          i.   Set `workers_done_event`.
    #          ii.  Put `None` in each worker's `index_queue`.
    #          iii. Join the workers.
    #          iv.  Call `.cancel_join_thread()` on each worker's `index_queue`.
    #
    #        NOTE: (c) is better placed after (b) because it may leave corrupted
    #              data in `worker_result_queue`, which `pin_memory_thread`
    #              reads from, in which case the `pin_memory_thread` can only
    #              happen at timeing out, which is slow. Nonetheless, same thing
    #              happens if a worker is killed by signal at unfortunate times,
    #              but in other cases, we are better off having a non-corrupted
    #              `worker_result_queue` for `pin_memory_thread`.
    #
    #   NOTE: If `pin_memory=True`, there is no `pin_memory_thread` and (b)
    #         can be omitted
    #
    # NB: `done_event`s isn't strictly needed. E.g., we can just check for
    #     `None` from `index_queue`, but it allows us to skip wasting resources
    #     processing indices already in `index_queue` if we are already shutting
    #     down.

    def __init__(self, loader):
        super(_MultiProcessingDataLoaderIter, self).__init__(loader)
        
        assert self._num_workers > 0
        assert self._prefetch_factor > 0

        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context
        
        self._batch_size = loader.batch_size
        self._microbatch_size = loader.worker_batch_size
        self._microbatch_to_minibatch = self._batch_size // self._microbatch_size
        self._worker_batch_counter = 0
        self._progress_worker_batch_counter = 0
        self._next_progress_worker_batch_counter = 0
        self._complete_batch_counter = 0
        
        self._stray_task_key = []
        self._origin_workers = []
        self._except_origin_workers = []
        self._aggr_shape_hint = loader.aggr_shape_hint
        
                
        self._worker_batch_target_counter = self._batch_size
        # print(f"batch size: {self._batch_size}")
        # FIXME: Give More flexiable with aggregation template
        # works with other dataset
        
        self._aggr_template = [torch.empty(self._aggr_shape_hint),
                               torch.empty(self._batch_size, dtype=torch.long)]
        
        self._worker_init_fn = loader.worker_init_fn
        self._max_workers = loader.max_workers
        # No certainty which module multiprocessing_context is
        self._worker_result_queue = multiprocessing_context.Queue()  # type: ignore
        self._task_queue = multiprocessing_context.Queue()

        for i in range(self._max_workers):
            if self._tail_cpu_id <= i and i < self._num_workers + self._tail_cpu_id:
                self._origin_workers.append(i)
            else:
                self._except_origin_workers.append(i)
        
        # For debug
        # self._toggle = self._rank
        # if __debug__:
        #     log.debug(f"create _worker_result_queue:\n{self._worker_result_queue}")
        self._worker_pids_set = False
        self._shutdown = False
        self._worker_controller_stop_event = multiprocessing_context.Event()
        self._intentional_stop_event = multiprocessing_context.Event()
        self._gpu_finish_event = multiprocessing_context.Event()
        self._intentional_stop_event.set()
        self._iter_start_event = multiprocessing_context.Event()
        self._controller_flag = True
        try:
            self._world_size = dist.get_world_size()
            self._single_worker = False
        except RuntimeError:
            self._world_size = 1
            self._single_worker = True
            self._max_workers = self._num_workers
        self._worker_queue_idx_cycle = itertools.cycle(range(self._max_workers))
        self._workers_done_event = multiprocessing_context.Event()
        self._workers_control_events = [multiprocessing_context.Event() for _ in range(self._max_workers)]
        self._workers_status = [multiprocessing_context.Value(c_bool, False) for _ in range(self._max_workers)]
        self._local_manager_status = multiprocessing_context.Value(c_bool, False)
        self._index_queues = []
        self._workers = []
        self._gpu_workers = []
        self._worker_progress_info = []
        self._internal_shm_metas = []
        self._aug_type = loader.aug_type
        num_gpus = torch.cuda.device_count()
        # self._gpu_dali_queue = multiprocessing_context.Queue()
        self._gpu_dali_queues = []
        self.gpu_done_event = multiprocessing_context.Event()
        self._save_np_arr_to_shm(
                type="worker_progress_info", worker_id=self._max_workers, shm_arrs=self._worker_progress_info, np_arr=np.array([-1, -1], dtype=gc.INFO_DATA_TYPE), dtype=gc.INFO_DATA_TYPE)
        
        # Single Sample is not supported!
        assert not self._single_sample
        
        print("GPU_TINYBATCH_NUM", _utils.GPU_TINYBATCH_NUM, flush=True)
        self._gpu_batch_size = self._microbatch_size
        # self._dali_not_in_progress_event = multiprocessing_context.Event()

        for i in range(self._max_workers):
            self._save_np_arr_to_shm(
                type="worker_progress_info", worker_id=i, shm_arrs=self._worker_progress_info, np_arr=np.array([-1, -1], dtype=gc.INFO_DATA_TYPE), dtype=gc.INFO_DATA_TYPE)
            # No certainty which module multiprocessing_context is
            index_queue = multiprocessing_context.Queue()  # type: ignore
            # if __debug__:
            #     log.debug(f"CREATE single thread index_queue:\n{index_queue}")
            # index_queue.cancel_join_thread()
            w = multiprocessing_context.Process(
                target=_utils.worker._worker_loop,
                args=(self._dataset_kind, self._dataset, index_queue,
                      self._worker_result_queue, self._workers_done_event,
                      self._auto_collation, self._collate_fn, self._drop_last,
                      self._base_seed + i, self._worker_init_fn, i, self._num_workers,
                      self._persistent_workers, self._workers_control_events[i], self._workers_status[i], 
                      self.cur_cpus, loader.dataloader_processes, self._rank, self._gpu_dali_queues, self._aug_type))
            
            w.daemon = True

            # NOTE: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self._workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            w.start()
            # if __debug__:
            #     log.debug(f"Start worker{i}")
            self._index_queues.append(index_queue)
            # if __debug__:
            #     log.debug(f"APPEND worker{i} index_queue:{index_queue} to")
            self._workers.append(w)
        
        # print("GPU worker initiate")
        # self.gpu_worker_process = multiprocessing_context.Process(
        #         target=_utils.randaugment_dali._gpu_worker_loop,
        #         args=(self._aug_type,
        #               self._gpu_dali_queue, self._worker_result_queue, self._index_queues, self._origin_workers, self._dataset, 
        #               self._gpu_batch_size, self._gpu_batch_size * _utils.GPU_TINYBATCH_NUM,
        #               self._rank, self._world_size, 1, self.gpu_done_event, self._gpu_finish_event, self._max_workers, self._aggr_shape_hint,
        #               self._dali_not_in_progress_event))
        # self.gpu_worker_process.daemon = True
        # self.gpu_worker_process.start()
        num_gpu_workers = 4
        # cpus_to_pin_for_gpu = [None, 0, 1, 2], for single GPU!
        cpus_to_pin_for_gpu = [(None, 0, 1, 2),
                               (None, 5, 6, 7),
                               (None, 10, 11, 12),
                               (None, 16, 17, 18),
                               (None, 21, 22, 24),
                               (None, 26, 27, 28)][self._rank] # Multi GPU
        self._dali_not_in_progress_events = []
        for i in range(num_gpu_workers):
            print(f"GPU worker {i} initiate")
            gpu_dali_queue = multiprocessing_context.Queue() 
            dali_not_in_progress_event = multiprocessing_context.Event()
            w = multiprocessing_context.Process(
                target=_utils.randaugment_dali._gpu_worker_loop,
                args=(self._aug_type,
                      gpu_dali_queue, self._worker_result_queue, self._index_queues, self._origin_workers, self._dataset, 
                      self._gpu_batch_size, self._gpu_batch_size * _utils.GPU_TINYBATCH_NUM,
                      self._rank, self._world_size, 1, self.gpu_done_event, self._gpu_finish_event, self._max_workers, self._aggr_shape_hint,
                      dali_not_in_progress_event, cpus_to_pin_for_gpu[i], (True, True, True, True)[i], (num_gpu_workers, i)))
            w.daemon = True
            w.start()
            self._gpu_workers.append(w)
            self._gpu_dali_queues.append(gpu_dali_queue)
            self._dali_not_in_progress_events.append(dali_not_in_progress_event)

        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()
            self._pin_memory_intentional_stop_solved_event = threading.Event()
            
            # Queue is not type-annotated
            self._data_queue = queue.Queue()  # type: ignore
            # if __debug__:
            #     log.debug(f"create _data_queue:\n{self._data_queue}")
            pin_memory_thread = threading.Thread(
                target=_utils.pin_memory._pin_memory_loop,
                args=(self._worker_result_queue, self._data_queue,
                      torch.cuda.current_device(),
                      self._pin_memory_thread_done_event,
                      self._pin_memory_intentional_stop_solved_event))
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            # Similar to workers (see comment above), we only register
            # pin_memory_thread once it is started.
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue
        #     if __debug__:
        #         log.debug(f"redefine _data_queue as _worker_result_queue:\n{self._worker_result_queue}")
        #         log.debug(f"create _data_queue:\n{self._data_queue}")
        # .pid can be None only before process is spawned (not the case, so ignore)
        # _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self._workers) + tuple(w.pid for w in self._gpu_workers))  # type: ignore
        _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self._workers))  # type: ignore
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True
        # self._dataloader_phase_event = multiprocessing_context.Event()
        # Start background worker_control_process
        self.worker_control_process = multiprocessing_context.Process(
                target=_utils.worker_controller._control_loop,
                args=(self._rank, self._index_queues, self._task_queue, self._index_sampler, self._workers_control_events, self._intentional_stop_event, self._prefetch_factor, 
                      self._worker_controller_stop_event, self._origin_workers, self._except_origin_workers, self._max_workers, self.cur_cpus, self._worker_queue_idx_cycle,
                      self._microbatch_to_minibatch, self._iter_start_event, self._persistent_workers, self._num_workers, loader.dataloader_processes, self._local_manager_status,
                      self._microbatch_size, self._workers_status, self._dataset, self._world_size, self._gpu_dali_queues, self._single_sample,
                      self._remote_dataset, self._data_queue, self._collate_fn, self._control_queue))
        self.worker_control_process.daemon = True
        self.worker_control_process.start()
        #########################add self.dataset##################
        
        
        self._reset(loader, first_iter=True)
        
    def _save_np_arr_to_shm(self, type, worker_id, shm_arrs, np_arr, dtype):
        shm_name = f'{type}{self._rank}_{int(worker_id)}'
        shape = np_arr.shape
        nbytes = np_arr.nbytes

        if __debug__:
            log.debug(f"GPU: {self._rank} Local Manager Name:{shm_name} size: {nbytes}")
        print(f"GPU: {self._rank} Local Manager Name:{shm_name} size: {nbytes}")

        shm = SharedMemory(shm_name, O_CREAT, size=nbytes)
        shm_buf = mmap.mmap(shm.fd, nbytes)
        shm.close_fd()
        shm_np_arr = np.ndarray(shape, dtype=dtype, buffer=shm_buf)
        # if __debug__:
        # log.debug(f"Global_Controller shm arr: {shm_np_arr}")
        shm_np_arr[:] = np_arr[:]
        # if __debug__:
        # log.debug(f"After Alloc, Global_Controller shm arr: {shm_np_arr}")

        shm_arrs.append(shm_np_arr)
        self._internal_shm_metas.append((shm_buf, shm_name))    
        
    def _reset(self, loader, first_iter=False):
        super()._reset(loader, first_iter)
        # self._local_manager_status.value = True
        self._send_idx = 0  # idx of the next task to be sent to workers
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__
        # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)
        self._task_info = {}
        self._tasks_outstanding = 0  # always equal to count(v for v in task_info.values() if len(v) == 1)
        self._complete_batch_counter = 0
        self._gc_helper.Send(current_batch=self._complete_batch_counter, 
                                        micro_batch_num=0)
        # A list of booleans representing whether each worker still has work to
        # do, i.e., not having exhausted its iterable dataset object. It always
        # contains all `True`s if not using an iterable-style dataset
        # (i.e., if kind != Iterable).
        # Not that this indicates that a worker still has work to do *for this epoch*.
        # It does not mean that a worker is dead. In case of `_persistent_workers`, 
        # the worker will be reset to available in the next epoch.
        if self._num_workers > self._max_workers:
            raise ValueError(f"rank: {self._rank} Max workers are small than num_workers, num_worker: {self._num_workers} tail_cpu_id: {self._tail_cpu_id} max_workers: {self._max_workers}")
        
        for i in range(self._max_workers):
            if self._tail_cpu_id <= i and i < self._num_workers + self._tail_cpu_id:
                self._workers_control_events[i].set()
            else:
                self._workers_control_events[i].clear()
        self._iter_start_event.set()
        if hasattr(self, '_pin_memory_thread'):
            self._pin_memory_intentional_stop_solved_event.set()
        if __debug__:
            log.debug(f"GPU: {self._rank} Local Manager tail_cpu_id: {self._tail_cpu_id} num_workers: {self._num_workers} max_workers: {self._max_workers}")
        
        # We resume the prefetching in case it was enabled
        if not first_iter:
            self._controller_flag = True
            for idx in range(self._max_workers):
                self._index_queues[idx].put(_utils.worker._ResumeIteration())
                if __debug__:
                    log.debug(f"GPU: {self._rank} Local Manager PUT {_utils.worker._ResumeIteration()} to worker{idx}")
            # resume_iteration_cnt = self._num_workers
            # while resume_iteration_cnt > 0:
            #     return_idx, return_data = self._get_data()
            #     if isinstance(return_idx, _utils.worker._ResumeIteration):
            #         assert return_data is None
            #         resume_iteration_cnt -= 1
            
        # prime the prefetch loop
        
        if __debug__:
            start = time.perf_counter()
        # self._try_check_task_queue()
        if __debug__:
            end = time.perf_counter()
            log.debug(f"GPU: {self._rank} Local Manager Check task queue time: {end-start}")
        # for _ in range(self._prefetch_factor * self._num_workers):
        #     self._try_put_index()
        

    def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
        # Tries to fetch data from `self._data_queue` once for a given timeout.
        # This can also be used as inner loop of fetching without timeout, with
        # the sender status as the loop condition.
        #
        # This raises a `RuntimeError` if any worker died expectedly. This error
        # can come from either the SIGCHLD handler in `_utils/signal_handling.py`
        # (only for non-Windows platforms), or the manual check below on errors
        # and timeouts.
        #
        # Returns a 2-tuple:
        #   (bool: whether successfully get data, any: data if successful else None)
        try:
            if __debug__:
                log.debug(f"GPU: {self._rank} Local Manager wait for data queue send_idx: {self._send_idx}, rcvd_idx: {self._rcvd_idx}")
            # if self._rank == 0: 
            #     print(f"GPU: {self._rank} Local Manager wait for data queue send_idx: {self._send_idx}, rcvd_idx: {self._rcvd_idx}")    
            data = self._data_queue.get(timeout=timeout)

            # if __debug__:
            #     log.debug(f"GET data object at {hex(id(data))} data size: {sys.getsizeof(data)} _data_queue:\n{self._data_queue}")
            return (True, data)
        except Exception as e:
            # At timeout and error, we manually check whether any worker has
            # failed. Note that this is the only mechanism for Windows to detect
            # worker failures.
            failed_workers = []
            for worker_id, w in enumerate(self._workers):
                if self._workers_status[worker_id].value and not w.is_alive():
                    failed_workers.append(w)
                    self._mark_worker_as_unavailable(worker_id)
            if len(failed_workers) > 0:
                pids_str = ', '.join(str(w.pid) for w in failed_workers)
                raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str)) from e
            if isinstance(e, queue.Empty):
                return (False, None)
            import tempfile
            import errno
            try:
                # Raise an exception if we are this close to the FDs limit.
                # Apparently, trying to open only one file is not a sufficient
                # test.
                # See NOTE [ DataLoader on Linux and open files limit ]
                fds_limit_margin = 10
                fs = [tempfile.NamedTemporaryFile() for i in range(fds_limit_margin)]
            except OSError as e:
                if e.errno == errno.EMFILE:
                    raise RuntimeError(
                        "Too many open files. Communication with the"
                        " workers is no longer possible. Please increase the"
                        " limit using `ulimit -n` in the shell or change the"
                        " sharing strategy by calling"
                        " `torch.multiprocessing.set_sharing_strategy('file_system')`"
                        " at the beginning of your code") from None
            raise

# NOTE [ DataLoader on Linux and open files limit ]
#
# On Linux when DataLoader is used with multiprocessing we pass the data between
# the root process and the workers through SHM files. We remove those files from
# the filesystem as soon as they are created and keep them alive by
# passing around their file descriptors through AF_UNIX sockets. (See
# docs/source/multiprocessing.rst and 'Multiprocessing Technical Notes` in
# the wiki (https://github.com/pytorch/pytorch/wiki).)
#
# This sometimes leads us to exceeding the open files limit. When that happens,
# and the offending file descriptor is coming over a socket, the `socket` Python
# package silently strips the file descriptor from the message, setting only the
# `MSG_CTRUNC` flag (which might be a bit misleading since the manpage says that
# it _indicates that some control data were discarded due to lack of space in
# the buffer for ancillary data_). This might reflect the C implementation of
# AF_UNIX sockets.
#
# This behaviour can be reproduced with the script and instructions at the
# bottom of this note.
#
# When that happens, the standard Python `multiprocessing` (and not
# `torch.multiprocessing`) raises a `RuntimeError: received 0 items of ancdata`
#
# Sometimes, instead of the FD being stripped, you may get an `OSError:
# Too many open files`, both in the script below and in DataLoader. However,
# this is rare and seems to be nondeterministic.
#
#
#   #!/usr/bin/env python3
#   import sys
#   import socket
#   import os
#   import array
#   import shutil
#   import socket
#
#
#   if len(sys.argv) != 4:
#       print("Usage: ", sys.argv[0], " tmp_dirname iteration (send|recv)")
#       sys.exit(1)
#
#   if __name__ == '__main__':
#       dirname = sys.argv[1]
#       sock_path = dirname + "/sock"
#       iterations = int(sys.argv[2])
#       def dummy_path(i):
#           return dirname + "/" + str(i) + ".dummy"
#
#
#       if sys.argv[3] == 'send':
#           while not os.path.exists(sock_path):
#               pass
#           client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
#           client.connect(sock_path)
#           for i in range(iterations):
#               fd = os.open(dummy_path(i), os.O_WRONLY | os.O_CREAT)
#               ancdata = array.array('i', [fd])
#               msg = bytes([i % 256])
#               print("Sending fd ", fd, " (iteration #", i, ")")
#               client.sendmsg([msg], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, ancdata)])
#
#
#       else:
#           assert sys.argv[3] == 'recv'
#
#           if os.path.exists(dirname):
#               raise Exception("Directory exists")
#
#           os.mkdir(dirname)
#
#           print("Opening socket...")
#           server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
#           server.bind(sock_path)
#
#           print("Listening...")
#           for i in range(iterations):
#               a = array.array('i')
#               msg, ancdata, flags, addr = server.recvmsg(1, socket.CMSG_SPACE(a.itemsize))
#               assert(len(ancdata) == 1)
#               cmsg_level, cmsg_type, cmsg_data = ancdata[0]
#               a.frombytes(cmsg_data)
#               print("Received fd ", a[0], " (iteration #", i, ")")
#
#           shutil.rmtree(dirname)
#
# Steps to reproduce:
#
# 1. Run two shells and set lower file descriptor limit in the receiving one:
# (shell1) ulimit -n 1020
# (shell2) ulimit -n 1022
#
# 2. Run the script above with the `recv` option in the first shell
# (shell1) ./test_socket.py sock_tmp 1017 recv
#
# 3. Run the script with the `send` option in the second shell:
# (shell2) ./test_socket.py sock_tmp 1017 send

    def _get_data(self):
        # Fetches data from `self._data_queue`.
        #
        # We check workers' status every `MP_STATUS_CHECK_INTERVAL` seconds,
        # which we achieve by running `self._try_get_data(timeout=MP_STATUS_CHECK_INTERVAL)`
        # in a loop. This is the only mechanism to detect worker failures for
        # Windows. For other platforms, a SIGCHLD handler is also used for
        # worker failure detection.
        #
        # If `pin_memory=True`, we also need check if `pin_memory_thread` had
        # died at timeouts.
        if self._timeout > 0:
            success, data = self._try_get_data(self._timeout)
            if success:
                return data
            else:
                raise RuntimeError('DataLoader timed out after {} seconds'.format(self._timeout))
        elif self._pin_memory:
            while self._pin_memory_thread.is_alive():
                success, data = self._try_get_data()
                if success:
                    return data
            else:
                # while condition is false, i.e., pin_memory_thread died.
                raise RuntimeError('Pin memory thread exited unexpectedly')
            # In this case, `self._data_queue` is a `queue.Queue`,. But we don't
            # need to call `.task_done()` because we don't use `.join()`.
        else:
            while True:
                success, data = self._try_get_data()
                if success:
                    return data

    def _next_data(self):
        self._gc_helper.Gpu_done()
        print("[gpu_finish_event DATALOADER] Set to True", flush=True)
        # self._dataloader_phase_event.set()
        self._gpu_finish_event.set()
        self._local_manager_status.value = True
        # Initial for new mini-batch
        self._worker_batch_counter = 0
        if self._next_progress_worker_batch_counter > self._worker_batch_target_counter:
            self._progress_worker_batch_counter = self._worker_batch_target_counter
            self._next_progress_worker_batch_counter -= self._worker_batch_target_counter
        else:
            self._progress_worker_batch_counter = self._next_progress_worker_batch_counter
            self._next_progress_worker_batch_counter = 0
            
        if self._progress_worker_batch_counter != 0:
            self._gc_helper.Send(current_batch=self._complete_batch_counter, 
                                            micro_batch_num=self._progress_worker_batch_counter)
            
        # FIXME: Hard-code for image data only, return single data
        complete_data = [self._aggr_template[0].cuda(self._rank, non_blocking=True),
                        self._aggr_template[1].cuda(self._rank, non_blocking=True)]
        self._try_check_task_queue()
        self._gc_helper.Job_start()
        
        if __debug__:
            log.debug(f"GPU: {self._rank} Local Manager Enter the next_data rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}")
        
        while True:
            if not self._intentional_stop_event.is_set() and self._data_queue.qsize() < 0:
                self._check_task_queue()
                if __debug__:
                    log.debug(f"GPU: {self._rank} Local Manager Wait for intentional stop")
                # if hasattr(self, '_pin_memory_thread'):
                #     self._pin_memory_intentional_stop_solved_event.clear()
                self._intentional_stop_event.wait()
                # if hasattr(self, '_pin_memory_thread'):
                #     self._pin_memory_intentional_stop_solved_event.set()
                if __debug__:
                    log.debug(f"GPU: {self._rank} Local Manager restart from intentional stop")
                # check intentional stop workers by global manager
                # time.sleep(_utils.MP_CONTROL_CHECK_INTERVAL)
            
            self._check_task_queue()
            # if __debug__:
            #     start = time.perf_counter()
            # if __debug__:
            #     end = time.perf_counter()
            #     log.debug(f"GPU: {self._rank} Local Manager Check task queue time: {end-start}")
            # If the worker responsible for `self._rcvd_idx` has already ended
            # and was unable to fulfill this task (due to exhausting an `IterableDataset`),
            # we try to advance `self._rcvd_idx` to find the next valid index.
            #
            # This part needs to run in the loop because both the `self._get_data()`
            # call and `_IterableDatasetStopIteration` check below can mark
            # extra worker(s) as dead.
            while self._rcvd_idx < self._send_idx:
                try:
                    info = self._task_info[self._rcvd_idx]
                except KeyError:
                    if __debug__:
                        log.debug(f"GPU: {self._rank} Local Manager fallback try to check task queue to test")
                    # if self._rank == 0: 
                    #     print(f"GPU: {self._rank} Local Manager fallback try to check task queue to test")
                    # self._try_check_task_queue()
                    self._task_info[self._rcvd_idx] = (-1,)
                    info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if worker_id == _utils.MP_GPU_WORKER:
                    if __debug__:
                        log.debug(f"GPU: {self._rank} Local Manager test idx {self._rcvd_idx}, info len: {len(info)} worker_id: {worker_id}, worker_status: GPU")
                    # if self._rank == 0: 
                    #     print(f"GPU: {self._rank} Local Manager test idx {self._rcvd_idx}, info len: {len(info)} worker_id: {worker_id}, worker_status: GPU")
                else:
                    if __debug__:
                        log.debug(f"GPU: {self._rank} Local Manager test idx {self._rcvd_idx}, info len: {len(info)} worker_id: {worker_id}, worker_status: {self._workers_status[worker_id].value}")
                    # if self._rank == 0: 
                    #     print(f"GPU: {self._rank} Local Manager test idx {self._rcvd_idx}, info len: {len(info)} worker_id: {worker_id}, worker_status: {self._workers_status[worker_id].value}")
                if len(info) == 2 or worker_id == -1 or worker_id == _utils.MP_GPU_WORKER or self._workers_status[worker_id].value or self._data_queue.qsize() > 0:  # has data or on stray or is still active
                    break
                
                del self._task_info[self._rcvd_idx]
                if __debug__:
                    log.debug(f"GPU: {self._rank} Local Manager delete idx {self._rcvd_idx}, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}, worker_id: {worker_id}")
                # if self._rank == 0: 
                #     print(f"GPU: {self._rank} Local Manager delete idx {self._rcvd_idx}, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}, worker_id: {worker_id}")
                self._rcvd_idx += 1
            else:
                
                # Handle for remaining data
                # if self._worker_batch_counter != 0:
                #     if __debug__:
                #         log.debug(f"GPU: {self._rank} Local Manager remaining complete data idx {self._rcvd_idx}, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}")
                #     # self._worker_reset()
                #     complete_data[0] = complete_data[0][:self._worker_batch_counter]
                #     complete_data[1] = complete_data[1][:self._worker_batch_counter]
                #     return 
                if self._controller_flag or not self._intentional_stop_event.is_set():
                    self._try_check_task_queue()
                    continue
                
                if __debug__:
                    log.debug(f"GPU: {self._rank} Local Manager Stop Iteration, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}")
                # print(f"GPU: {self._rank} Local Manager Stop Iteration, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}")
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                if not self._persistent_workers:
                    self._gc_helper.Job_done()
                    self._shutdown_workers()
                print("Dataloader line 1432", self._send_idx, self._rcvd_idx, flush=True)
                raise StopIteration

            # Now `self._rcvd_idx` is the batch index we want to fetch
            
            # Check if the next sample has already been generated
            if len(self._task_info[self._rcvd_idx]) == 2:
                worker_id, data = self._task_info.pop(self._rcvd_idx)
                if __debug__:
                    start=time.perf_counter()
                self._process_data(complete_data, data)
                if __debug__:
                    end=time.perf_counter()
                    log.debug(f"GPU: {self._rank} Local Manager Process Buffer data: {end-start} idx {self._rcvd_idx-1}, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}")
                # if self._rank == 0: 
                #     print(f"GPU: {self._rank} Local Manager Process Buffer data: {end-start} idx {self._rcvd_idx-1}, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}")
                
                if self._worker_batch_counter >= self._worker_batch_target_counter or self._shutdown:
                    self._complete_batch_counter += 1
                    self._gc_helper.Send(current_batch=self._complete_batch_counter, 
                                         micro_batch_num=0)
                    # self._worker_control()
                    # self._check_queue_put_index()
                    if __debug__:
                        end=time.perf_counter()
                        log.debug(f"GPU: {self._rank} Local Manager Complete data: {end-start} idx {self._rcvd_idx}, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}")
                    # if self._rank == 0: 
                    #     print(f"GPU: {self._rank} Local Manager Complete data: {end-start} idx {self._rcvd_idx}, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}")
                    # self._worker_reset()
                    # self._gc_helper.Job_done()
                    self._local_manager_status.value = False
                    print("[gpu_finish_event DATALOADER] Set to False", flush=True)

                    if _utils.USE_SHARED_MEMORY:
                        self._gpu_finish_event.clear()
                        for e in self._dali_not_in_progress_events:
                            e.wait()
                        # self._dali_not_in_progress_event.wait()
                    # self._dataloader_phase_event.clear()
                    self._gc_helper.Gpu_start()
                    return complete_data
                
                del data
                # if __debug__:
                #     end=time.perf_counter()
                #     log.debug(f"GPU: {self._rank} Local Manager Processing data: {end-start} idx {self._rcvd_idx}, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}")
                # self._worker_control()
                # self._check_queue_put_index()
                continue

            assert not self._shutdown and self._tasks_outstanding > 0
            idx, data = self._get_data()
            print(f"Received Idx {idx}!", flush=True)
            
            # if isinstance(idx, _utils.worker._ResumeIteration):
            #     assert data is None
            #     continue
            
            
            if __debug__:
                start=time.perf_counter()
            self._tasks_outstanding -= 1
            
            # if __debug__:
            #     log.debug(f"tasks_outstanding down to {se lf._tasks_outstanding}")
            if self._dataset_kind == _DatasetKind.Iterable:
                # Check for _IterableDatasetStopIteration
                if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                    if self._persistent_workers:
                        self._workers_control_events[data.worker_id].clear()
                        # if __debug__:
                        #     log.debug(f"FALSE worker id{data.worker_id} _workers_status:\n{self._workers_status}")
                    else:
                        self._mark_worker_as_unavailable(data.worker_id)
                    # self._worker_control()
                    # self._check_queue_put_index()
                    continue
            
            # self._worker_control()
            # self._check_queue_put_index()
            # self._try_put_index()
            
            # Sequential update
            if self._progress_worker_batch_counter < self._worker_batch_target_counter:
                self._progress_worker_batch_counter += self._microbatch_size
            else:
                self._next_progress_worker_batch_counter += self._microbatch_size
            
            if idx != self._rcvd_idx:
                # store out-of-order samples
                try:
                    self._task_info[idx] += (data,)
                except KeyError:
                    if __debug__:
                        log.debug(f"GPU: {self._rank} Local Manager fallback try to check task queue to buffer")
                    # self._try_check_task_queue()
                    self._task_info[idx] = (-2,data)
                    self._send_idx += 1
                    self._tasks_outstanding += 1
                if __debug__:
                    end=time.perf_counter()
                    log.debug(f"GPU: {self._rank} Local Manager Buffer data: {end-start} idx {idx}, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}")
            else:
                self._process_data(complete_data, data)
                
                del data
                    
                if self._worker_batch_counter >= self._worker_batch_target_counter or self._shutdown:
                    self._complete_batch_counter += 1
                    self._gc_helper.Send(current_batch=self._complete_batch_counter, 
                                         micro_batch_num=0)
                    # self._worker_control()
                    # self._check_queue_put_index()
                    if __debug__:
                        end=time.perf_counter()
                        log.debug(f"GPU: {self._rank} Local Manager Complete data: {end-start} idx {idx}, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}")
                    # if self._rank == 0: 
                    #     print(f"GPU: {self._rank} Local Manager Complete data: {end-start} idx {idx}, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}")
                    # self._worker_reset()
                    # self._gc_helper.Job_done()
                    self._local_manager_status.value = False
                    print("[gpu_finish_event DATALOADER] Set to False", flush=True)

                    if _utils.USE_SHARED_MEMORY:
                        self._gpu_finish_event.clear()
                        for e in self._dali_not_in_progress_events:
                            e.wait()
                        # self._dali_not_in_progress_event.wait()
                    # self._dataloader_phase_event.clear()
                    self._gc_helper.Gpu_start()
                    return complete_data
                
                if __debug__:
                    end=time.perf_counter()
                    log.debug(f"GPU: {self._rank} Local Manager Processing data: {end-start} idx {idx}, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}")
                # if self._rank == 0: 
                #     print(f"GPU: {self._rank} Local Manager Processing data: {end-start} idx {idx}, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}")
            self._gc_helper.Send(current_batch=self._complete_batch_counter, 
                                         micro_batch_num=self._progress_worker_batch_counter)
            
            
    def _try_check_task_queue(self):
        if not self._controller_flag:
            return
        
        try:
            new_task_infos=self._task_queue.get(timeout=5)
        except queue.Empty:
            return
        
        if isinstance(new_task_infos, _utils.worker_controller._ControllerStopSignal):
            self._controller_flag = False
            if __debug__:
                log.debug(f"GPU: {self._rank} Local Manager detect stop signal of worker controller")
            return
        
        for idx in new_task_infos:
            if not idx in self._task_info:
                self._task_info[idx] = new_task_infos[idx]
                self._send_idx += 1
                self._tasks_outstanding += 1
            elif idx in self._task_info and len(self._task_info[idx]) != 2:
                self._task_info[idx] = new_task_infos[idx]
        
        if __debug__:
            log.debug(f"GPU: {self._rank} Local Manager task_info update")
        
        
    def _check_task_queue(self):
        if not self._controller_flag:
            return
        
        if self._task_queue.qsize() > 0:
            try:
                new_task_infos=self._task_queue.get(timeout=0.003)
            except queue.Empty:
                return
        else:
            return
        # else:
        #     if __debug__:
        #         log.debug(f"GPU: {self._rank} Local Manager no_task_info")
        #     return
        
        if isinstance(new_task_infos, _utils.worker_controller._ControllerStopSignal):
            self._controller_flag = False
            if __debug__:
                log.debug(f"GPU: {self._rank} Local Manager detect stop signal of worker controller")
            return
        
        for idx in new_task_infos:
            if not idx in self._task_info:
                self._task_info[idx] = new_task_infos[idx]
                self._send_idx += 1
                self._tasks_outstanding += 1
            elif len(self._task_info[idx]) != 2:
                self._task_info[idx] = new_task_infos[idx]
        
        if __debug__:
            log.debug(f"GPU: {self._rank} Local Manager task_info update")
    
    
    def _process_data(self, complete_data, data):
        # if __debug__:
        #     log.debug(f"PROCESS data object at {hex(id(data))}")
        self._rcvd_idx += 1
        
        if isinstance(data, ExceptionWrapper):
            data.reraise()
        
        batch_size = data[0].shape[0]
        next_worker_batch = self._worker_batch_counter + batch_size
        if data[0].device == torch.device('cpu'):
            data_cpu = data
            data[0] = data[0].cuda(self._rank, non_blocking=True)
            data[1] = data[1].cuda(self._rank, non_blocking=True)
            # torch.cuda.synchronize()
            del data_cpu

        if __debug__:
            log.debug(f"GPU: {self._rank} aggr shape {data[0].shape} {data[1].shape} to {complete_data[0][self._worker_batch_counter:next_worker_batch].shape} {complete_data[1][self._worker_batch_counter:next_worker_batch].shape}")

        complete_data[0][self._worker_batch_counter:next_worker_batch].copy_(data[0])
        complete_data[1][self._worker_batch_counter:next_worker_batch].copy_(data[1])
        if __debug__:
            log.debug(f"GPU: {self._rank} label {data[1]} to {complete_data[1][self._worker_batch_counter:next_worker_batch]}")
        del data

        self._worker_batch_counter = next_worker_batch
        
    def _mark_worker_as_unavailable(self, worker_id, shutdown=False):
        # Mark a worker as having finished its work e.g., due to
        # exhausting an `IterableDataset`. This should be used only when this
        # `_MultiProcessingDataLoaderIter` is going to continue running.

        assert self._workers_control_events[worker_id].is_set() or (self._persistent_workers and shutdown)

        # Signal termination to that specific worker.
        q = self._index_queues[worker_id]
        # if __debug__:
        #     log.debug(f"TERMINATE worker id{worker_id} index_queue:\n{q}")
        # Indicate that no more data will be put on this queue by the current
        # process.
        self._workers_control_events[worker_id].set()
        q.put(None)
        if __debug__:
            log.debug(f"TERMINATE worker id{worker_id}")
        # Note that we don't actually join the worker here, nor do we remove the
        # worker's pid from C side struct because (1) joining may be slow, and
        # (2) since we don't join, the worker may still raise error, and we
        # prefer capturing those, rather than ignoring them, even though they
        # are raised after the worker has finished its job.
        # Joinning is deferred to `_shutdown_workers`, which it is called when
        # all workers finish their jobs (e.g., `IterableDataset` replicas) or
        # when this iterator is garbage collected.

        # self._workers_status[worker_id] = False
        # self._workers_control_events[worker_id].clear()
        # if __debug__:
        #     log.debug(f"FALSE worker id{worker_id} _workers_status:\n{self._workers_status}")
        assert self._workers_done_event.is_set() == shutdown

    def _shutdown_workers(self):
        if __debug__:
            log.debug(f"GPU: {self._rank} Local Manager Shutdown workers rcvd_idx: {self._rcvd_idx}, send_idx: {self._send_idx}")
        # if self._rank == 0: 
        #     print(f"GPU: {self._rank} Local Manager Shutdown workers rcvd_idx: {self._rcvd_idx}, send_idx: {self._send_idx}")
        # Called when shutting down this `_MultiProcessingDataLoaderIter`.
        # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on
        # the logic of this function.
        python_exit_status = _utils.python_exit_status
        if python_exit_status is True or python_exit_status is None:
            # See (2) of the note. If Python is shutting down, do no-op.
            return
        # Normal exit when last reference is gone / iterator is depleted.
        # See (1) and the second half of the note.
        if not self._shutdown:
            self._shutdown = True
            # Kill worker controller and cleanup
            if hasattr(self, 'worker_control_process'):
                self._worker_controller_stop_event.set()
                if __debug__:
                    log.debug(f"GPU: {self._rank} Local Manager Shutdown worker controller")
            
            try:
                for shm_buf, shm_name in self._internal_shm_metas:
                    shm_buf.close()
                    unlink_shared_memory(shm_name)
            except Exception as e:
                log.warn(e)
                
            try:
                # Exit `pin_memory_thread` first because exiting workers may leave
                # corrupted data in `worker_result_queue` which `pin_memory_thread`
                # reads from.
                if hasattr(self, '_pin_memory_thread'):
                    # Use hasattr in case error happens before we set the attribute.
                    self._pin_memory_intentional_stop_solved_event.set()
                    self._pin_memory_thread_done_event.set()
                    # Send something to pin_memory_thread in case it is waiting
                    # so that it can wake up and check `pin_memory_thread_done_event`
                    self._worker_result_queue.put((None, None))
                    if __debug__:
                        log.debug(f"GPU{self._rank}: PUT (None, None)")
                    self._pin_memory_thread.join()
                    # if __debug__:
                    #     log.debug(f"join _pin_memory_thread")
                    self._worker_result_queue.cancel_join_thread()
                    # if __debug__:
                    #     log.debug(f"cancel_join_thread _worker_result_queue:\n{self._worker_result_queue}")
                    self._worker_result_queue.close()
                    # if __debug__:
                    #     log.debug(f"close _worker_result_queue:\n{self._worker_result_queue}")

                # for w in self._gpu_workers:
                #     w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                #     if w.is_alive():
                #         # Existing mechanisms try to make the workers exit
                #         # peacefully, but in case that we unfortunately reach
                #         # here, which we shouldn't, (e.g., pytorch/pytorch#39570),
                #         # we kill the worker.
                #         w.terminate()

                # for q in self._gpu_dali_queues:
                #     q.cancel_join_thread()
                #     q.close()


                # Exit workers now.
                
                self._workers_done_event.set()
                for worker_id in range(len(self._workers)):
                    # Get number of workers from `len(self._workers)` instead of
                    # `self._num_workers` in case we error before starting all
                    # workers.
                    # If we are using workers_status with persistent_workers
                    # we have to shut it down because the worker is paused
                    if self._persistent_workers or self._workers_control_events[worker_id].is_set():
                        self._mark_worker_as_unavailable(worker_id, shutdown=True)
                for w in self._workers:
                    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                    if w.is_alive():
                        # Existing mechanisms try to make the workers exit
                        # peacefully, but in case that we unfortunately reach
                        # here, which we shouldn't, (e.g., pytorch/pytorch#39570),
                        # we kill the worker.
                        w.terminate()
                        
                for q in self._index_queues:
                    q.cancel_join_thread()
                    # if __debug__:
                    #     log.debug(f"cancel_join_thread index_queue:\n{q}")
                    q.close()
                    # if __debug__:
                    #     log.debug(f"close index_queue:\n{q}")
                
                if hasattr(self, 'worker_control_process'):
                    self.worker_control_process.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                    
                    if self.worker_control_process.is_alive():
                        self.worker_control_process.terminate()
                        
                    self._task_queue.cancel_join_thread()
                    self._task_queue.close()
            finally:
                # Even though all this function does is putting into queues that
                # we have called `cancel_join_thread` on, weird things can
                # happen when a worker is killed by a signal, e.g., hanging in
                # `Event.set()`. So we need to guard this with SIGCHLD handler,
                # and remove pids from the C side data structure only at the
                # end.
                #
                # FIXME: Unfortunately, for Windows, we are missing a worker
                #        error detection mechanism here in this function, as it
                #        doesn't provide a SIGCHLD handler.
                if self._worker_pids_set:
                    _utils.signal_handling._remove_worker_pids(id(self))
                    self._worker_pids_set = False

    def __del__(self):
        print("ENTERING DEL", sys.exc_info()[2], flush=True)
        self._gc_helper.Close()
        self._shutdown_workers()
