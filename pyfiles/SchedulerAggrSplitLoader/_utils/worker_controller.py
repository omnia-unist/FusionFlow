r""""Contains definitions of the methods used by the _BaseDataLoaderIter to control and put task index to workers
"""

import torch
from torch._six import queue, container_abcs, string_classes

from . import worker
from . import MP_STATUS_CHECK_INTERVAL
import psutil

from time import sleep, perf_counter
from torch._utils import ExceptionWrapper
from bisect import bisect_left
from collections import namedtuple
from itertools import cycle
import sys
sys.path.append("..")
from posix_ipc import O_CREAT, SharedMemory
import mmap
import globalManager as gc
from efficientnet_pytorch import EfficientNet
from vit_pytorch import ViT
import numpy as np
from copy import deepcopy
if __debug__:
    import logging  
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger(__name__)

r"""Dummy class used to resume the fetching when worker reuse is enabled"""
_ControllerStopSignal = namedtuple('_ControllerStopSignal', [])
WM_INTERVAL = 0.001
WM_QUEUE_INTERVAL = 0.002
WM_STATUS_CHECK_INTERVAL = 0.002
WM_STOP_ITERATION_SIGNAL = -2
WM_CONTINUE_ITERATION_SIGNAL = 1

class workerManager():
    def __init__(self,_workers_control_events,_max_workers,_worker_queue_idx_cycle,
                    _rank, _index_queues, _prefetch_factor, _cur_cpus, _num_workers,
                    _index_sampler,_microbatch_to_minibatch, _worker_batch_target_counter,
                    _except_origin_workers, _origin_workers, _iteration_end_status,
                    microbatch_size, worker_status, _aggr_template, _data_queue, _minibatch_done_event,
                    _controller_queue, _worker_controller_stop_event
                    ) -> None:
        self._rank = _rank
        self._gc_helper = gc.globalManagerHelper(rank = self._rank)
        self.cur_cpus = _cur_cpus
        self._max_workers = _max_workers
        self._workers_control_events = _workers_control_events
        self._worker_queue_idx_cycle = _worker_queue_idx_cycle
        self._index_queues = _index_queues
        self._index_sampler = _index_sampler
        self._stray_task_key = []
        self._microbatch_to_minibatch = _microbatch_to_minibatch
        self._num_workers = _num_workers
        self._minibatch_done_event = _minibatch_done_event
        self._prefetch_factor = _prefetch_factor
        self._send_idx  = 0
        self._rcvd_idx = 0
        self._progress_idx = None
        self._sampler_iter = None
        self._task_info = None
        self._task_batch = None
        self._next_task_idx = None
        self._worker_batch_counter = None
        self._done_iteration = False
        self._intentional_stop = False
        self._timeout = 5
        
        self._internal_shm_metas = []
        self._worker_progress_info = []
        self._microbatch_size = microbatch_size
        self._worker_batch_target_counter = _worker_batch_target_counter
        self._progress_threshold = self._microbatch_size // 2
        self._worker_info = _origin_workers       
        self._workers_status = worker_status
        self._controller_queue = _controller_queue
        self._worker_controller_stop_event = _worker_controller_stop_event
        self._queue_tasks_counter = None
        
        self._data_queue = _data_queue
        self._aggr_template = _aggr_template
        # FIXME: Hardcode dtype fix with aggr template dtype
        self.complete_data = deepcopy(self._aggr_template)

        for worker_id in range(self._max_workers):
            _worker_progress = self._load_np_arr("worker_progress_info", worker_id=worker_id, frame=np.array([-1, -1], dtype=gc.INFO_DATA_TYPE), dtype=gc.INFO_DATA_TYPE)
            self._worker_progress_info.append(_worker_progress)
        
        self._except_origin_workers = _except_origin_workers
        self._origin_workers = _origin_workers
        self._stray_task_buffer = []
        # self._iteration_end_status = _iteration_end_status
        
    def _load_np_arr(self, type, worker_id, frame, dtype=np.uint):
        shm_name = f'{type}{self._rank}_{int(worker_id)}'
        shape = frame.shape
        try:
            size = frame.nbytes
        except AttributeError:
            size = sys.getsizeof(frame.storage)
        if __debug__:
            log.debug(f"GPU: {self._rank} Name: {shm_name} size:{size}")
        shm = SharedMemory(name=shm_name)
        shm_buf = mmap.mmap(shm.fd, size)
        shm.close_fd()
        shm_np_arr = np.ndarray(
            shape=shape, dtype=dtype, buffer=shm_buf)
        self._internal_shm_metas.append(shm_buf)
        return shm_np_arr
    
    def _reset(self):
        self._done_iteration = False
        self._sampler_iter = iter(self._index_sampler)
        self._send_idx = 0  # idx of the next task to be sent to workers
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__
        self._progress_idx = self._num_workers
        self._next_task_idx = self._microbatch_to_minibatch
        self._tasks_outstanding = 0
        self._complete_batch_counter = 0
        self._progress_worker_batch_counter = 0
        self._next_progress_worker_batch_counter = 0
        self._worker_batch_counter = 0
        self._gc_helper.Send(current_batch=self._complete_batch_counter, 
                                micro_batch_num=self._progress_worker_batch_counter)

        self._index_waiting_queue = [[] for _ in range(self._max_workers)]
        self._next_index_waiting_queue = [[] for _ in range(self._max_workers)]
        self._queue_tasks_counter = 0
        self._next_queue_tasks_counter = 0
        # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)
        self._task_batch = 0
        self._gc_helper.Report_task(batch_num=self._task_batch, tasks=self._microbatch_to_minibatch-self._progress_idx)
            
        self._task_info = {}
        self._gc_helper.Job_start()
        if __debug__:
            log.debug(f"GPU: {self._rank} Reset worker manager")
            
    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

    def _try_put_target_tasks(self, task_infos):
        # assert _tasks_outstanding < self._prefetch_factor * self._num_workers
        for task in task_infos:
            idx, index = task
            
            for _ in range(self._max_workers):  # find the next active worker, if any
                worker_queue_idx = next(self._worker_queue_idx_cycle)
                if self._workers_control_events[worker_queue_idx].is_set():
                    break
            else:
                # not found (i.e., didn't break)
                # Stray update infos
                for task in task_infos:
                    idx, _ = task
                    self._put_stray_task(task)
                    self._task_info[idx] = (-1,)
                    if __debug__:
                        log.debug(f"GPU: {self._rank} Put idx {idx} to stray buffer")
                return

            self._index_queues[worker_queue_idx].put(task)
            self._task_info[idx] = (worker_queue_idx,)
            if __debug__:
                log.debug(f"GPU: {self._rank} RePUT ({idx}, {index}) to worker_queue_idx{worker_queue_idx}")

    def _try_put_index(self):
        # assert _tasks_outstanding < self._prefetch_factor * self._num_workers
        try:
            index = self._next_index()
        except StopIteration:
            self._intentional_stop = False
            return
        for _ in range(self._max_workers):  # find the next active worker, if any
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_control_events[worker_queue_idx].is_set():
                break
        else:
            if __debug__:
                log.debug(f"GPU: {self._rank} Failed to find the active worker")
            # not found (i.e., didn't break)
            return

        self._index_queues[worker_queue_idx].put((self._send_idx, index))
        self._task_info[self._send_idx] = (worker_queue_idx,)
        if __debug__:
            log.debug(f"GPU: {self._rank} PUT ({self._send_idx}, {index}) to worker_queue_idx{worker_queue_idx} task_info_idx{self._send_idx}: {self._task_info[self._send_idx]}")
        self._send_idx += 1
        # if __debug__:
        #     log.debug(f"tasks_outstanding up to {_tasks_outstanding}")
    
    def _check_queue_put_index(self):
        if self._done_iteration:
            return
        
        for worker_queue_idx in self._worker_info:  # find the next active worker, if any
            if not self._workers_control_events[worker_queue_idx].is_set():
                continue
            
            if self._gc_helper.Get_Job_done_flag() and not worker_queue_idx in self._origin_workers:
                self._update_worker_progress_info(worker_queue_idx)
                continue
            
            if self._index_queues[worker_queue_idx].qsize() < self._prefetch_factor:
                if len(self._stray_task_buffer):
                    task = self._stray_task_buffer.pop(0)
                    self._stray_task_key.pop(0)
                    self._index_queues[worker_queue_idx].put(task)
                    if task[0] < self._next_task_idx:
                        self._index_waiting_queue[worker_queue_idx].append(task[0])
                        self._queue_tasks_counter += 1
                    else:
                        self._next_index_waiting_queue[worker_queue_idx].append(task[0])
                        self._next_queue_tasks_counter += 1
                    if __debug__:
                        log.debug(f"GPU: {self._rank} Revive ({task}) from stray buffer to worker_queue_idx{worker_queue_idx}")
                    self._task_info[task[0]] = (worker_queue_idx,)
                else:
                    try:
                        index = self._next_index()
                    except StopIteration:
                        if __debug__:
                            log.debug(f"GPU: {self._rank} Worker Control increase detect stop iteration! rcvd:{self._rcvd_idx} send:{self._send_idx}")
                        self._intentional_stop = False
                        self._done_iteration = True
                        return WM_STOP_ITERATION_SIGNAL
                    self._index_queues[worker_queue_idx].put((self._send_idx, index))
                    if self._send_idx < self._next_task_idx:
                        self._index_waiting_queue[worker_queue_idx].append(self._send_idx)
                        self._queue_tasks_counter += 1
                    else:
                        self._next_index_waiting_queue[worker_queue_idx].append(self._send_idx)
                        self._next_queue_tasks_counter += 1
                    self._task_info[self._send_idx] = (worker_queue_idx,)
                    if __debug__:
                        log.debug(f"GPU: {self._rank} PUT ({self._send_idx}, {index}) to worker_queue_idx{worker_queue_idx} task_info_idx{self._send_idx}: {self._task_info[self._send_idx]}")
                    self._send_idx += 1
                    self._tasks_outstanding += 1
            self._update_worker_progress_info(worker_queue_idx)

        return WM_CONTINUE_ITERATION_SIGNAL

            
    def _update_worker_progress_info(self,worker_id):
        unsent_tasks = self._next_task_idx - self._send_idx
        idx = int(self._worker_progress_info[worker_id][0])
        
        if idx in self._index_waiting_queue[worker_id]:
            self._index_waiting_queue[worker_id].remove(idx)
            self._queue_tasks_counter -= 1
        elif unsent_tasks < 1 and len(self._index_waiting_queue[worker_id]) and self._worker_progress_info[worker_id][1] >= self._progress_threshold:
            self._index_waiting_queue[worker_id].pop(0)
            self._queue_tasks_counter -= 1
                
        if idx in self._next_index_waiting_queue[worker_id]:
            self._next_index_waiting_queue[worker_id].remove(idx)
            self._next_queue_tasks_counter -= 1
            
        true_tasks = unsent_tasks+self._queue_tasks_counter
        waiting_tasks = self._queue_tasks_counter if unsent_tasks < 0 else true_tasks
        # if __debug__:
        #     log.debug(f"GPU: {self._rank} worker_id: {worker_id} wm view:{self._index_waiting_queue[worker_id]} queue_task {self._queue_tasks_counter}, unsent tasks {unsent_tasks}")
        if waiting_tasks < 1 or true_tasks < 0:
            self._gc_helper.Job_done()
            self._next_task_idx += self._microbatch_to_minibatch
            unsent_tasks = self._next_task_idx - self._send_idx
            self._queue_tasks_counter = self._next_queue_tasks_counter
            waiting_tasks = self._queue_tasks_counter if unsent_tasks < 0 else unsent_tasks+self._queue_tasks_counter
            self._task_batch += 1
            self._gc_helper.Report_task(self._task_batch, waiting_tasks)
            # if __debug__:
            #     log.debug(f"GPU: {self._rank} worker_id: {worker_id} Next wm view:{self._index_waiting_queue[worker_id]} queue_task {self._queue_tasks_counter}, unsent tasks {unsent_tasks}")
            self._index_waiting_queue = self._next_index_waiting_queue
            self._next_index_waiting_queue = [[] for _ in range(self._max_workers)]
            self._next_queue_tasks_counter = 0
        else:
            self._gc_helper.Report_task(self._task_batch, waiting_tasks)
        
    def _put_stray_task(self, task):
        if isinstance(task,worker._ResumeIteration):
            return
        task_idx = task[0]
        stray_idx = bisect_left(self._stray_task_key, task_idx)
        self._stray_task_key.insert(stray_idx, task_idx)
        self._stray_task_buffer.insert(stray_idx, task)
        
    def _worker_increase(self, cpus):
        valid_cpu_ids = []
        # Add index queue directly when increase workers only
        for cpu_id in cpus:  # find the next active worker, if any
            # Translate cpu id to worker id
            worker_id = cpu_id
            if self._workers_control_events[worker_id].is_set():
                continue
            
            self._workers_control_events[worker_id].set()
            valid_cpu_ids.append(cpu_id)
            if self._index_queues[worker_id].qsize() < self._prefetch_factor:
                if len(self._stray_task_buffer):
                    idx, index = self._stray_task_buffer.pop(0)
                    self._stray_task_key.pop(0)
                    self._index_queues[worker_id].put((idx, index))
                    self._task_info[idx] = (worker_id,)
                    if __debug__:
                        log.debug(f"GPU: {self._rank} Revive ({idx}, {index}) from stray buffer to worker_queue_idx{worker_id}")
                elif self._send_idx < self._next_task_idx:
                    try:
                        index = self._next_index()
                    except StopIteration:
                        self._intentional_stop = False
                        if __debug__:
                            log.debug(f"GPU: {self._rank} Worker Control increase detect stop iteration! rcvd:{self._rcvd_idx} send:{self._send_idx}")
                        continue

                    self._index_queues[worker_id].put((self._send_idx, index))
                    if self._send_idx < self._next_task_idx:
                        self._index_waiting_queue.append(self._send_idx)
                        self._queue_tasks_counter += 1
                    else:
                        self._next_index_waiting_queue.append(self._send_idx)
                        self._next_queue_tasks_counter += 1
                    if __debug__:
                        log.debug(f"GPU: {self._rank} PUT ({self._send_idx}, {index}) to worker_queue_idx{worker_id}")
                    self._task_info[self._send_idx] = (worker_id,)
                    self._send_idx += 1
                elif not self._gc_helper.Get_Job_done_flag():
                    #  Scavenging existing workers
                    for target_worker_id in self._worker_info:
                        if self._worker_progress_info[target_worker_id][1] < self._progress_threshold and self._index_queues[target_worker_id].qsize() > 0:
                            try:
                                data = self._index_queues[target_worker_id].get(timeout=WM_QUEUE_INTERVAL)
                            except queue.Empty:
                                if __debug__:
                                    log.debug(f"GPU: {self._rank} Worker Scavenging: Queue empty")
                                continue
                            
                            if isinstance(data,worker._ResumeIteration):
                                self._index_queues[target_worker_id].put(data)
                                continue
                            else:
                                self._index_queues[worker_id].put(data)
                                if __debug__:
                                    log.debug(f"GPU: {self._rank} Scavenging Worker{worker_id} Grep {data} from Worker{target_worker_id}")
                                idx, _ = data
                                self._task_info[idx] = (worker_id,)
                            break
        
        # Update new workers
        self._worker_info.extend(valid_cpu_ids)
                    
        # Check stopped intentionally
        if len(self._worker_info) > 0 and self._intentional_stop:
            self._intentional_stop = False  
            if __debug__:
                log.debug(f"GPU: {self._rank} Clear(set) intentional stop")
        if __debug__:
            log.debug(f"GPU: {self._rank} Increase worker = {len(self._worker_info)}, worker_info: {self._worker_info} cpus: {cpus}, valid: {valid_cpu_ids}")

    def _worker_decrease(self, cpus, timeout=WM_QUEUE_INTERVAL):
        valid_cpu_ids = []
        # Decrease workers and reclaim tasks
        remaining_tasks = []
        for cpu_id in cpus:
            # Translate cpu id to worker id
            worker_id = cpu_id
            if not self._workers_control_events[worker_id].is_set():
                continue
            
            self._workers_control_events[worker_id].clear()
            try:
                self._worker_info.remove(worker_id)
            except ValueError:
                log.debug(f"GPU: {self._rank} Dont have worker {worker_id} in worker_info: {self._worker_info} decrease: Queue empty")
            valid_cpu_ids.append(cpu_id)
            
            # Reclaim tasks
            while self._index_queues[worker_id].qsize() > 0:
                try:
                    data = self._index_queues[worker_id].get(timeout=timeout)
                except queue.Empty:
                    if __debug__:
                        log.debug(f"GPU: {self._rank} Worker decrease: Queue empty")
                    break
                if isinstance(data,worker._ResumeIteration):
                    self._index_queues[worker_id].put(data)
                else:
                    remaining_tasks.append(data)
                    
        for task in remaining_tasks:
            self._put_stray_task(task)
            self._task_info[task[0]] = (-1,)
            self._queue_tasks_counter -= 1
            
        if __debug__:
            log.debug(f"GPU: {self._rank} Put idx {remaining_tasks} to stray buffer")
                
        if __debug__:
            log.debug(f"GPU: {self._rank} stray keys {self._stray_task_key}")

        # Check intentional stop
        if len(self._worker_info) < 1 and not self._intentional_stop:
            self._intentional_stop = True
            if __debug__:
                log.debug(f"GPU: {self._rank} Raise(clear) intentional stop event")
        
        # Remove waiting queue tasks from return cpus
        for return_cpu_id in valid_cpu_ids:
            self._index_waiting_queue[return_cpu_id] = []
        
        if __debug__:
            log.debug(f"GPU: {self._rank} Decrease worker = {len(self._worker_info)}, worker_info: {self._worker_info}, cpus: {cpus}, valid: {valid_cpu_ids}")
    
    # FIXME: Will not use anymore
    def _worker_reset(self):
        if __debug__:
            log.debug(f"GPU: {self._rank} Reset workers, Decrease {self._except_origin_workers}, Increase {self._origin_workers}")
        self._worker_increase(self._origin_workers)
        self._worker_decrease(self._except_origin_workers)
    
    def _worker_control(self):
        feedback = self._gc_helper.Recv()
        if feedback is None:
            return
        
        order = feedback[0]
        
        # Process order
        # Need to filter INVAILD CPU IDs
        if order == gc.STALE_ORDER:
            return
        else:
            cpu_ids = feedback[1:]
            cpu_ids = cpu_ids[cpu_ids != gc.INVALID_CPU_ID]
            if __debug__:
                log.debug(f"GPU: {self._rank} Recv [{order}, {cpu_ids}]")
            if order == gc.INCREASE_ORDER:
                self._worker_increase(cpus=cpu_ids)
                self._check_queue_put_index()
            elif order == gc.DECREASE_ORDER:
                self._worker_decrease(cpus=cpu_ids)
            else:
                if __debug__:
                    log.debug(f"GPU: {self._rank} unexpected order {order}")
    
    def _try_get_data(self, timeout=MP_STATUS_CHECK_INTERVAL):
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
                log.debug(f"GPU: {self._rank} Controller wait for data queue")
            data = self._data_queue.get(timeout=timeout)

            # if __debug__:
            #     log.debug(f"GET data object at {hex(id(data))} data size: {sys.getsizeof(data)} _data_queue:\n{self._data_queue}")
            return (True, data)
        except Exception as e:
            # At timeout and error, we manually check whether any worker has
            # failed. Note that this is the only mechanism for Windows to detect
            # worker failures.
            # failed_workers = []
            # for worker_id, w in enumerate(self._workers):
            #     if self._workers_status[worker_id].value and not w.is_alive():
            #         failed_workers.append(w)
            #         self._mark_worker_as_unavailable(worker_id)
            # if len(failed_workers) > 0:
            #     pids_str = ', '.join(str(w.pid) for w in failed_workers)
            #     raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str)) from e
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

    def _get_data(self):
        # Fetches data from `self._data_queue`.
        #
        # We check workers' status every `MP_STATUS_CHECK_INTERVAL` seconds,
        # which we achieve by running `self._try_get_data(timeout=MP_STATUS_CHECK_INTERVAL)`
        # in a loop. This is the only mechanism to detect worker failures for
        # Windows. For other platforms, a SIGCHLD handler is also used for
        # worker failure detection.
        #
        # If `pin_memory=False`, we also need check if `pin_memory_thread` had
        # died at timeouts.
        if self._timeout > 0:
            success, data = self._try_get_data(self._timeout)
            if success:
                return data
            else:
                raise RuntimeError('DataLoader timed out after {} seconds'.format(self._timeout))
        else:
            while True:
                success, data = self._try_get_data()
                if success:
                    return data
    
    def _process_data(self, complete_data, data):
        self._rcvd_idx += 1
        
        if isinstance(data, ExceptionWrapper):
            data.reraise()
        next_worker_batch = self._worker_batch_counter + self._microbatch_size
        if __debug__:
            log.debug(f"GPU: {self._rank} Controller Process worker_batch_counter: {self._worker_batch_counter}, next_worker_batch: {next_worker_batch}, _worker_batch_target_counter {self._worker_batch_target_counter}")
        complete_data[0][self._worker_batch_counter:next_worker_batch] = data[0]
        complete_data[1][self._worker_batch_counter:next_worker_batch] = data[1]
        self._worker_batch_counter = next_worker_batch
    
    def _try_put_controller_queue(self):
        while not self._worker_controller_stop_event.is_set():
            try:
                self._controller_queue.put(self.complete_data)
                # self._minibatch_done_event.set()
            except queue.Full: 
                continue
            break

    def _stop_iteration(self):
        while not self._worker_controller_stop_event.is_set():
            try:
                self._controller_queue.put(None)
                # self._minibatch_done_event.set()
            except queue.Full: 
                continue
            break
        # self._iteration_end_status.value = True

    def _finish_data(self):
        self._try_put_controller_queue()
        # Notify done
        self._complete_batch_counter += 1
        
        # Update for next minibatch metadata
        self._gc_helper.Send(current_batch=self._complete_batch_counter, 
                                micro_batch_num=self._progress_worker_batch_counter)
        
        # Initial for new mini-batch
        self._worker_batch_counter = 0
        if self._next_progress_worker_batch_counter > self._worker_batch_target_counter:
            self._progress_worker_batch_counter = self._worker_batch_target_counter
            self._next_progress_worker_batch_counter -= self._worker_batch_target_counter
        else:
            self._progress_worker_batch_counter = self._next_progress_worker_batch_counter
            self._next_progress_worker_batch_counter = 0

        # Send progress info to global manager
        self._gc_helper.Send(current_batch=self._complete_batch_counter, 
                                micro_batch_num=self._progress_worker_batch_counter)
        self.complete_data = deepcopy(self._aggr_template)
        self._gc_helper.Job_start()


    def _aggr_data(self):
        if self._intentional_stop:
            return WM_CONTINUE_ITERATION_SIGNAL
        
        # FIXME: Fix for _IterableDataset such as "del self._task_info"
        while self._rcvd_idx < self._send_idx:
            try:
                info = self._task_info[self._rcvd_idx]
            except KeyError:
                if __debug__:
                    log.debug(f"GPU: {self._rank} Controller fallback try to check task queue to test")
                info = self._task_info[self._rcvd_idx]
            worker_id = info[0]
            if __debug__:
                log.debug(f"GPU: {self._rank} Controller test idx {self._rcvd_idx}, info len: {len(info)} worker_id: {worker_id}, worker_status: {self._workers_status[worker_id].value}")
            if len(info) == 2 or worker_id == -1 or self._workers_status[worker_id].value or self._data_queue.qsize() > 0:  # has data or on stray or is still active
                break
            
            del self._task_info[self._rcvd_idx]
            if __debug__:
                log.debug(f"GPU: {self._rank} Controller delete idx {self._rcvd_idx}, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}, worker_id: {worker_id}")
            self._rcvd_idx += 1
        else:
            # Handle for remaining data
            if __debug__:
                log.debug(f"GPU: {self._rank} Controller Stop Iteration, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}")
            # no valid `self._rcvd_idx` is found (i.e., didn't break)
            # TODO: Turn FLAG
            return WM_STOP_ITERATION_SIGNAL
        
        if len(self._task_info[self._rcvd_idx]) == 2:
            data = self._task_info.pop(self._rcvd_idx)[1]
            if __debug__:
                start=perf_counter()
            self._process_data(self.complete_data, data)
            
            del data
            
            if self._worker_batch_counter >= self._worker_batch_target_counter:
                self._finish_data()
                if __debug__:
                    end=perf_counter()
                    log.debug(f"GPU: {self._rank} Controller Complete data: {end-start} idx {self._rcvd_idx}, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}")
                    
                return WM_CONTINUE_ITERATION_SIGNAL
            if __debug__:
                end=perf_counter()
                log.debug(f"GPU: {self._rank} Controller Process Buffer data: {end-start} idx {self._rcvd_idx-1}, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}")

            # self._worker_control()
            # self._check_queue_put_index()
            return WM_CONTINUE_ITERATION_SIGNAL

        if self._data_queue.qsize() < 1:
            return WM_CONTINUE_ITERATION_SIGNAL

        if __debug__:
            start=perf_counter()
        idx, data = self._get_data()
        if __debug__:
            end=perf_counter()
            log.debug(f"GPU: {self._rank} Controller Get data: {end-start} idx {idx}, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}")

        if __debug__:
            start=perf_counter()
        self._tasks_outstanding -= 1
        
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
                    log.debug(f"GPU: {self._rank} Controller fallback try to check task queue to buffer")
                self._task_info[idx] = (-2,data)
                self._send_idx += 1
                self._tasks_outstanding += 1
            if __debug__:
                end=perf_counter()
                log.debug(f"GPU: {self._rank} Controller Buffer data: {end-start} idx {idx}, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}")
        else:
            self._process_data(self.complete_data, data)
            
            del data
                
            if self._worker_batch_counter >= self._worker_batch_target_counter:
                self._finish_data()
                if __debug__:
                    end=perf_counter()
                    log.debug(f"GPU: {self._rank} Controller Complete data: {end-start} idx {idx}, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}")

            if __debug__:
                end=perf_counter()
                log.debug(f"GPU: {self._rank} Controller Processing data: {end-start} idx {idx}, rcvd_idx: {self._rcvd_idx} send_idx: {self._send_idx}")
                
        self._gc_helper.Send(current_batch=self._complete_batch_counter, 
                                micro_batch_num=self._progress_worker_batch_counter)
        return WM_CONTINUE_ITERATION_SIGNAL
    
    def _shm_cleanup(self):
        if __debug__:
            log.debug(f"GPU{self._rank}: Cleanup")
        try:
            for shm_buf in self._internal_shm_metas:
                shm_buf.close()
        except Exception as e:
            log.warn(e)

    def __del__(self):
        self._shm_cleanup() 
    
# if self._sampler_iter is None:
#     self._reset()

def _control_loop(rank, index_queues, index_sampler, worker_control_events, prefetch_factor, 
                      worker_controller_stop_event, origin_workers, _except_origin_workers, max_workers, cur_cpus, worker_queue_idx_cycle,
                      microbatch_to_minibatch, persistent_workers, _num_workers, dataloader_processes, _iteration_end_status, microbatch_size,
                      worker_status, _aggr_template, _data_queue, _minibatch_done_event, _worker_batch_target_counter, _controller_queue):
    pos = rank%len(dataloader_processes)
    cpu_id = dataloader_processes[pos]

    p = psutil.Process()
    p.cpu_affinity([cpu_id])
    
    wm = workerManager(_rank=rank, _workers_control_events=worker_control_events, _max_workers=max_workers, _cur_cpus=cur_cpus,
                       _worker_queue_idx_cycle=worker_queue_idx_cycle, _index_queues=index_queues, _prefetch_factor=prefetch_factor, 
                       _except_origin_workers=_except_origin_workers, _origin_workers=origin_workers, _microbatch_to_minibatch=microbatch_to_minibatch, 
                       _index_sampler=index_sampler, _num_workers=_num_workers, _worker_batch_target_counter=_worker_batch_target_counter,
                       _iteration_end_status=_iteration_end_status, microbatch_size=microbatch_size, _minibatch_done_event=_minibatch_done_event,
                       worker_status=worker_status, _aggr_template=_aggr_template, _data_queue=_data_queue, _controller_queue=_controller_queue,
                       _worker_controller_stop_event=worker_controller_stop_event)
    try:
        while not worker_controller_stop_event.is_set():
            wm._reset()
            while not worker_controller_stop_event.is_set():
                wm._worker_control()
                wm._check_queue_put_index()
                out = wm._aggr_data()
                if out != WM_CONTINUE_ITERATION_SIGNAL or worker_controller_stop_event.is_set():
                    break
                sleep(WM_INTERVAL)
            wm._stop_iteration()
            if not persistent_workers or worker_controller_stop_event.is_set():
                break

            
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass
    
    if worker_controller_stop_event.is_set():
        for q in index_queues:
            q.cancel_join_thread()
            # if __debug__:
            #     log.debug(f"GPU: {rank} worker_id: {worker_id}, cancel_join_thread data_queue:\n{data_queue}")
            q.close()
            