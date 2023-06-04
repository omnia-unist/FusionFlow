r""""Contains definitions of the methods used by the _BaseDataLoaderIter to control and put task index to workers
"""

import torch
from torch._six import queue, container_abcs, string_classes

from . import worker
import psutil
from time import sleep
from torch._utils import ExceptionWrapper
from bisect import bisect_left
from collections import namedtuple
from itertools import cycle
from copy import deepcopy
import sys
sys.path.append("..")
from posix_ipc import O_CREAT, SharedMemory
import mmap
import globalManager as gc
from efficientnet_pytorch import EfficientNet
from vit_pytorch import ViT
import numpy as np

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
                    _index_sampler,_microbatch_to_minibatch, _intentional_stop_event,
                    task_queue, _except_origin_workers, _origin_workers, _local_manager_status,
                    microbatch_size, worker_status) -> None:
        self._rank = _rank
        self._gc_helper = gc.globalManagerHelper(rank = self._rank)
        self.cur_cpus = _cur_cpus
        self._max_workers = _max_workers
        self._workers_control_events = _workers_control_events
        self._worker_queue_idx_cycle = _worker_queue_idx_cycle
        self._index_queues = _index_queues
        self.task_queue = task_queue
        self._index_sampler = _index_sampler
        self._stray_task_key = []
        self._microbatch_to_minibatch = _microbatch_to_minibatch
        self._num_workers = _num_workers
        self._intentional_stop_event = _intentional_stop_event
        self._prefetch_factor = _prefetch_factor
        self._send_idx  = 0
        self._rcvd_idx = 0
        self._progress_idx = None
        self._sampler_iter = None
        self._task_info = None
        self._task_batch = None
        self._next_task_idx = None
        self._mark_to_return = []
        
        self._internal_shm_metas = []
        self._worker_progress_info = []
        self._microbatch_size = microbatch_size
        self._progress_threshold = self._microbatch_size // 2
        self._return_threshold = self._microbatch_size - 2
        self._worker_info = _origin_workers       
        self._workers_status = worker_status
        self._queue_tasks_counter = None
        
        for worker_id in range(self._max_workers):
            _worker_progress = self._load_np_arr("worker_progress_info", worker_id=worker_id, frame=np.array([-1, -1], dtype=gc.INFO_DATA_TYPE), dtype=gc.INFO_DATA_TYPE)
            self._worker_progress_info.append(_worker_progress)
        
        # FIXME: Will not use anymore
        self._except_origin_workers = _except_origin_workers
        self._origin_workers = deepcopy(self._worker_info)
        self._stray_task_buffer = []
        self._local_manager_status = _local_manager_status
        
    def _load_np_arr(self, type, worker_id, frame, dtype=np.uint):
        shm_name = f'{type}{self._rank}_{int(worker_id)}'
        shape = frame.shape
        size = frame.nbytes
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
        self._sampler_iter = iter(self._index_sampler)
        self._send_idx = 0  # idx of the next task to be sent to workers
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__
        self._progress_idx = self._num_workers
        self._next_task_idx = self._microbatch_to_minibatch
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
            self._intentional_stop_event.set()
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
    def _cleanup_return_workers(self):
        for worker_id in self._mark_to_return:
            if self._worker_progress_info[worker_id][0] == -1 or self._worker_progress_info[worker_id][1] >= self._return_threshold or not self._workers_status[worker_id].value:
                if __debug__:
                    log.debug(f"GPU: {self._rank} mark_to_return {self._mark_to_return} worker{worker_id} progress{self._worker_progress_info[worker_id]} status{self._workers_status[worker_id].value}")
                self._gc_helper.Report_return(worker_id)
                try:
                    self._worker_info.remove(worker_id)
                except ValueError:
                    if __debug__:
                        log.debug(f"GPU: {self._rank} Remove worker{worker_id} from worker_info{self._worker_info}")
                    pass
    
    def _check_queue_put_index(self):
        for worker_queue_idx in self._worker_info:  # find the next active worker, if any
            if worker_queue_idx in self._mark_to_return:
                self._update_worker_progress_info(worker_queue_idx)
                continue
            elif not worker_queue_idx in self._origin_workers and (self._gc_helper.Get_Job_done_flag() or self._send_idx >= self._next_task_idx):
                if not worker_queue_idx in self._mark_to_return:
                    self._mark_to_return.append(worker_queue_idx)
                # if __debug__:
                #     log.debug(f"GPU: {self._rank} stolen worker status {self._workers_status[worker_queue_idx].value}")
                self._update_worker_progress_info(worker_queue_idx)
                continue
            elif not self._workers_control_events[worker_queue_idx].is_set():
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
                        self._intentional_stop_event.set()
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
        Reput_tasks = []
        Scavenged_tasks = []
        worker_cycle = cycle(self._worker_info)
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
                elif not self._gc_helper.Get_Job_done_flag() and self._send_idx < self._next_task_idx:
                    try:
                        index = self._next_index()
                    except StopIteration:
                        self._intentional_stop_event.set()
                        if __debug__:
                            log.debug(f"GPU: {self._rank} Worker Control increase detect stop iteration! rcvd:{self._rcvd_idx} send:{self._send_idx}")
                        continue

                    self._index_queues[worker_id].put((self._send_idx, index))
                    if self._send_idx < self._next_task_idx:
                        self._index_waiting_queue[worker_id].append(self._send_idx)
                        self._queue_tasks_counter += 1
                    else:
                        self._next_index_waiting_queue[worker_id].append(self._send_idx)
                        self._next_queue_tasks_counter += 1
                    if __debug__:
                        log.debug(f"GPU: {self._rank} PUT ({self._send_idx}, {index}) to worker_queue_idx{worker_id}")
                    self._task_info[self._send_idx] = (worker_id,)
                    self._send_idx += 1
                elif not self._gc_helper.Get_Job_done_flag():
                    #  Scavenging existing workers
                    for i in range(len(self._worker_info)):
                        target_worker_id = next(worker_cycle)
                        if self._index_waiting_queue[target_worker_id] and self._worker_progress_info[target_worker_id][1] < self._progress_threshold and self._index_queues[target_worker_id].qsize() > 0:
                            if self._index_waiting_queue[target_worker_id][0] >= self._send_idx:
                                continue
                            # else:
                            #     self._gc_helper.return_info(cpu_id)
                            # continue
                            
                            while self._index_queues[target_worker_id].qsize():
                                try:
                                    data = self._index_queues[target_worker_id].get(timeout=WM_QUEUE_INTERVAL)
                                    idx = self._index_waiting_queue[target_worker_id].pop(0)
                                    
                                    if isinstance(data,worker._ResumeIteration):
                                        self._index_queues[target_worker_id].put(data)
                                        break
                                    
                                    if idx < self._next_task_idx:
                                        Scavenged_tasks.append(data)
                                        break
                                    else:
                                        Reput_tasks.append(data)
                                    
                                except queue.Empty:
                                    if __debug__:
                                        log.debug(f"GPU: {self._rank} Worker Scavenging: Queue empty")
                                    break
                                
                            while len(Reput_tasks):
                                task = Reput_tasks.pop(0)
                                self._index_queues[target_worker_id].put(task)
                            
                            if len(Scavenged_tasks):
                                try:
                                    task = Scavenged_tasks.pop(0)
                                    self._index_queues[worker_id].put(task)
                                    if __debug__:
                                        log.debug(f"GPU: {self._rank} Scavenging Worker{worker_id} Grep {data} from Worker{target_worker_id}")
                                    idx, _ = data
                                    self._task_info[idx] = (worker_id,)
                                    self._index_waiting_queue[worker_id].append(data[0])
                                    break
                                except queue.Full:
                                    if __debug__:
                                        log.debug(f"GPU: {self._rank} Worker Scavenging: Queue Full")
                                    Scavenged_tasks.append(task)
                                    break
                        else:
                            self._gc_helper.Report_return(cpu_id)
                self._update_worker_progress_info(worker_id)
                    # for target_worker_id in self._worker_info:
                    #     if self._index_waiting_queue[target_worker_id] and self._worker_progress_info[target_worker_id][1] < self._progress_threshold and self._index_queues[target_worker_id].qsize() > 0:
                    #         if self._index_waiting_queue[target_worker_id][0] >= self._send_idx:
                    #             continue
                                
                    #         try:
                    #             data = self._index_queues[target_worker_id].get(timeout=WM_QUEUE_INTERVAL)
                    #             self._index_waiting_queue[target_worker_id].pop(0)
                    #         except queue.Empty:
                    #             if __debug__:
                    #                 log.debug(f"GPU: {self._rank} Worker Scavenging: Queue empty")
                    #             continue
                            
                    #         if isinstance(data,worker._ResumeIteration):
                    #             self._index_queues[target_worker_id].put(data)
                    #             continue
                    #         else:
                    #             self._index_queues[worker_id].put(data)
                    #             self._index_waiting_queue[worker_id].append(data[0])
                    #             if __debug__:
                    #                 log.debug(f"GPU: {self._rank} Scavenging Worker{worker_id} Grep {data} from Worker{target_worker_id}")
                    #             idx, _ = data
                    #             self._task_info[idx] = (worker_id,)
                    #         break
                # self._update_worker_progress_info(worker_id)
        # if not self._gc_helper.Get_Job_done_flag():
        #     for target_worker_id in self._worker_info:
                # if self._index_waiting_queue[target_worker_id] and self._worker_progress_info[target_worker_id][1] < self._progress_threshold and self._index_queues[target_worker_id].qsize() > 0:
                #     if self._index_waiting_queue[target_worker_id][0] >= self._send_idx:
                #         continue
                    
                #     while self._index_queues[target_worker_id].qsize():
                #         try:
                #             data = self._index_queues[target_worker_id].get(timeout=WM_QUEUE_INTERVAL)
                #             idx = self._index_waiting_queue[target_worker_id].pop(0)
                            
                #             if isinstance(data,worker._ResumeIteration):
                #                 self._index_queues[target_worker_id].put(data)
                #                 break
                            
                #             if idx < self._next_task_idx:
                #                 Scavenged_tasks.append(data)
                #             else:
                #                 Reput_tasks.append(data)
                            
                #         except queue.Empty:
                #             if __debug__:
                #                 log.debug(f"GPU: {self._rank} Worker Scavenging: Queue empty")
                #             break
                        
                #     while len(Reput_tasks):
                #         task = Reput_tasks.pop(0)
                #         self._index_queues[target_worker_id].put(task)
                        
            
                # while len(Scavenged_tasks):
                #     for worker_id in valid_cpu_ids:
                #     try:
                #         task = Scavenged_tasks.pop(0)
                #         self._index_queues[worker_id].put(task)
                #         if __debug__:
                #             log.debug(f"GPU: {self._rank} Scavenging Worker{worker_id} Grep {data} from Worker{target_worker_id}")
                #         idx, _ = data
                #         self._task_info[idx] = (worker_id,)
                #         self._index_waiting_queue[worker_id].append(data[0])
                #         self._index_waiting_queue[worker_id]
                #     except queue.Full:
                #         if __debug__:
                #             log.debug(f"GPU: {self._rank} Worker Scavenging: Queue Full")
                #         continue
                
                
        # Update new workers
        self._worker_info.extend(valid_cpu_ids)
                    
        # Check stopped intentionally
        if len(self._worker_info) > 0 and not self._intentional_stop_event.is_set():
            self._intentional_stop_event.set()  
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
            valid_cpu_ids.append(cpu_id)
            self._workers_control_events[worker_id].clear()
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
                    self._index_waiting_queue[worker_id] = []
                    
        if __debug__:
            log.debug(f"GPU: {self._rank} Put idx {remaining_tasks} to stray buffer")
                
        if __debug__:
            log.debug(f"GPU: {self._rank} stray keys {self._stray_task_key}")
            
        for task in remaining_tasks:
            self._put_stray_task(task)
            self._task_info[task[0]] = (-1,)
            self._queue_tasks_counter -= 1
            
        for cpu_id in valid_cpu_ids:
            worker_id = cpu_id# - self.cur_cpus[0]
            self._gc_helper.Cleanup_return(cpu_id)
            
            # This will handle in worker_queue_idx
            try:
                self._worker_info.remove(worker_id)
            except ValueError:
                if __debug__:
                    log.debug(f"GPU: {self._rank} Already removed {worker_id} in worker_info: {self._worker_info}")
                pass
                
            try:
                self._mark_to_return.remove(worker_id)
            except ValueError:
                if __debug__:
                    log.debug(f"GPU: {self._rank} No need to remove {worker_id} in mark_to_return: {self._mark_to_return}")
                pass

        # Check intentional stop
        if len(self._worker_info) < 1 and self._intentional_stop_event.is_set():
            self._intentional_stop_event.clear()
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
            elif order == gc.DECREASE_ORDER:
                self._worker_decrease(cpus=cpu_ids)
            else:
                if __debug__:
                    log.debug(f"GPU: {self._rank} unexpected order {order}")
        
    def _task_info_notify(self):
        # Reclaim and init again
        if not self._task_info:
            return
        
        if self._local_manager_status.value and len(self._task_info) >= len(self._worker_info):
            if __debug__:
                log.debug(f"GPU: {self._rank} task_info notify {self._task_info}")
            self.task_queue.put(self._task_info)
            self._task_info = {}
    
    def _task_info_remaining(self):
        if self._task_info:
            self.task_queue.put(self._task_info)
            self._task_info = {}
        
    def _stop_task_info(self):
        self._task_info_remaining()
        data = _ControllerStopSignal()
        self.task_queue.put(data)
        if __debug__:
            log.debug(f"GPU: {self._rank} Worker Manager Sending stop task")
        self._gc_helper.Job_done()
    
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

def _control_loop(rank, index_queues, task_queue, index_sampler, worker_control_events, _intentional_stop_event, prefetch_factor, 
                      worker_controller_stop_event, origin_workers, _except_origin_workers, max_workers, cur_cpus, worker_queue_idx_cycle,
                      microbatch_to_minibatch, iter_start_event, persistent_workers, _num_workers, dataloader_processes, _local_manager_status, microbatch_size,
                      worker_status):
    pos = rank%len(dataloader_processes)
    cpu_id = dataloader_processes[pos]

    p = psutil.Process()
    p.cpu_affinity([cpu_id])
    
    wm = workerManager(_rank=rank, _workers_control_events=worker_control_events, _max_workers=max_workers, _cur_cpus=cur_cpus,
                       _worker_queue_idx_cycle=worker_queue_idx_cycle, _index_queues=index_queues, _prefetch_factor=prefetch_factor, 
                       _except_origin_workers=_except_origin_workers, _origin_workers=origin_workers, _microbatch_to_minibatch=microbatch_to_minibatch, 
                       _index_sampler=index_sampler, _intentional_stop_event=_intentional_stop_event, task_queue=task_queue, _num_workers=_num_workers,
                       _local_manager_status=_local_manager_status, microbatch_size=microbatch_size,
                       worker_status=worker_status)
    try:
        while not worker_controller_stop_event.is_set():
            wm._reset()
            iter_start_event.wait()
            while iter_start_event.is_set():
                wm._worker_control()
                out = wm._check_queue_put_index()
                wm._cleanup_return_workers()
                wm._task_info_notify()
                if out != WM_CONTINUE_ITERATION_SIGNAL or worker_controller_stop_event.is_set():
                    iter_start_event.clear()
                    break
                sleep(WM_INTERVAL)
            if not persistent_workers or worker_controller_stop_event.is_set():
                break
            
            wm._stop_task_info()
            iter_start_event.wait()
            
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass
    
    if worker_controller_stop_event.is_set():
        for q in index_queues:
            q.cancel_join_thread()
            # if __debug__:
            #     log.debug(f"GPU: {rank} worker_id: {worker_id}, cancel_join_thread data_queue:\n{data_queue}")
            q.close()
            
        task_queue.cancel_join_thread()
        # if __debug__:
        #     log.debug(f"GPU: {rank} worker_id: {worker_id}, cancel_join_thread data_queue:\n{data_queue}")
        task_queue.close()