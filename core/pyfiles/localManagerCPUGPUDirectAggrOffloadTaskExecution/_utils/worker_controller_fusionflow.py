r""""Contains definitions of the methods used by the _BaseDataLoaderIter to control and put task index to workers
"""

from . import operation_offloading as CO

import torch
# from torch._six import queue, container_abcs, string_classes
# PyTorch 2.0 Fix
import queue
import collections.abc as container_abcs
string_classes = (str, bytes)


from . import worker
from . import dali
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
from vit_pytorch import ViT
import psutil
CPU_NUM = psutil.cpu_count(logical = False)
import numpy as np
from . import get_image_size

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
WM_GPU_WORKER = -960
# THREHOLD_FLAG = True

class workerManager():
    #add argument: dataset (exsist in CPUpolicyGPU) 
    def __init__(self,_workers_control_events,_max_workers,_worker_queue_idx_cycle,
                    _rank, _index_queues, _prefetch_factor, _cur_cpus, _num_workers,
                    _index_sampler,_microbatch_to_minibatch, _intentional_stop_event,
                    task_queue, _except_origin_workers, _origin_workers, _local_manager_status,
                    microbatch_size, worker_status, dataset, world_size, gpu_task_queue,
                    single_sample) -> None:
        self._rank = _rank
        self._gc_helper = gc.globalManagerHelper(rank = self._rank)
        self.cur_cpus = _cur_cpus
        self._max_workers = _max_workers
        self._workers_control_events = _workers_control_events
        self._worker_queue_idx_cycle = _worker_queue_idx_cycle
        self._index_queues = _index_queues
        self.task_queue = task_queue
        self.dataset=dataset
        self.world_size = world_size
        self._gpu_task_queue = gpu_task_queue
        self._index_sampler = _index_sampler
        self._stray_task_key = []
        self._microbatch_to_minibatch = _microbatch_to_minibatch + 1
        self._num_workers = _num_workers
        self._intentional_stop_event = _intentional_stop_event
        self._prefetch_factor = 2 # _prefetch_factor
        self._gpu_prefetch_factor = 2
        self._send_idx  = 0
        self._rcvd_idx = 0
        self._profile_idx = 0
        self._progress_idx = None
        self._sampler_iter = None
        self._task_info = None
        self._task_batch = None
        self._next_task_idx = None
        self._mark_to_return = []
        self.dummy_flag = False
        self._single_sample = single_sample
        self._internal_shm_metas = []
        self._worker_progress_info = []
        self._microbatch_size = microbatch_size
        self._progress_threshold = self._microbatch_size # // 2
        self._return_threshold = self._microbatch_size - 2
        self._worker_info = _origin_workers       
        self._workers_status = worker_status
        self._queue_tasks_counter = None
        self._sorted_task_prefetch_factor = self._microbatch_to_minibatch # // 2 # including GPU workers
        self._sorted_task_queue = None
        self._sorted_task_size = None
        self._next_sorted_task_queue = None
        self._next_sorted_task_size = None
        self._THREHOLD_FLAG = True
        self._prev_increase = 0
        self._increase = 0
        
        for worker_id in range(self._max_workers+1):
            _worker_progress = self._load_np_arr("worker_progress_info", worker_id=worker_id, frame=np.array([-1, -1], dtype=gc.INFO_DATA_TYPE), dtype=gc.INFO_DATA_TYPE)
            self._worker_progress_info.append(_worker_progress)
        
        # FIXME: Will not use anymore
        self._except_origin_workers = _except_origin_workers
        self._origin_workers = deepcopy(self._worker_info)
        self._stray_task_buffer = []
        self._local_manager_status = _local_manager_status
        
        # self._task_threshold = 2
        #################Task Execution#################
        self._task_threshold = 5
        self._prev_task_threshold = 5
        ###############################################
    def _load_np_arr(self, type, worker_id, frame, dtype=np.uint):
        shm_name = f'{type}{self._rank}_{int(worker_id)}'
        shape = frame.shape
        size = frame.nbytes
        # if __debug__:
        #     log.debug(f"GPU: {self._rank} Name: {shm_name} size:{size}")
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
        self._profile_idx = 0 # idx of the next task to be profiled for size
        self._sorted_task_queue = []
        self._sorted_task_size = []
        self._next_sorted_task_queue = []
        self._next_sorted_task_size = []

        self._progress_idx = self._num_workers
        self._next_task_idx = self._microbatch_to_minibatch
        self._index_waiting_queue = [[] for _ in range(self._max_workers+1)]
        self._next_index_waiting_queue = [[] for _ in range(self._max_workers+1)]
        self._queue_tasks_counter = 0
        self._next_queue_tasks_counter = 0
        self.dummy_flag = False
        # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)
        self._task_batch = 0
        self._gc_helper.Report_task(batch_num=self._task_batch, tasks=self._microbatch_to_minibatch-self._progress_idx)
            
        self._task_info = {}
        if __debug__:
            log.debug(f"GPU: {self._rank} Reset worker manager")

    def _next_size_index(self, device):
        # print("This much I have!", len(self._sorted_task_queue), flush=True)
        if len(self._sorted_task_queue):
            if device == "cpu":
                size_temp = self._sorted_task_size.pop(0)
                out = self._sorted_task_queue.pop(0)
                # print(f'CPU: {size_temp}, task: {out}')
            elif device == "gpu":
                size_temp = self._sorted_task_size.pop()
                out = self._sorted_task_queue.pop()
                # print(f'GPU: {size_temp}, task: {out}')
            else:
                raise RuntimeError(f"Device {device} is not supported.")
        elif len(self._next_sorted_task_queue):
            if device == "cpu":
                size_temp = self._next_sorted_task_size.pop(0)
                out = self._next_sorted_task_queue.pop(0)
                # print(f'CPU: {size_temp}, task: {out}')
            elif device == "gpu":
                size_temp = self._next_sorted_task_size.pop()
                out = self._next_sorted_task_queue.pop()
                # print(f'GPU: {size_temp}, task: {out}')
            else:
                raise RuntimeError(f"Device {device} is not supported.")
        else:
            return None
            # if __debug__:
            #     log.debug(f"GPU: {self._rank} means stop Iteration")
            raise StopIteration
        return out 
    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration
    
    def _inspect_offload_workload(self):
        if len(self._next_sorted_task_queue) > self._sorted_task_prefetch_factor:
            return WM_CONTINUE_ITERATION_SIGNAL
        
        for _ in range(self._microbatch_to_minibatch):
            if __debug__:
                log.debug(f"GPU: {self._rank} cur_sorted_len {len(self._sorted_task_queue)}, task_prefetch {self._sorted_task_prefetch_factor}, next_sorted_len {len(self._next_sorted_task_queue)}")
            
            try:
                index = self._next_index() 
                size = self._get_image_size(self._profile_idx)
            except StopIteration:
                # if __debug__:
                #     log.debug(f"GPU: {self._rank} Worker Control increase detect stop iteration! rcvd:{self._rcvd_idx} send:{self._send_idx}")
                # self._intentional_stop_event.set()
                return WM_STOP_ITERATION_SIGNAL
            self._put_waiting_task(index, size)
            self._profile_idx += 1 
            if len(self._sorted_task_queue) > self._sorted_task_prefetch_factor:
                break
        # print(self._sorted_task_size)
        # print(f'check: {self._sorted_task_queue}')
        return WM_CONTINUE_ITERATION_SIGNAL

    def _get_image_size(self, index):
        # if __debug__:
        #     log.debug(f"GPU: {self._rank} cur_sorted_len {self._index_sampler.filesize_meta}")
        if hasattr(self._index_sampler, "filesize_meta"):
            size = self._index_sampler.filesize_meta[index]
        else:
            size = None
        return size
        
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
                    # if __debug__:
                    #     log.debug(f"GPU: {self._rank} Put idx {idx} to stray buffer")
                return

            self._index_queues[worker_queue_idx].put((CO.start_op(), task))
            self._task_info[idx] = (worker_queue_idx,)
            # if __debug__:
            #     log.debug(f"GPU: {self._rank} RePUT ({idx}, {index}) to worker_queue_idx{worker_queue_idx}")
    
    def _GPU_try_put_index(self):
        K = 1
        num_microbatches = 80 // self._microbatch_size
        num_times = num_microbatches // K
        # num_microbatches = 160 // self._microbatch_size
        # # Threshold 192, Large Batch is 16 Images
        # if unsent_tasks >= 1:   # and self._send_idx % self._microbatch_to_minibatch <= 160 // self._microbatch_size:
        #     num_microbatches = min(160 // self._microbatch_size, unsent_tasks + 16)
        while num_microbatches > 0:
            microbatches_to_send = ([], [])
            for _ in range(K):
                if num_microbatches <= 0:
                    break
                if self._gpu_task_queue.qsize() < self._gpu_prefetch_factor or True:
                    try:
                        index = self._next_size_index("gpu")
                        # print("Putting", index, "into gpu queue", flush=True)
                        if index is None:
                            break
                        
                        microbatches_to_send[0].append(self._send_idx)
                        microbatches_to_send[1].extend(index)

                        self._task_info[self._send_idx] = (WM_GPU_WORKER,)
                        self._index_waiting_queue[-1].append(self._send_idx)
                        self._queue_tasks_counter += 1
                        self._send_idx += 1
                        num_microbatches -= 1
                    except StopIteration:
                        self._intentional_stop_event.set()
                        return WM_STOP_ITERATION_SIGNAL
                else:
                    break
            # Put them 
            if microbatches_to_send[0]:
                self._gpu_task_queue.put((CO.start_op(), microbatches_to_send))      
        #########    
        return WM_CONTINUE_ITERATION_SIGNAL    
    
    def _try_put_index(self):
        # assert _tasks_outstanding < self._prefetch_factor * self._num_workers
        try:
            index = self._next_size_index("cpu")
        except StopIteration:
            self._intentional_stop_event.set()
            return
        for _ in range(self._max_workers):  # find the next active worker, if any
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_control_events[worker_queue_idx].is_set():
                break
        else:
            # if __debug__:
            #     log.debug(f"GPU: {self._rank} Failed to find the active worker")
            # not found (i.e., didn't break)
            return

        self._index_queues[worker_queue_idx].put((CO.start_op(), (self._send_idx, index)))
        self._task_info[self._send_idx] = (worker_queue_idx,)
        # if __debug__:
        #     log.debug(f"GPU: {self._rank} PUT ({self._send_idx}, {index}) to worker_queue_idx{worker_queue_idx} task_info_idx{self._send_idx}: {self._task_info[self._send_idx]}")
        self._send_idx += 1
        # if __debug__:
        #     log.debug(f"tasks_outstanding up to {_tasks_outstanding}")
        
    def _cleanup_return_workers(self):
        for worker_id in self._mark_to_return:
            # if __debug__:
            #     log.debug(f"GPU: {self._rank} mark_to_return {self._mark_to_return} worker{worker_id} progress{self._worker_progress_info[worker_id]} status{self._workers_status[worker_id].value}")
            if self._worker_progress_info[worker_id][0] == -1 or self._worker_progress_info[worker_id][1] >= self._return_threshold or not self._workers_status[worker_id].value:
                self._gc_helper.Report_return(worker_id)
                try:
                    self._worker_info.remove(worker_id)
                except ValueError:
                    # if __debug__:
                    #     log.debug(f"GPU: {self._rank} Remove worker{worker_id} from worker_info{self._worker_info}")
                    pass

    def _responsive_put_index(self):
        index = None
        run = True
        threshold = 240 // 4
        large_batch = 8 // 4
        while run:
            # Put into GPU
            microbatches_to_send = ([], [])
            num_minibatches = 1
            if threshold > 0 and self._gpu_task_queue.qsize() < self._gpu_prefetch_factor:
                num_minibatches = large_batch
                threshold -= 1
            
            for _ in range(num_minibatches):
                if self._gpu_task_queue.qsize() < self._gpu_prefetch_factor:
                    try:
                        index = self._next_size_index("gpu")
                        if index is None:
                            run = False
                            break
                        microbatches_to_send[0].append(self._send_idx)
                        microbatches_to_send[1].extend(index)
                        self._task_info[self._send_idx] = (WM_GPU_WORKER,)
                        self._index_waiting_queue[-1].append(self._send_idx)
                        self._queue_tasks_counter += 1
                        self._send_idx += 1
                    except StopIteration:
                        self._intentional_stop_event.set()
                        return WM_STOP_ITERATION_SIGNAL
            if microbatches_to_send[0]:
                self._gpu_task_queue.put((CO.start_op(), microbatches_to_send)) 
            # Put into CPU
            for worker_queue_idx in self._origin_workers:
                if self._index_queues[worker_queue_idx].qsize() < self._prefetch_factor:
                    index = self._next_size_index("cpu")
                    if index is None:
                        run = False
                        break
                    self._index_queues[worker_queue_idx].put((CO.start_op(), (self._send_idx, index)))
                    self._task_info[self._send_idx] = (worker_queue_idx,)
                    self._send_idx += 1
                    index = None
            if run:
                sleep(WM_INTERVAL)
        return WM_CONTINUE_ITERATION_SIGNAL
    
    def _check_queue_put_index(self, cnt=100000000):
        index = None
        while cnt > 0:
            for worker_queue_idx in self._origin_workers:
                if index is None:
                    index = self._next_size_index("cpu")
                if index is None:
                    cnt = 0
                    break 
                if True or self._index_queues[worker_queue_idx].qsize() < self._prefetch_factor:
                    self._index_queues[worker_queue_idx].put((CO.start_op(), (self._send_idx, index)))
                    self._task_info[self._send_idx] = (worker_queue_idx,)
                    self._send_idx += 1
                    cnt -= 1
                    index = None
                if cnt <= 0:
                    break
            # Logical Balancing!
            # for worker_queue_idx in self._origin_workers[::-1]:
            #     if index is None:
            #         index = self._next_size_index("cpu")
            #     if index is None:
            #         cnt = 0
            #         break 
            #     if True or self._index_queues[worker_queue_idx].qsize() < self._prefetch_factor:
            #         self._index_queues[worker_queue_idx].put((CO.start_op(), (self._send_idx, index)))
            #         self._task_info[self._send_idx] = (worker_queue_idx,)
            #         self._send_idx += 1
            #         cnt -= 1
            #         index = None
            #     if cnt <= 0:
            #         break
        return WM_CONTINUE_ITERATION_SIGNAL

            
    # def _update_worker_progress_info(self,worker_id):
    #     # self._next_task_idx = minibatch_to_microbatch = 60
    #     # self._send_idx = how many tiny-batch have been sent
    #     unsent_tasks = self._next_task_idx - self._send_idx
    #     idx = int(self._worker_progress_info[worker_id][0])
        
    #     # if self._rank == 0:            
    #     #     print(f"[_update_worker_profress_info]: before GPU {self._rank} worker_id: {worker_id} _worker_progress_info = {self._worker_progress_info[worker_id]} wm view:{self._index_waiting_queue[worker_id]} queue_task_counter {self._queue_tasks_counter}, unsent tasks {unsent_tasks} ") 
    #     #### Heelim : if worker progressing the index of micro-batch in waiting queue, remove index of processing from waiting queue
    #     #### and queue_task_counter -1 
    #     if idx in self._index_waiting_queue[worker_id]:
    #         self._index_waiting_queue[worker_id].remove(idx)
    #         # self._queue_tasks_counter = # of tiny-batches that are prefetched
    #         self._queue_tasks_counter -= 1
    #     #### Heelim : if every tasks from mini-batch has sent to workers && eliminating dummy
    #     elif unsent_tasks < 1 and len(self._index_waiting_queue[worker_id]) and self._worker_progress_info[worker_id][1] >= self._progress_threshold:
    #         self._index_waiting_queue[worker_id].pop(0)
    #         self._queue_tasks_counter -= 1
                
    #     if idx in self._next_index_waiting_queue[worker_id]:
    #         self._next_index_waiting_queue[worker_id].remove(idx)
    #         self._next_queue_tasks_counter -= 1
            
    #     # self.queue_tasks_counter : tasks that are in CPU and GPU queues(prefetched)
    #     # true_tasks : unsent tasks + tasks in CPU and GPU queues
    #     true_tasks = unsent_tasks+self._queue_tasks_counter
    #     # waiting tasks : if unsent tasks < 0, tasks that are prefetched to the CPU or GPU
    #     waiting_tasks = self._queue_tasks_counter if unsent_tasks < 0 else true_tasks
    #     # if __debug__:
    #     #     log.debug(f"GPU: {self._rank} worker_id: {worker_id} wm view:{self._index_waiting_queue[worker_id]} queue_task {self._queue_tasks_counter}, unsent tasks {unsent_tasks}")
    #     # if self._rank == 0:
    #     #     print(f"[_update_worker_profress_info]: after GPU {self._rank} worker_id: {worker_id} _worker_progress_info = {self._worker_progress_info[worker_id]} wm view:{self._index_waiting_queue[worker_id]} queue_task_counter {self._queue_tasks_counter}, unsent tasks {unsent_tasks}") 
    #     if waiting_tasks < 1 or true_tasks < 0:
    #         # when job is done -> RESET waiting_task(left tiny batch size) to mini-batch size
    #         self._gc_helper.Job_done()
    #         self._next_task_idx += self._microbatch_to_minibatch
    #         unsent_tasks = self._next_task_idx - self._send_idx
    #         self._queue_tasks_counter = self._next_queue_tasks_counter
    #         waiting_tasks = self._queue_tasks_counter if unsent_tasks < 0 else unsent_tasks+self._queue_tasks_counter
    #         self._task_batch += 1
    #         self._gc_helper.Report_task(self._task_batch, waiting_tasks)
    #         # if __debug__:
    #         #     log.debug(f"GPU: {self._rank} worker_id: {worker_id} Next wm view:{self._index_waiting_queue[worker_id]} queue_task {self._queue_tasks_counter}, unsent tasks {unsent_tasks}")
    #         self._index_waiting_queue = self._next_index_waiting_queue
    #         self._next_index_waiting_queue = [[] for _ in range(self._max_workers)]
    #         self._next_queue_tasks_counter = 0
            
    #         for task in self._sorted_task_queue:
    #             self._put_stray_task((self._send_idx,task))
    #             self._send_idx += 1
                
    #         self._sorted_task_queue = self._next_sorted_task_queue
    #         self._sorted_task_size = self._next_sorted_task_size
    #         self._next_sorted_task_size = []
    #         self._next_sorted_task_queue = []
    #     else:
    #         self._gc_helper.Report_task(self._task_batch, waiting_tasks)
        
    def _put_waiting_task(self, task, size):
        if isinstance(task,worker._ResumeIteration):
            return
        size_idx = bisect_left(self._sorted_task_size, size)
        self._sorted_task_size.insert(size_idx, size)
        self._sorted_task_queue.insert(size_idx, task)

    # def _put_next_waiting_task(self, task, size):
    #     if isinstance(task,worker._ResumeIteration):
    #         return
    #     size_idx = bisect_left(self._sorted_task_size, size)
    #     self._next_sorted_task_size.insert(size_idx, size)
    #     self._next_sorted_task_queue.insert(size_idx, task)
    
    # def _put_stray_task(self, task):
    #     if isinstance(task,worker._ResumeIteration):
    #         return
    #     task_idx = task[0]
    #     stray_idx = bisect_left(self._stray_task_key, task_idx)
    #     self._stray_task_key.insert(stray_idx, task_idx)
    #     self._stray_task_buffer.insert(stray_idx, task)
        
    def _worker_increase(self, cpus):
        valid_cpu_ids = []
        Reput_tasks = []
        Scavenged_tasks = []
        worker_cycle = cycle(self._worker_info)
        # Add index queue directly when increase workers only
        for cpu_id in cpus:  # find the next active worker, if any
            # Make it to available worker id
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
                    self._index_queues[worker_id].put((CO.start_op(), (idx, index)))
                    self._task_info[idx] = (worker_id,) 
                    # if __debug__:
                    #     log.debug(f"GPU: {self._rank} Revive ({idx}, {index}) from stray buffer to worker_queue_idx{worker_id}")
                elif not self._gc_helper.Get_Job_done_flag() and self._send_idx < self._next_task_idx:
                    try:
                        index = self._next_size_index("cpu")
                    except StopIteration:
                        self._intentional_stop_event.set()
                        # if __debug__:
                        #     log.debug(f"GPU: {self._rank} Worker Control increase detect stop iteration! rcvd:{self._rcvd_idx} send:{self._send_idx}")
                        continue

                    self._index_queues[worker_id].put((CO.start_op(), (self._send_idx, index)))
                    if self._send_idx < self._next_task_idx:
                        self._index_waiting_queue[worker_id].append(self._send_idx)
                        self._queue_tasks_counter += 1
                    else:
                        self._next_index_waiting_queue[worker_id].append(self._send_idx)
                        self._next_queue_tasks_counter += 1
                    # if __debug__:
                    #     log.debug(f"GPU: {self._rank} PUT ({self._send_idx}, {index}) to worker_queue_idx{worker_id}")
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
                                    # if __debug__:
                                    #     log.debug(f"GPU: {self._rank} Worker Scavenging: Queue empty")
                                    break
                                
                            while len(Reput_tasks):
                                task = Reput_tasks.pop(0)
                                self._index_queues[target_worker_id].put(task)
                            
                            if len(Scavenged_tasks):
                                try:
                                    task = Scavenged_tasks.pop(0)
                                    self._index_queues[worker_id].put(task)
                                    # if __debug__:
                                    #     log.debug(f"GPU: {self._rank} Scavenging Worker{worker_id} Grep {data} from Worker{target_worker_id}")
                                    idx, _ = data
                                    self._task_info[idx] = (worker_id,)
                                    self._index_waiting_queue[worker_id].append(data[0])
                                    break
                                except queue.Full:
                                    # if __debug__:
                                    #     log.debug(f"GPU: {self._rank} Worker Scavenging: Queue Full")
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
        #     if __debug__:
        #         log.debug(f"GPU: {self._rank} Clear(set) intentional stop")
        # if __debug__:
        #     log.debug(f"GPU: {self._rank} Increase worker = {len(self._worker_info)}, worker_info: {self._worker_info} cpus: {cpus}, valid: {valid_cpu_ids}")

    def _worker_decrease(self, cpus, timeout=WM_QUEUE_INTERVAL):
        valid_cpu_ids = []
        # Decrease workers and reclaim tasks
        remaining_tasks = []
        for cpu_id in cpus:
            # Translate cpu id to worker id
            # - self.cur_cpus[0]
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
                    # if __debug__:
                    #     log.debug(f"GPU: {self._rank} Worker decrease: Queue empty")
                    break
                if isinstance(data,worker._ResumeIteration):
                    self._index_queues[worker_id].put(data)
                else:
                    remaining_tasks.append(data)
                    self._index_waiting_queue[worker_id] = []
                    
        # if __debug__:
        #     log.debug(f"GPU: {self._rank} Put idx {remaining_tasks} to stray buffer")
                
        # if __debug__:
        #     log.debug(f"GPU: {self._rank} stray keys {self._stray_task_key}")
            
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
                # if __debug__:
                #     log.debug(f"GPU: {self._rank} Already removed {worker_id} in worker_info: {self._worker_info}")
                pass
                
            try:
                self._mark_to_return.remove(worker_id)
            except ValueError:
                # if __debug__:
                #     log.debug(f"GPU: {self._rank} No need to remove {worker_id} in mark_to_return: {self._mark_to_return}")
                pass

        # Check intentional stop
        if len(self._worker_info) < 1 and self._intentional_stop_event.is_set():
            self._intentional_stop_event.clear()
            # if __debug__:
            #     log.debug(f"GPU: {self._rank} Raise(clear) intentional stop event")
        
        # Remove waiting queue tasks from return cpus
        for return_cpu_id in valid_cpu_ids:
            self._index_waiting_queue[return_cpu_id] = []
        
        # if __debug__:
        #     log.debug(f"GPU: {self._rank} Decrease worker = {len(self._worker_info)}, worker_info: {self._worker_info}, cpus: {cpus}, valid: {valid_cpu_ids}")
    
    # NOTE: Will not use anymore
    def _worker_reset(self):
        # if __debug__:
        #     log.debug(f"GPU: {self._rank} Reset workers, Decrease {self._except_origin_workers}, Increase {self._origin_workers}")
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
            # if __debug__:
            #     log.debug(f"GPU: {self._rank} Recv [{order}, {cpu_ids}]")
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
        
        # if self._local_manager_status.value and len(self._task_info) >= len(self._worker_info):
        #     # if __debug__:
        #     #     log.debug(f"GPU: {self._rank} task_info notify")
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
        # if __debug__:
        #     log.debug(f"GPU: {self._rank} Worker Manager Sending stop task")
        self._gc_helper.Job_done()
    
    def _shm_cleanup(self):
        # if __debug__:
        #     log.debug(f"GPU{self._rank}: Cleanup")
        try:
            for shm_buf in self._internal_shm_metas:
                shm_buf.close()
        except Exception as e:
            log.warn(e)

    def __del__(self):
        self._shm_cleanup()
    
# if self._sampler_iter is None:
#     self._reset()
#add argument: dataset
def _control_loop(rank, index_queues, task_queue, index_sampler, worker_control_events, _intentional_stop_event, prefetch_factor, 
                      worker_controller_stop_event, origin_workers, _except_origin_workers, max_workers, cur_cpus, worker_queue_idx_cycle,
                      microbatch_to_minibatch, iter_start_event, persistent_workers, _num_workers, dataloader_processes, _local_manager_status, microbatch_size,
                      worker_status, dataset, world_size, gpu_task_queue, single_sample, dataloading_phase_event):
    pos = rank%len(dataloader_processes)
    cpu_id = dataloader_processes[pos]

    p = psutil.Process()
    p.cpu_affinity([cpu_id])
    wm = workerManager(_rank=rank, _workers_control_events=worker_control_events, _max_workers=max_workers, _cur_cpus=cur_cpus,
                       _worker_queue_idx_cycle=worker_queue_idx_cycle, _index_queues=index_queues, _prefetch_factor=prefetch_factor, 
                       _except_origin_workers=_except_origin_workers, _origin_workers=origin_workers, _microbatch_to_minibatch=microbatch_to_minibatch, 
                       _index_sampler=index_sampler, _intentional_stop_event=_intentional_stop_event, task_queue=task_queue, _num_workers=_num_workers,
                       _local_manager_status=_local_manager_status, microbatch_size=microbatch_size,
                       worker_status=worker_status, dataset=dataset, world_size=world_size, gpu_task_queue=gpu_task_queue,
                       single_sample=single_sample)
    try:
        while not worker_controller_stop_event.is_set():
            # iteration for GPUs -> 6 GPUs
            wm._reset()
            wm._inspect_offload_workload()
            iter_start_event.wait()

            # Do Prefetching
            prefetch_depth = 8
            for _ in range(prefetch_depth):
                # wm._GPU_try_put_index()
                # out = wm._check_queue_put_index()
                wm._responsive_put_index()
                wm._task_info_notify()
                wm._inspect_offload_workload()

            # prev_increase = 0
            # increase = 0
            while iter_start_event.is_set():
                dataloading_phase_event.wait()
                print("I am putting data!", flush=True)
                # Stage of Putting into GPU
                out = wm._responsive_put_index()
                # wm._GPU_try_put_index()

                # Stage ot Putting into CPU
                # out = wm._check_queue_put_index()
                # print("I HAVE DONE!", flush=True)

                wm._task_info_notify()
                # Prefetch Stage 
                wm._inspect_offload_workload()

                while dataloading_phase_event.is_set():
                    continue

                if out != WM_CONTINUE_ITERATION_SIGNAL or worker_controller_stop_event.is_set():
                    iter_start_event.clear()
                    break
                
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
            q.close()
            
        task_queue.cancel_join_thread()
        task_queue.close()        