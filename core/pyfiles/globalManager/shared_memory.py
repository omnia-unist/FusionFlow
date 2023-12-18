from numpy.core.numeric import False_
import posix_ipc
from multiprocessing import Process, Value
from time import time, sleep
import numpy as np
from posix_ipc import O_CREAT, O_RDONLY, O_TRUNC, SharedMemory, unlink_shared_memory
import numa
import torch
import mmap
import psutil
from math import ceil

# CONSTANTS
# NOTE: interval unit is sec
GM_PROCESS_INTERVAL = 0.001
GM_BUSY_WAIT_INTERVAL = 0
INVALID_CPU_ID = -1
GM_CHARITY_THRESHOLD = 2.0

INCREASE_ORDER = 1
STALE_ORDER = 0
DECREASE_ORDER = -1

NUM_GPU = torch.cuda.device_count()

# FIXME: Hardcode using GPU? get from args how?
if NUM_GPU == 8:
    NUM_GPU = 6
    
AVAIL_GPU = torch.cuda.device_count()

INFO_DATA_TYPE = np.int64 # np.int32
PROGRESS_INFO_DATA_TYPE = np.int64 # np.uint
DIRTY_DATA_TYPE = np.bool_

# NOTE: numa node info
# FIXME: Not sure with uma system
NODE_MAX = numa.get_max_node()
# NODE_MAX = 2 
NUM_NODE = NODE_MAX+1
# GPU_PER_NODE = NUM_GPU // NUM_NODE
if NUM_GPU == 1:
    GPU_PER_NODE = 1
else:
    GPU_PER_NODE = NUM_GPU // (NODE_MAX+1)
ALL_CPU_IDS = [list(numa.node_to_cpus(i)) for i in range(NUM_NODE)]
# ALL_CPU_IDS: [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47], [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]]
PHYSICAL_CPU_NUM = psutil.cpu_count(logical = False)
# PHYSICAL_CPU_NUM: 32
CPU_IDS_PER_GPU = PHYSICAL_CPU_NUM  // NUM_GPU
# CPU_IDS_PER_GPU = PHYSICAL_CPU_NUM  // 6
# CPU_IDS_PER_GPU: 5

WORKER_INFO_NAME = "WORKER_INFO"
PROGRESS_INFO_NAME = "PROGRESS_INFO"

if __debug__:
    import logging
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger(__name__)


class globalManager():
    def __init__(self, num_workers, batch_size, micro_batch_size, num_gpu=NUM_GPU, max_workers=None) -> None:
        # NOTE: Flexible max gpu? needed?
        assert num_gpu == NUM_GPU
        self.num_gpu = num_gpu
        
        self.num_workers = num_workers
        # num_workers = 4
        
        # worker_info
        # {<GPU_NUM>:<CPU_IDS>}
        self.worker_info = []
        
        self.batch_size = batch_size
        # batch_size = 240
        
        self.micro_batch_size = micro_batch_size
        # micro_batch_size = 4
        
        # Prgoress info
        # {<GPU_NUM>:(<PROCESSED_BATCH_NO>,<PROCESSED_MICRO_BATCH_NO>)}
        self.progress_info = []
        
        self.feedback_buffer = [{"count": 0} for _ in range(self.num_gpu)]
        # feedback_buffer : [{'count': 0}, {'count': 0}, {'count': 0}, {'count': 0}, {'count': 0}, {'count': 0}]
        
        self.gm_view_progress_info = [[0, 0] for _ in range(self.num_gpu)]
        # self.gm_view_progress_info :  [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        
        self.job_finish_flags = []
        self.estimate_gpu_idleness_flags = []
        self.return_info = []
        self.left_tasks = []
        # FeedBack type
        # Decrease: INVALID_CPU_NUM, <CPU_ID>
        # Stall: 0, <ERROR CODE> (INVALID_CPU_NUM=No Error, -255=globalManager failed)
        # Increase: 1, <CPU_ID>
        self.feedback_info = []

        # for cleanup
        self._internal_shm_metas = []

        self.dataloader_cpus = [CPU_IDS_PER_GPU * i + self.num_workers for i in range(self.num_gpu)]
        print(self.dataloader_cpus)
        self.dataloader_cpus = [4, 9, 14, 20, 25, 30]
        if max_workers == None:
            self.max_workers = PHYSICAL_CPU_NUM
            # PHYSICAL_CPU_NUM: 32
        else:
            self.max_workers = max_workers
        self.dirties = {"progress_info": [], "feedback_info": []}

        if __debug__:
            log.debug(
                f"globalManager: num_workers: {self.num_workers}, datalader_cpu: {self.dataloader_cpus}")
            
        self.filtered_cpu_lists = [[] for i in range(len(ALL_CPU_IDS))]
        # print(self.filtered_cpu_lists)
        # print(ALL_CPU_IDS)
        # self.filtered_cpu_lists :  [[0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13], [16, 17, 18, 20, 21, 22, 23, 25, 26, 27, 28]]
        _tail_cpu_id_list = [0, 5, 10, 16, 21, 26] 

        for cur_gpu in range(self.num_gpu):
            print(cur_gpu)
            cur_node = cur_gpu // GPU_PER_NODE
            # cur_node = 0 or 1
            _tail_cpu_id = CPU_IDS_PER_GPU * (cur_gpu)
            print(_tail_cpu_id)
            # CPU_IDS_PER_GPU = 5 -> _tail_cpu_id = (5,10,15,20,25)
            _tail_cpu_id = _tail_cpu_id_list[cur_gpu]
            print(f"Modified {_tail_cpu_id}")
            gpu_worker_info = np.array([], dtype=INFO_DATA_TYPE)
            feedback_shape = (self.max_workers+1,)
            return_shape = (self.max_workers,)
            
            feedback_info_frame = np.full(
                feedback_shape, -1, dtype=INFO_DATA_TYPE)
            return_info_frame = np.full(
                return_shape, -1, dtype=INFO_DATA_TYPE)

            for cpu_id in ALL_CPU_IDS[cur_node]:
                if cpu_id == 19 or cpu_id == 24:
                    print("CPU ID missing")
                if not cpu_id in self.dataloader_cpus:
                    print(f"{_tail_cpu_id} <= {cpu_id} and {cpu_id} < {self.num_workers} + {_tail_cpu_id} => {_tail_cpu_id <= cpu_id and cpu_id < self.num_workers + _tail_cpu_id}")
                    if _tail_cpu_id <= cpu_id and cpu_id < self.num_workers + _tail_cpu_id:
                        gpu_worker_info = np.append(gpu_worker_info, cpu_id)
            self.filtered_cpu_lists[cur_node].extend(gpu_worker_info)
            print("###### cur_node: ", cur_node)
            print("###### filtered_cpu_lists: ", self.filtered_cpu_lists[cur_node])
            self.worker_info.append(gpu_worker_info)
            # self._save_np_arr_to_shm(
            #     type="worker_info", gpu_num=cur_gpu, shm_arrs=self.worker_info, np_arr=gpu_worker_info)
            self._save_np_arr_to_shm(
                type="feedback_info", gpu_num=cur_gpu, shm_arrs=self.feedback_info, np_arr=np.array(feedback_info_frame, dtype=INFO_DATA_TYPE))
            self._save_np_arr_to_shm(
                type="return_info", gpu_num=cur_gpu, shm_arrs=self.return_info, np_arr=np.array(return_info_frame, dtype=INFO_DATA_TYPE))
            self._save_np_arr_to_shm(
                type="left_tasks", gpu_num=cur_gpu, shm_arrs=self.left_tasks, np_arr=np.array([0, 0], dtype=PROGRESS_INFO_DATA_TYPE), dtype=PROGRESS_INFO_DATA_TYPE)
            self._save_bool_to_shm(
                type="feedback_info_dirty", gpu_num=cur_gpu, shm_arrs=self.dirties["feedback_info"])
            self._save_np_arr_to_shm(
                type="progress_info", gpu_num=cur_gpu, shm_arrs=self.progress_info, np_arr=np.array([0, 0], dtype=PROGRESS_INFO_DATA_TYPE), dtype=PROGRESS_INFO_DATA_TYPE)
            self._save_bool_to_shm(
                type="job_finish_flag", gpu_num=cur_gpu, shm_arrs=self.job_finish_flags)
            self._save_bool_to_shm(
                type="estimate_gpu_flag", gpu_num=cur_gpu, shm_arrs=self.estimate_gpu_idleness_flags)
            # NOTE: Unused because frequenty write caused 'free(): Invalid pointer' error
            # self._save_bool_to_shm(
            #     type="progress_info_dirty", gpu_num=cur_gpu, shm_arrs=self.dirties["progress_info"])

        self.stop_signals = []
        for gpu_id in range(AVAIL_GPU):
            self._save_bool_to_shm(
                type="stop_signal", gpu_num=gpu_id, shm_arrs=self.stop_signals)
            name = f"job_finish_flag{gpu_id}"

        self.cpu_control_func = {DECREASE_ORDER: self._del_cpu_ids,
                                 STALE_ORDER:   self._stale_cpu_ids,
                                 INCREASE_ORDER: self._add_cpu_ids}
        print(self.worker_info)
        self.origin_worker_info = np.copy(self.worker_info)
        for i in range(len(self.origin_worker_info)):
            filter_arr = self.origin_worker_info[i] != INVALID_CPU_ID
            self.origin_worker_info[i] = self.origin_worker_info[i][filter_arr]
        if __debug__:
            log.debug(
                f"globalManager: worker_info: {self.worker_info}, filter: {self.filtered_cpu_lists}, origin_workers: {self.origin_worker_info}")
            
    # FIXME: Change to np boolean value
    def _save_bool_to_shm(self, type, gpu_num, shm_arrs, np_arr=np.array([False], dtype=DIRTY_DATA_TYPE)):
        shm_name = f'{type}{gpu_num}'
        shape = np_arr.shape
        nbytes = np_arr.nbytes
        shm = SharedMemory(shm_name, flags=O_CREAT, size=nbytes)
        shm_buf = mmap.mmap(shm.fd, nbytes)
        shm.close_fd()
        shm_np_arr = np.ndarray(shape, dtype=DIRTY_DATA_TYPE, buffer=shm_buf)

        shm_np_arr[:] = np_arr[:]

        shm_arrs.append(shm_np_arr)
        self._internal_shm_metas.append((shm_buf, shm_name))

    def _save_np_arr_to_shm(self, type, gpu_num, shm_arrs, np_arr, dtype=INFO_DATA_TYPE):
        shm_name = f'{type}{gpu_num}'
        shape = np_arr.shape
        nbytes = np_arr.nbytes

        if __debug__:
            log.debug(f"globalManager: {type} size: {nbytes}")
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

    def _merge_feedback(self, order, gpu_id, cpu_ids):
        if order in self.feedback_buffer[gpu_id]:
            if len(self.feedback_buffer[gpu_id][order]) == 0:
                self.feedback_buffer[gpu_id][order] = cpu_ids
            else:
                self.feedback_buffer[gpu_id][order] = np.union1d(
                    self.feedback_buffer[gpu_id][order], cpu_ids)
        else:
            self.feedback_buffer[gpu_id][order] = cpu_ids
            self.feedback_buffer[gpu_id]["count"] += 1

        if __debug__:
            log.debug(
                f'globalManager: GPU{gpu_id}: buffer feedback {self.feedback_buffer[gpu_id]}')

        if INCREASE_ORDER in self.feedback_buffer[gpu_id] and DECREASE_ORDER in self.feedback_buffer[gpu_id]:
            decrease_order_cpus = np.setdiff1d(
                self.feedback_buffer[gpu_id][DECREASE_ORDER], self.feedback_buffer[gpu_id][INCREASE_ORDER])
            increase_order_cpus = np.setdiff1d(
                self.feedback_buffer[gpu_id][INCREASE_ORDER], self.feedback_buffer[gpu_id][DECREASE_ORDER])
            self.feedback_buffer[gpu_id][INCREASE_ORDER] = increase_order_cpus
            self.feedback_buffer[gpu_id][DECREASE_ORDER] = decrease_order_cpus
            if __debug__:
                log.debug(
                    f'globalManager: GPU{gpu_id}: Merge feedback {self.feedback_buffer[gpu_id]}')

    def _cpu_ids_control(self, order, gpu_id, cpu_ids):
        # Check send stop signal
        if self.stop_signals[gpu_id]:
            return False

        # Check non empty
        if len(cpu_ids) == 0:
            return False

        # Update worker_info of current(gpu_id) GPU
        if __debug__:
            log.debug(
                f'globalManager: GPU{gpu_id}: cpu_control with {order}, {cpu_ids}')
            
        valid_cpu_ids = self.cpu_control_func[order](gpu_id=gpu_id, cpu_ids=cpu_ids)

        # Check valid order
        if len(valid_cpu_ids) == 0:
            if __debug__:
                log.debug(
                    f'globalManager: GPU{gpu_id}: order{order} with invalid{cpu_ids}')
            return False

        if __debug__:
            log.debug(
                f'globalManager: GPU{gpu_id}: update worker_info {self.worker_info[gpu_id]}')
        # If available to send, give instant feedback.
        # Send order and set target CPU IDs
        # Then, fill others as unused data
        # Finally, set flag for dirty
        # If not, send it to order queue
        # cpu_ids_range = 1+len(valid_cpu_ids)

        self._merge_feedback(order, gpu_id, valid_cpu_ids)
        return valid_cpu_ids
    
    def _add_cpu_ids(self, gpu_id, cpu_ids):
        mask = np.isin(cpu_ids, self.worker_info[gpu_id], invert=True)
        valid_cpu_ids = cpu_ids[mask]
        self.worker_info[gpu_id] = np.append(
            self.worker_info[gpu_id], valid_cpu_ids)
        return valid_cpu_ids

    def _stale_cpu_ids(self, gpu_id, cpu_ids):
        return []

    def _del_cpu_ids(self, gpu_id, cpu_ids):
        mask = np.isin(cpu_ids, self.worker_info[gpu_id])
        valid_cpu_ids = cpu_ids[mask]
        for cpu_id in valid_cpu_ids:
            self.worker_info[gpu_id] = self.worker_info[gpu_id][self.worker_info[gpu_id] != cpu_id]
        return valid_cpu_ids

    def _get_valid_cpu_ids(self, gpu_id):   # MARKER
        # filter_arr = self.worker_info[gpu_id] != INVALID_CPU_ID
        return self.worker_info[gpu_id]#[filter_arr]
        # return self.worker_info[gpu_id]
        
    def _get_valid_return_cpu_ids(self, gpu_id):
        filter_arr = self.return_info[gpu_id] != INVALID_CPU_ID
        return self.return_info[gpu_id][filter_arr]

    def _is_cpu_ids_in_return_info(self, gpu_id, target_gpu_id_arr):
        index = np.where(np.isin(self.return_info[gpu_id], target_gpu_id_arr) == True)
        # if __debug__:
        #     log.debug(
        #         f'globalManager: GPU{gpu_id}: return_info{self.return_info[gpu_id]}, index {index[0]} , GPU{target_gpu_id} origin_info{self.origin_worker_info[target_gpu_id]}')
        return index[0]

    def _stop_handler(self):
        for stop_signal in self.stop_signals:
            if not stop_signal:
                return False
        return True

    def _policy(self):
        pass
        # for i, feedback_dirty in enumerate(self.dirties["feedback_info"]):
            # Order
            # self.feedback_info[i][0] = 0
            # CPU ID
            # self.feedback_info[i][1:] = INVALID_CPU_ID
            # self.dirties["feedback_info"][i][0] = True
            # if __debug__:
            #    log.debug(
            #     f'globalManager: GPU{i}, feedback_info={self.feedback_info}')
        # return

        # return

    def _process_feedback_buffer(self):
        for gpu_id, feedbacks in enumerate(self.feedback_buffer):
            if feedbacks["count"] == 0:
                continue
            for order in list(feedbacks.keys()):
                # Check receive
                if self.dirties["feedback_info"][gpu_id][0]:
                    break

                # Skip Metadata
                if order == "count":
                    continue

                # Delete order
                cpu_ids = feedbacks.pop(order)
                feedbacks["count"] -= 1

                # No CPU ids to send
                if len(cpu_ids) == 0:
                    continue

                # Send them
                cpu_ids_range = len(cpu_ids)+1
                self.feedback_info[gpu_id][0] = order
                self.feedback_info[gpu_id][1:cpu_ids_range] = cpu_ids[:]
                self.feedback_info[gpu_id][cpu_ids_range:] = INVALID_CPU_ID

                if __debug__:
                    log.debug(
                        f'globalManager: GPU{gpu_id}: Give bufferred feedback {self.feedback_info[gpu_id]}')
                self.dirties["feedback_info"][gpu_id][0] = True
                break

    def Server(self):
        if __debug__:
            log.debug("globalManager Execute Info Server")
        while True:
            #
            # check info dirties N sec? 100ms
            for gpu_id in range(AVAIL_GPU):
                self.gm_view_progress_info[gpu_id][:
                                                   ] = self.progress_info[gpu_id][:]

                # if __debug__:
                # log.debug(
                #     f'globalManager: GPU{gpu_id}, progress_dirty={progress_dirty}')
                # if progress_dirty[0]:
                # progress_dirty[0] = False
            self._policy()

            sleep(GM_PROCESS_INTERVAL)

            self._process_feedback_buffer()
            # Stop with unanimous approval
            if self._stop_handler():
                if __debug__:
                    log.debug("globalManager receive unanimous stop server")
                break

    def _shm_cleanup(self):
        if __debug__:
            log.debug("globalManager shared_memory cleanup")
        try:
            for shm_buf, shm_name in self._internal_shm_metas:
                shm_buf.close()
                unlink_shared_memory(shm_name)
        except Exception as e:
            log.warn(e)

    def __del__(self):
        self._shm_cleanup()


class FIFOglobalManager(globalManager):
    def __init__(self, num_workers, batch_size, micro_batch_size, num_gpu=NUM_GPU, threshold=GM_CHARITY_THRESHOLD, max_workers=None) -> None:
        super().__init__(num_workers, batch_size,
                         micro_batch_size, num_gpu, max_workers=max_workers)
        self.charity_flags = [False for _ in range(NUM_NODE)]
        self.detect_stop_signals = [False for _ in range(NUM_NODE)]
        self.steal_ledger = [{} for _ in range(NUM_NODE)]
        self.threshold_slow = int(self.num_workers * threshold)
        self.half_workers = ceil(self.num_workers / 2)
    
    def _get_available_origin_cpu_ids(self, gpu_id):
        cpu_ids = self._get_valid_cpu_ids(gpu_id)
        origin_cpu_ids = self.origin_worker_info[gpu_id]
        mask = np.isin(cpu_ids, origin_cpu_ids)
        return cpu_ids[mask]
    
    def detect_fast_slow_gpus(self, start_gpu, end_gpu):
        fastest_batch_num = self.gm_view_progress_info[start_gpu][0]
        slowest_batch_num = self.gm_view_progress_info[start_gpu][0]

        for gpu_progress_info in self.gm_view_progress_info[start_gpu+1:end_gpu]:
            if slowest_batch_num > gpu_progress_info[0]:
                slowest_batch_num = gpu_progress_info[0]
            if fastest_batch_num < gpu_progress_info[0]:
                fastest_batch_num = gpu_progress_info[0]

        return fastest_batch_num, slowest_batch_num

    def flag_handler(self, cur_node, iter_gpu_ids):
        # if __debug__:
        #     log.debug(f"Cur NODE {cur_node} with {iter_gpu_ids}")
        for gpu_id in iter_gpu_ids:
            if self.stop_signals[gpu_id] and not self.detect_stop_signals[cur_node]:
                self.detect_stop_signals[cur_node] = True
            # Restore workers to gpu
            # 1. Find target GPUs that had given workers
            # 2. Stop GPU's workers which had given
            # 3. Reclaim as Origin
            if gpu_id in self.steal_ledger[cur_node]:
                # if __debug__:
                #     log.debug(
                #         f'globalManager: Detect GPU{gpu_id} Job Finished Flag')
                if __debug__:
                    return_flag = False
                # steal_cpu_ids = self._get_valid_cpu_ids(gpu_id)
                
                # NOTE: identify given worker with initial worker bound
                #       Need to change if not efficient
                if __debug__:
                    log.debug(
                        f'globalManager: Return GPU{gpu_id} have {self.steal_ledger[cur_node][gpu_id]}')
                for stolen_gpu_id in list(self.steal_ledger[cur_node][gpu_id].keys()):
                    given_cpu_ids = self._is_cpu_ids_in_return_info(gpu_id,self.steal_ledger[cur_node][gpu_id][stolen_gpu_id])
                    
                    # stolen_idx = np.isin(given_cpu_ids, steal_cpu_ids)
                    if len(given_cpu_ids):
                        if __debug__:
                            log.debug(
                                f'globalManager: Return GPU{gpu_id} recover {given_cpu_ids} to GPU{stolen_gpu_id} from {self.worker_info[gpu_id]} and return {self.return_info[gpu_id]}')
                        # copy valid cpu_ids
                        # given_cpu_ids = self.origin_worker_info[stolen_gpu_id]
                        
                        success_cpu_ids = self._cpu_ids_control(
                            order=DECREASE_ORDER, gpu_id=gpu_id, cpu_ids=given_cpu_ids)
                        
                        self._cpu_ids_control(
                                order=INCREASE_ORDER, gpu_id=stolen_gpu_id, cpu_ids=given_cpu_ids)
                         
                        self.steal_ledger[cur_node][gpu_id][stolen_gpu_id] = np.delete(self.steal_ledger[cur_node][gpu_id][stolen_gpu_id],np.where(np.isin(self.steal_ledger[cur_node][gpu_id][stolen_gpu_id],success_cpu_ids)==True))
                            
                        if __debug__:
                            return_flag = True
                            
                        if len(self.steal_ledger[cur_node][gpu_id][stolen_gpu_id]) == 0:
                            del self.steal_ledger[cur_node][gpu_id][stolen_gpu_id]
                    elif stolen_gpu_id in self.steal_ledger[cur_node][gpu_id] and len(self.steal_ledger[cur_node][gpu_id][stolen_gpu_id]) == 0:
                        del self.steal_ledger[cur_node][gpu_id][stolen_gpu_id]
                    elif np.all(np.isin(self.origin_worker_info[stolen_gpu_id], self._get_available_origin_cpu_ids(stolen_gpu_id))):
                        if __debug__:
                            log.debug(
                                f'globalManager: Return GPU{gpu_id} have no {given_cpu_ids}: GPU{stolen_gpu_id} already has {self.origin_worker_info[stolen_gpu_id]}')
                        del self.steal_ledger[cur_node][gpu_id][stolen_gpu_id]
                    
                    
                if __debug__:
                    if return_flag:
                        log.debug(
                            f'globalManager: Return GPU{gpu_id} new worker_info {self.worker_info[iter_gpu_ids[0]:iter_gpu_ids[-1]+1]} steal_ledger{self.steal_ledger[cur_node][gpu_id]} with GPU {iter_gpu_ids}')

                if len(self.steal_ledger[cur_node][gpu_id]) == 0:
                    self.steal_ledger[cur_node].pop(gpu_id)
                    

    def charity_slow_from_fast(self, cur_node, iter_gpu_ids, fastest_batch_num, slowest_batch_num):

        if (self.charity_flags[cur_node] and slowest_batch_num == fastest_batch_num):
            self.charity_flags[cur_node] = False

        # NOTE: Reallocate finished iter workers
        # 1. Find target GPUs to give workers and receive workers
        # 2. Stop GPU's workers which will be given
        # 3. Make charity CPU list to give each GPUs
        # TODO: Check also job done signal
        elif not self.charity_flags[cur_node] and (slowest_batch_num < fastest_batch_num or self.detect_stop_signals[cur_node]):
            fastest_gpus = []
            slowest_gpus = []
            for gpu_id in iter_gpu_ids:
                # check faster than slowest batch num don't accel if faster than threshold (almost processed)
                if self.gm_view_progress_info[gpu_id][0] > slowest_batch_num or self.stop_signals[gpu_id]:
                    fastest_gpus.append(gpu_id)
                elif self.gm_view_progress_info[gpu_id][0] == slowest_batch_num and self.left_tasks[gpu_id][1] < self.threshold_slow:
                    pass
                elif not self.job_finish_flags[gpu_id]:
                    slowest_gpus.append(gpu_id)

            # There is no slowest one...
            if len(slowest_gpus) == 0:
                return
            
            available_cpu_list = {}
            total_cpu_ids = []
            if len(fastest_gpus) > len(slowest_gpus):
                for gpu_id in fastest_gpus:
                    cpu_ids = self._get_valid_cpu_ids(gpu_id)[:self.half_workers]
                    if len(cpu_ids):
                        self._cpu_ids_control(
                            order=DECREASE_ORDER, gpu_id=gpu_id, cpu_ids=cpu_ids)
                        available_cpu_list[gpu_id] = cpu_ids
                        total_cpu_ids.extend(cpu_ids)
            else:
                for gpu_id in fastest_gpus:
                    cpu_ids = self._get_valid_cpu_ids(gpu_id)
                    if len(cpu_ids):
                        self._cpu_ids_control(
                            order=DECREASE_ORDER, gpu_id=gpu_id, cpu_ids=cpu_ids)
                        available_cpu_list[gpu_id] = cpu_ids
                        total_cpu_ids.extend(cpu_ids)

            # Cannot available CPU...
            if len(total_cpu_ids) == 0:
                return
            
            if __debug__:
                log.debug(
                    f'globalManager: Charity fast_gpus:{fastest_gpus}, slow_gpus:{slowest_gpus}, fast_batch = {fastest_batch_num}, slow_batch = {slowest_batch_num} progress:{self.gm_view_progress_info}')
            
            total_cpu_ids = np.array(total_cpu_ids)
            
            for steal_gpu_id in slowest_gpus:
                if not steal_gpu_id in self.steal_ledger[cur_node]:
                    self.steal_ledger[cur_node][steal_gpu_id] = {}
            
            charity_cpu_id_lists = np.array_split(
                total_cpu_ids, len(slowest_gpus))

            for steal_gpu_id, charity_cpu_ids in zip(slowest_gpus, charity_cpu_id_lists):
                for stolen_gpu_id in available_cpu_list:
                    filter_arr = np.where(np.isin(charity_cpu_ids, available_cpu_list[stolen_gpu_id]) == True)
                    stolen_cpu_ids = charity_cpu_ids[filter_arr]
                    if not stolen_gpu_id in self.steal_ledger[cur_node][steal_gpu_id]:
                        self.steal_ledger[cur_node][steal_gpu_id][stolen_gpu_id] = stolen_cpu_ids
                    else:
                        self.steal_ledger[cur_node][steal_gpu_id][stolen_gpu_id] = np.append(self.steal_ledger[cur_node][steal_gpu_id][stolen_gpu_id],stolen_cpu_ids)
                        
                self._cpu_ids_control(
                    order=INCREASE_ORDER, gpu_id=steal_gpu_id, cpu_ids=charity_cpu_ids)
            
            if __debug__:
                log.debug(
                    f'globalManager: Charity new worker_info {self.worker_info[iter_gpu_ids[0]:iter_gpu_ids[-1]+1]} given_cpu_list={charity_cpu_id_lists}')
            self.charity_flags[cur_node] = True
    
    
    def _Intra_FIFO_policy(self):
        # Allocate only for same node gpu
        end_gpu = GPU_PER_NODE
        for start_gpu in range(0, AVAIL_GPU, GPU_PER_NODE):
            cur_node = start_gpu // GPU_PER_NODE
            iter_gpu_ids = range(start_gpu, end_gpu)
            fastest_batch_num, slowest_batch_num = self.detect_fast_slow_gpus(
                start_gpu, end_gpu)
            self.flag_handler(cur_node, iter_gpu_ids)
            # if __debug__:
            # log.debug(
            #     f'globalManager: Tracker {self.gm_view_progress_info[start_gpu:end_gpu]}')

            self.charity_slow_from_fast(
                cur_node, iter_gpu_ids, fastest_batch_num, slowest_batch_num)
            end_gpu += GPU_PER_NODE
    
    def _policy(self):
        self._Intra_FIFO_policy()

class AdaptiveFIFOglobalManager(FIFOglobalManager):
    def __init__(self, num_workers, batch_size, micro_batch_size, num_gpu=NUM_GPU, threshold=GM_CHARITY_THRESHOLD, max_workers=None) -> None:
        super().__init__(num_workers, batch_size,
                         micro_batch_size, num_gpu, max_workers=max_workers, threshold=threshold)
        self.gm_view_left_tasks = np.array([0 for _ in range(NUM_GPU)], dtype=np.int32)
        
    def charity_slow_from_fast(self, cur_node, iter_gpu_ids, fastest_batch_num, slowest_batch_num):

        # if (self.charity_flags[cur_node] and slowest_batch_num == fastest_batch_num):
        #     self.charity_flags[cur_node] = False

        # # NOTE: Reallocate finished iter workers
        # # 1. Find target GPUs to give workers and receive workers
        # # 2. Stop GPU's workers which will be given
        # # 3. Make charity CPU list to give each GPUs
        # # TODO: Check also job done signal
        # elif not self.charity_flags[cur_node] and (
        if slowest_batch_num < fastest_batch_num or self.detect_stop_signals[cur_node]:
            fastest_gpus = []
            slowest_gpus = []
            for fast_gpu_id in iter_gpu_ids:
                # check faster than slowest batch num don't accel if faster than threshold (almost processed)
                if self.gm_view_progress_info[fast_gpu_id][0] > slowest_batch_num or self.stop_signals[fast_gpu_id]:
                    fastest_gpus.append(fast_gpu_id)
                elif not self.job_finish_flags[fast_gpu_id]:
                    slowest_gpus.append(fast_gpu_id)
            # There is no slowest one...
            if len(slowest_gpus) == 0:
                return
            
            # Calculate left tasks for needed workers
            needed_workers = 0
            # new_slowest_gpus = []
            for fast_gpu_id in slowest_gpus:
                # if self.job_finish_flags[gpu_id]:
                #     break
                self.gm_view_left_tasks[fast_gpu_id] = self.left_tasks[fast_gpu_id][1]
                needed_workers += self.gm_view_left_tasks[fast_gpu_id]
                # new_slowest_gpus.append(gpu_id)
            if needed_workers == 0:
                return
            
            
            # Proportionally stealing workers
            candidate_cpu_num = needed_workers // len(fastest_gpus)
            additional_cpu_num = needed_workers % len(fastest_gpus)
            additional_candidate_cpu_num = candidate_cpu_num+1
            
            available_cpu_list = {}
            total_cpu_ids = []
            
            for fast_gpu_id in fastest_gpus:
                fast_cpu_ids = self._get_available_origin_cpu_ids(fast_gpu_id)
                # print("MANS fast_cpu_id", fast_cpu_ids, "for gpu", fast_gpu_id)
                if len(fast_cpu_ids):
                    if additional_cpu_num > 0 and additional_candidate_cpu_num < len(fast_cpu_ids):
                        fast_cpu_ids = fast_cpu_ids[:additional_candidate_cpu_num]
                        additional_cpu_num -= 1
                    elif candidate_cpu_num < len(fast_cpu_ids):
                        fast_cpu_ids = fast_cpu_ids[:candidate_cpu_num]
                    self._cpu_ids_control(
                        order=DECREASE_ORDER, gpu_id=fast_gpu_id, cpu_ids=fast_cpu_ids)
                    available_cpu_list[fast_gpu_id] = fast_cpu_ids
                    total_cpu_ids.extend(fast_cpu_ids)
            
            # if __debug__:
            #     log.debug(
            #         f'globalManager: Charity fastest_gpus:{fastest_gpus}, available_cpu_ids:{available_cpu_list}')
            # Cannot available CPU...
            if len(total_cpu_ids) == 0:
                return

            if __debug__:
                log.debug(
                    f'globalManager: Charity fast_gpus:{fastest_gpus}, slow_gpus:{slowest_gpus}, fast_batch = {fastest_batch_num}, slow_batch = {slowest_batch_num} progress:{self.gm_view_progress_info}')
            
            for steal_gpu_id in slowest_gpus:
                if not steal_gpu_id in self.steal_ledger[cur_node]:
                    self.steal_ledger[cur_node][steal_gpu_id] = {}
            
            # Proportionally giving cpu_ids
            if needed_workers > len(total_cpu_ids):
                split_slowest_gpus = []
                if len(slowest_gpus) > 1:
                    slow_left_tasks = self.gm_view_left_tasks[slowest_gpus]
                    slowest_gpus = np.array(slowest_gpus)
                    filter = slow_left_tasks>0
                    while np.any(slow_left_tasks):
                        slow_left_tasks[filter] -= np.min(slow_left_tasks[filter]) 
                        filter = slow_left_tasks>0
                        split_slowest_gpus.extend(slowest_gpus[filter])
                        if len(split_slowest_gpus) > len(total_cpu_ids):
                            split_slowest_gpus = np.flip(split_slowest_gpus)
                            break
                    else:
                        split_slowest_gpus.extend(slowest_gpus)
                else:
                    split_slowest_gpus = slowest_gpus
                # Guarantee at least 1 ratio
                # To avoid lowest gpu_id get first cpu
                
                charity_cpu_id_lists = np.array_split(
                    total_cpu_ids, len(split_slowest_gpus))
                
                if __debug__:
                    log.debug(f'globalManager: Charity split_slowest_gpus:{split_slowest_gpus}, charity_cpu_id_lists:{charity_cpu_id_lists}')              
                for steal_gpu_id, charity_cpu_ids in zip(split_slowest_gpus, charity_cpu_id_lists):
                    for stolen_gpu_id in available_cpu_list:
                        filter_arr = np.where(np.isin(charity_cpu_ids, available_cpu_list[stolen_gpu_id]) == True)
                        stolen_cpu_ids = charity_cpu_ids[filter_arr]
                        if not stolen_gpu_id in self.steal_ledger[cur_node][steal_gpu_id]:
                            self.steal_ledger[cur_node][steal_gpu_id][stolen_gpu_id] = stolen_cpu_ids
                        else:
                            self.steal_ledger[cur_node][steal_gpu_id][stolen_gpu_id] = np.append(self.steal_ledger[cur_node][steal_gpu_id][stolen_gpu_id],stolen_cpu_ids)
                    self._cpu_ids_control(
                        order=INCREASE_ORDER, gpu_id=steal_gpu_id, cpu_ids=charity_cpu_ids)
                
            # Giving amount of waiting tasks
            else:
                total_cpu_ids = np.array(total_cpu_ids)
                cpu_id_head = 0
                cpu_id_tail = 0
                for i, steal_gpu_id in enumerate(slowest_gpus):
                    cpu_id_tail += self.gm_view_left_tasks[steal_gpu_id]
                    if __debug__:
                        log.debug(f'globalManager: Charity cpu_id_tail:{cpu_id_tail}, cpu_id_head:{cpu_id_head}, available_cpu_ids:{available_cpu_list}')
                    charity_cpu_ids = total_cpu_ids[cpu_id_head:cpu_id_tail]
                        
                    for stolen_gpu_id in available_cpu_list:
                        filter_arr = np.where(np.isin(charity_cpu_ids, available_cpu_list[stolen_gpu_id]) == True)
                        stolen_cpu_ids = charity_cpu_ids[filter_arr]
                        if not stolen_gpu_id in self.steal_ledger[cur_node][steal_gpu_id]:
                            self.steal_ledger[cur_node][steal_gpu_id][stolen_gpu_id] = stolen_cpu_ids
                        else:
                            self.steal_ledger[cur_node][steal_gpu_id][stolen_gpu_id] = np.append(self.steal_ledger[cur_node][steal_gpu_id][stolen_gpu_id],stolen_cpu_ids)
                    
                    self._cpu_ids_control(
                        order=INCREASE_ORDER, gpu_id=steal_gpu_id, cpu_ids=charity_cpu_ids)
                    
                    cpu_id_head += self.gm_view_left_tasks[steal_gpu_id]

            if __debug__:
                log.debug(
                    f'globalManager: Charity new worker_info {self.worker_info[iter_gpu_ids[0]:iter_gpu_ids[-1]+1]} given_cpu_list={total_cpu_ids}, steal_ledger={self.steal_ledger[cur_node]}')
            
            self.charity_flags[cur_node] = True


class IntraInterFIFOglobalManager(FIFOglobalManager):
    def __init__(self, num_workers, batch_size, micro_batch_size, num_gpu=NUM_GPU, threshold=GM_CHARITY_THRESHOLD) -> None:
        max_workers = 0
        for cpu_ids in ALL_CPU_IDS:
            max_workers += len(cpu_ids)
        if __debug__:
            log.debug(f"gm_max_workers: {max_workers}")
        super().__init__(num_workers, batch_size, micro_batch_size,
                         num_gpu=num_gpu, threshold=threshold, max_workers=max_workers)
        # last area for inter boosting
        self.charity_flags.append(False)
        self.detect_stop_signals.append(False)
        self.steal_ledger.append({})
        self.progress_internode_fast_flag = [False for _ in range(NUM_NODE)]
        all_cpus_ids = []
        for cpu_list in self.filtered_cpu_lists:
            all_cpus_ids.extend(cpu_list)

        self.filtered_cpu_lists.append(cpu_list)
        self.interbaxtch_id = -1
    
    def _Inter_FIFO_policy(self):
        fastest_batch_num, slowest_batch_num = self.detect_fast_slow_gpus(0, AVAIL_GPU)
        if fastest_batch_num != slowest_batch_num:
            end_gpu = GPU_PER_NODE
            for start_gpu in range(0, AVAIL_GPU, GPU_PER_NODE):
                cur_node = start_gpu // GPU_PER_NODE
                iter_gpu_ids = range(start_gpu, end_gpu)
                for gpu_id in iter_gpu_ids:
                    # If slowest socket and cannot solved in Internal stealing GPUs
                    if self.gm_view_progress_info[gpu_id][0] < fastest_batch_num:
                        self.progress_internode_fast_flag[cur_node] = False
                        break
                    elif len(self._get_available_origin_cpu_ids(gpu_id)) == 0:
                        self.progress_internode_fast_flag[cur_node] = False
                    else:
                        self.progress_internode_fast_flag[cur_node] = True
                        break
                        
                end_gpu += GPU_PER_NODE

        if (np.any(self.progress_internode_fast_flag) and fastest_batch_num != slowest_batch_num) or len(self.steal_ledger[self.interbaxtch_id]):
            if __debug__:
                log.debug(
                    f'globalManager: Inter-Socket allocation {self.gm_view_progress_info} stolen {self.steal_ledger[self.interbaxtch_id]}')
            iter_gpu_ids = range(0, AVAIL_GPU)
            self.flag_handler(self.interbaxtch_id, iter_gpu_ids)
            self.charity_slow_from_fast(
                self.interbaxtch_id, iter_gpu_ids, fastest_batch_num, slowest_batch_num)

    def _policy(self):
        self._Intra_FIFO_policy()
        self._Inter_FIFO_policy()
        
            
class AdaptiveIntraInterFIFOglobalManager(AdaptiveFIFOglobalManager):
    # micro_batch_size == args.threads
    def __init__(self, num_workers, batch_size, micro_batch_size, num_gpu=NUM_GPU, threshold=GM_CHARITY_THRESHOLD) -> None:
        max_workers = 0
        for cpu_ids in ALL_CPU_IDS:
            max_workers += len(cpu_ids)
            
        # gm_max_workesr == 64
        if __debug__:
            log.debug(f"gm_max_workers: {max_workers}")
            
        super().__init__(num_workers, batch_size, micro_batch_size,
                         num_gpu=num_gpu, threshold=threshold, max_workers=max_workers)
        # last area for inter boosting
        self.charity_flags.append(False)
        self.detect_stop_signals.append(False)
        self.steal_ledger.append({})
        self.progress_internode_fast_flag = [False for _ in range(NUM_NODE)]
        all_cpus_ids = []
        # print("######## Heelim : self.filtered_cpu_lists ########: ", self.filtered_cpu_lists)
        for cpu_list in self.filtered_cpu_lists:
            all_cpus_ids.extend(cpu_list)

        self.filtered_cpu_lists.append(cpu_list)
        self.interbaxtch_id = -1
        
    def _Inter_FIFO_policy(self):
        fastest_batch_num, slowest_batch_num = self.detect_fast_slow_gpus(0, AVAIL_GPU)
        if fastest_batch_num != slowest_batch_num:
            end_gpu = GPU_PER_NODE
            for start_gpu in range(0, AVAIL_GPU, GPU_PER_NODE):
                cur_node = start_gpu // GPU_PER_NODE
                iter_gpu_ids = range(start_gpu, end_gpu)
                for gpu_id in iter_gpu_ids:
                    # If slowest socket and cannot solved in Internal stealing GPUs
                    if self.gm_view_progress_info[gpu_id][0] < fastest_batch_num:
                        self.progress_internode_fast_flag[cur_node] = False
                        break
                    elif len(self._get_available_origin_cpu_ids(gpu_id)) == 0:
                        self.progress_internode_fast_flag[cur_node] = False
                    else:
                        self.progress_internode_fast_flag[cur_node] = True
                        break
                        
                end_gpu += GPU_PER_NODE

        if (np.any(self.progress_internode_fast_flag) and fastest_batch_num != slowest_batch_num) or len(self.steal_ledger[self.interbaxtch_id]):
            if __debug__:
                log.debug(
                    f'globalManager: Inter-Socket allocation {self.gm_view_progress_info} stolen {self.steal_ledger[self.interbaxtch_id]}')
            iter_gpu_ids = range(0, AVAIL_GPU)
            self.flag_handler(self.interbaxtch_id, iter_gpu_ids)
            self.charity_slow_from_fast(
                self.interbaxtch_id, iter_gpu_ids, fastest_batch_num, slowest_batch_num)
                
    def _policy(self):
        self._Intra_FIFO_policy()
        self._Inter_FIFO_policy()

class AdaptiveInterFirstglobalManager(AdaptiveIntraInterFIFOglobalManager):
    def __init__(self, num_workers, batch_size, micro_batch_size, num_gpu=NUM_GPU, threshold=GM_CHARITY_THRESHOLD) -> None:
        super().__init__(num_workers, batch_size, micro_batch_size,
                        num_gpu=num_gpu, threshold=threshold)
    def _policy(self):
        self._Inter_FIFO_policy()
        self._Intra_FIFO_policy()

class SimpleHeuristicglobalManager(globalManager):
    def _policy(self):
        # TODO find slowdown GPU dataloader
        # and increase and decrease from fastest
        slowest_gpu = 0
        slowest_batch_num = 0
        slowest_micro_batch_num = 0

        fastest_gpu = 0
        fastest_batch_num = 0
        fastest_micro_batch_num = 0
        for i, gpu_progress_info in enumerate(self.progress_info):
            if fastest_batch_num < gpu_progress_info[0]:
                fastest_gpu = i
                fastest_batch_num = gpu_progress_info[0]
                fastest_micro_batch_num = gpu_progress_info[1]
            elif slowest_batch_num == gpu_progress_info[0] and fastest_micro_batch_num < gpu_progress_info[1]:
                fastest_gpu = i
                fastest_micro_batch_num = gpu_progress_info[1]

            if slowest_batch_num > gpu_progress_info[0]:
                slowest_gpu = i
                slowest_batch_num = gpu_progress_info[0]
                slowest_micro_batch_num = gpu_progress_info[1]
            elif slowest_batch_num == gpu_progress_info[0] and slowest_micro_batch_num > gpu_progress_info[1]:
                slowest_gpu = i
                slowest_micro_batch_num = gpu_progress_info[1]

        for i, feedback_dirty in enumerate(self.dirties["feedback_info"]):
            return
        return


class globalManagerHelper():
    def __init__(self, rank=None, max_workers=None) -> None:
        # log.info("Lookup Service")
        # NOTE Flexible max gpu? needed?
        # NOTE: worker_info
        # {<GPU_NUM>:<CPU_IDS>}
        if rank is None:
            rank = torch.cuda.current_device()
        self.gpu_num = rank

        if max_workers == None:
            self.max_workers = len(ALL_CPU_IDS[0]) * NUM_NODE
        else:
            self.max_workers = max_workers
        
        feedback_shape = (self.max_workers+1,)
        return_shape = (self.max_workers,)
        
        self.feedback_info_frame = np.full(
            feedback_shape, -1, dtype=INFO_DATA_TYPE)
        return_info_frame = np.full(
            return_shape, -1, dtype=INFO_DATA_TYPE)
        # FIXME: hardcoded frame
        # FIXME: Delete after allocating shared memory
        # self.worker_info_frame = np.array([], dtype=INFO_DATA_TYPE)
        lefttask_frame = np.array([0,0], dtype=INFO_DATA_TYPE)
        
        progress_info_frame = np.array(
            [0, 0], dtype=PROGRESS_INFO_DATA_TYPE)
        dirty_frame = np.array([False])
        
        # for cleanup
        self._internal_shm_metas = []

        if __debug__:
            log.info(
                f"GPU{self.gpu_num} globalManagerHelper: Load Shared Memory")
        # self.worker_info = self._load_np_arr("worker_info")

        # Prgoress info
        # [<PROCESSED_BATCH_NO>,<PROCESSED_MICRO_BATCH_NO>]
        self.progress_info = self._load_np_arr(
            "progress_info", dtype=PROGRESS_INFO_DATA_TYPE, frame=progress_info_frame)
        # self.progress_dirty = self._load_bool("progress_info_dirty")
        # FeedBack type
        # Decrease: INVALID_CPU_NUM, <CPU_ID>
        # Stall: 0, <ERROR CODE> (INVALID_CPU_NUM=No Error, -255=globalManager failed)
        # Increase: 1, <CPU_ID>
        self.feedback_info = self._load_np_arr("feedback_info", frame=self.feedback_info_frame)
        self.return_info = self._load_np_arr("return_info", frame=return_info_frame)
        self.feedback_dirty = self._load_bool("feedback_info_dirty", frame=dirty_frame)
        # if self.gpu_num == 0:
        self.job_finish_flag = self._load_bool("job_finish_flag", frame=dirty_frame)
        self.server_stop_signal = self._load_bool("stop_signal", frame=dirty_frame)
        self.left_tasks = self._load_np_arr("left_tasks", frame=lefttask_frame)
        self.estimate_gpu_flag = self._load_bool("estimate_gpu_flag", frame=dirty_frame)
        
    # FIXME: Change to np boolean value
    def _load_bool(self, type, frame):
        if __debug__:
            log.debug(
                f"GPU{self.gpu_num} globalManagerHelper: shm load {type}")

        size = frame.nbytes
        shape = frame.shape

        shm_name = f'{type}{self.gpu_num}'
        shm = SharedMemory(name=shm_name)
        shm_buf = mmap.mmap(shm.fd, size)
        shm.close_fd()
        shm_np_arr = np.ndarray(
            shape=shape, dtype=DIRTY_DATA_TYPE, buffer=shm_buf)
        self._internal_shm_metas.append((shm_buf, shm_name))
        return shm_np_arr

    def Report_task(self, batch_num, tasks):
        if self.left_tasks[0] != batch_num:
            self.left_tasks[0] = batch_num
        if self.left_tasks[1] != tasks:
            self.left_tasks[1] = tasks
            if __debug__:
                log.debug(
                    f"GPU{self.gpu_num} globalManagerHelper: batch_num {batch_num} tasks {tasks} Left tasks {self.left_tasks}")
        # print(f"[Report_task]: GPU{self.gpu_num} globalManagerHelper: batch_num {batch_num} tasks {tasks} Left tasks {self.left_tasks}")

    def _load_np_arr(self, type, frame=None, dtype=INFO_DATA_TYPE):
        shape = frame.shape
        size = frame.nbytes
        # if frame == None:
        #     raise ValueError(
        #         f"GPU{self.gpu_num} globalManagerHelper: Cannot load with None frame of {type}")

        shm_name = f'{type}{self.gpu_num}'
        print(shm_name)
        if __debug__:
            log.debug(f"globalHelper {type} size: {size}")
        shm = SharedMemory(name=shm_name)
        shm_buf = mmap.mmap(shm.fd, size)
        shm.close_fd()
        shm_np_arr = np.ndarray(
            shape=shape, dtype=dtype, buffer=shm_buf)
        self._internal_shm_metas.append((shm_buf, shm_name))
        return shm_np_arr

    def Send(self, current_batch, micro_batch_num):
        self.progress_info[0] = current_batch
        self.progress_info[1] = micro_batch_num
        if __debug__:
            log.debug(
                f"GPU{self.gpu_num} globalManagerHelper: progress_info={self.progress_info}")
        # self.progress_dirty[0] = True
    
    def Get_Job_done_flag(self):
        return self.job_finish_flag[0]
    
    def Get_GPU_flag(self):
        return self.estimate_gpu_flag[0]

    def Get_left_tasks(self):
        return self.left_tasks[1]

    def Get_progress_iter(self):
        return self.progress_info[0]
    
    def Get_progress_batch(self):
        return self.progress_info[1]
    
    def Report_return(self, cpu_id):
        if self.return_info[cpu_id] != cpu_id:
            self.return_info[cpu_id] = cpu_id
            if __debug__:
                log.debug(
                    f"GPU{self.gpu_num} globalManagerHelper: Update return_info={self.return_info}")

    def Cleanup_return(self, cpu_id):
        if self.return_info[cpu_id] != INVALID_CPU_ID:
            self.return_info[cpu_id] = INVALID_CPU_ID
            if __debug__:
                log.debug(
                    f"GPU{self.gpu_num} globalManagerHelper: Cleanup return_info={self.return_info}")
        
    def Job_done(self):
        if __debug__:
            log.debug(
                f"GPU{self.gpu_num} globalManagerHelper: Job done signal")
        if not self.job_finish_flag[0]:
            self.job_finish_flag[0] = True
    
    def Gpu_start(self):
        if __debug__:
            log.debug(
                f"GPU{self.gpu_num} globalManagerHelper: GPU start signal")
        if not self.estimate_gpu_flag[0]:
            self.estimate_gpu_flag[0] = True
            
    def Gpu_done(self):
        if __debug__:
            log.debug(
                f"GPU{self.gpu_num} globalManagerHelper: GPU done signal")
        if self.estimate_gpu_flag[0]:
            self.estimate_gpu_flag[0] = False
            
    def Job_start(self):
        if __debug__:
            log.debug(
                f"GPU{self.gpu_num} globalManagerHelper: Job start signal")
        if self.job_finish_flag[0] and self.progress_info[0] >= self.left_tasks[0]:
            self.job_finish_flag[0] = False

    def Recv(self):
        if self.feedback_dirty[0]:
            if __debug__:
                log.debug(
                    f"GPU{self.gpu_num} globalManagerHelper: feedback_info={self.feedback_info}")
            self.feedback_info_frame[:] = self.feedback_info[:]
            self.feedback_dirty[0] = False
            return list(self.feedback_info_frame)
        # if __debug__:
        #     log.debug(
            # f"GPU{self.gpu_num} globalManagerHelper: feedback_info=None")
        return None

    def Close(self):
        # if self.gpu_num == 0:
        if __debug__:
            log.debug(
                f"GPU{self.gpu_num} globalManagerHelper: Send Stop signal")
        self.server_stop_signal[0] = True
        if not self.job_finish_flag[0]:
            self.job_finish_flag[0] = True

    def _shm_cleanup(self):
        if __debug__:
            log.debug(f"GPU{self.gpu_num} globalManagerHelper: Cleanup")
        try:
            for shm_buf, shm_name in self._internal_shm_metas:
                shm_buf.close()
        except Exception as e:
            log.warn(e)

    def __del__(self):
        self._shm_cleanup()


def _globalManager_backend(num_workers, batch_size, micro_batch_size, controllerClass, threshold):
    # p = psutil.Process()
    # cpu_ids = [ALL_CPU_IDS[-1][-1]]

    # p.cpu_affinity(cpu_ids)
    if threshold == None:
        gm = controllerClass(
            num_workers=num_workers, batch_size=batch_size, micro_batch_size=micro_batch_size)
    else:
        gm = controllerClass(num_workers=num_workers, batch_size=batch_size,
                             micro_batch_size=micro_batch_size, threshold=threshold)
    gm.Server()


# TODO: Add CPU pinning to globalManager
def globalManager_init(num_workers, batch_size, micro_batch_size, controllerClass, rank=0, threshold=None):
    # if __debug__:
    #         log.debug(f"My rank is {rank}")
    if rank == 0:
        if __debug__:
            log.debug(f"spawn GM")
        gm_process = Process(
            target=_globalManager_backend,
            args=(num_workers, batch_size, micro_batch_size,
                  controllerClass, threshold,)
        )
        gm_process.daemon = True
        gm_process.start()
        if __debug__:
            log.debug(f"start GM")
    else:
        if __debug__:
            log.debug(f"Nothing only for rank{rank}, not rank0")

# NOTE: Unused internal function to close globalManager server


def _globalManager_close(Helper):
    Helper.Close()
