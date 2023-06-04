from multiprocessing import Process
import logging
import zmq
import torch
GPU_NUM = torch.cuda.device_count()
CONTEXT = zmq.Context()
# 
# PORTS = []
# text_format = "127.0.0.1:{PORT}"
# target_ports = [i for i in range(9990,9999)]
# for port in target_ports:
#     PORTS.append(text_format.format(PORT=port))
gpu_loader_ports = []
for i in GPU_NUM:
    gpu_loader_ports.append(f"ipc:///tmp/feeds/{i}")
    
FEEDBACK_PORT = "ipc:///tmp/feeds/0"

# if __debug__:
logging.basicConfig(
    level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

class globalManager():
    def __init__(self, batch_size, worker_batch_size, worker_num) -> None:
        print("Connecting to machine...")
        self.worker_info_collector = CONTEXT.socket(zmq.SUB)
        for port in gpu_loader_ports:
            self.worker_info_collector.connect(port)
            print("Successfully connected to machine %s" % port)

        self.feedback_controller = CONTEXT.socket(zmq.REP)
        # for port in gpu_loader_ports2:
        self.feedback_controller.bind(FEEDBACK_PORT)
        print("Successfully bind to machine %s" % FEEDBACK_PORT)
        
        self.worker_info = {}
        self.batch_size = batch_size
        self.worker_batch_size = worker_batch_size

        for i in range(self.rank):
            self.worker_info[i] = worker_num
            self.progress_info[i] = None

    def Get_comm(self):
        return self.comms

    def Server(self):
        log.debug("Execute Info Server", flush=True)
        log.debug(f"world_size: {self.world_size}, rank: {self.rank}", flush=True)

        while True:
            log.debug(f'globalManager Server recv', flush=True)

            req = self.worker_info_collector.recv_pyobj()
        
            if req == 'close':
                log.debug(f'globalManager Client connection closed', flush=True)
                self.world_size -= 1
                if self.world_size == 1:
                    break
            else:
                log.debug('globalManager Server processinfo', flush=True)
                data = req

                self.progess_info.update(data)
                log.debug(f'{self.progess_info}', flush=True)
                
            # TODO: Add feedback policy    
            #  Wait for next request from client
            message = self.feedback_controller.recv()
            print("Received request: ", message)
            self.feedback_controller.send_pyobj("test") # %s" % gpu_loader_ports[])
            
    # def __del__(self):
    #     if self.comm:
    #         self.comm.Disconnect()
    #         MPI.Unpublish_name(SERVICE, self.PORT)
    #         MPI.Close_port(self.PORT)

class globalManagerHelper():
    def __init__(self, rank) -> None:
        log.info("Lookup Service")
        
        print("Connecting to machine...")
        self.info_sender = CONTEXT.socket(zmq.PUB)
        self.info_sender.bind(gpu_loader_ports[rank])
        print("Successfully bind to machine %s" % gpu_loader_ports[rank])

        self.feedback_receiver = CONTEXT.socket(zmq.REQ)
        self.feedback_receiver.connect(FEEDBACK_PORT)
        log.debug(
            f'globalManager Helper Created for rank {self.rank}')

    def Send(self, data):
        self.info_sender.send_pyobj(data)

    def Recv(self):
        self.feedback_receiver.send("K")
        self.feedback_receiver.recv_pyobj()

    def Close(self):
        self.info_sender.send("close")
    
    def __del__(self):
        try:
            req = self.Close()
        except:
            pass

def globalManagerStart(batch_size, worker_batch_size, worker_num, world_size):
    log.debug("Start GC Server!!", flush=True)


def globalManager_init(batch_size, worker_batch_size, worker_num, rank=0, world_size=1):
    log.debug(f"My rank is {rank}")
    if rank == 0:
        log.debug(f"create GC")
        gc = globalManager(batch_size, worker_batch_size, worker_num, world_size)
        gc_process = Process(
            target=gc.Server(),
            args=(gc,)
        )
        gc_process.daemon = True
        log.debug(f"start GC")
        gc_process.start()

def globalManager_del():
    pass
    # MPI.Finalize()
