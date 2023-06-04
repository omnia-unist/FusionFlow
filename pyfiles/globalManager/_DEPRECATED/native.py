import logging
from multiprocessing import Process, JoinableQueue, Queue
if __debug__:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger(__name__)

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

    
class globalManager():
    def __init__(self, rank, world_size, worker_num) -> None:
        self.rank = rank
        self.world_size = world_size
        self.controller_address = ('localhost', 50000)     # family is deduced to be 'AF_INET'
        self.worker_info = {}
        self.feedbackqueues = {}
        progress_info = {}
        for gpu_id in range(self.world_size):
            for worker_id in range(worker_num):
                progress_info[worker_id] = (0,0)
                    
            self.worker_info[gpu_id] = progress_info
                
        self.conn = Queue()
        
    # def index_hash(self, ):
    
    def get_queue(self):
        return self.conn
    
    def server(self,feedback_clients):
        log.debug("Execute Info Server")
        
        log.debug(f'connection accepted from {listener.last_accepted}')
        while True:
            log.debug(f'Info Server recv')
            msg = conn.recv()
            if type(msg) is tuple:
                log.debug('Processing info')
                msg_type, data = msg
                if msg_type == "w":
                    self.worker_info.update(data)
                    log.debug(f'{self.worker_info}')
                elif msg_type == "p":
                    self.progess_info.update(data)
                    log.debug(f'{self.progess_info}')
            # do something with msg
            elif msg == 'close':
                conn.close()
                log.debug(f'Info Server connection closed')
                break
        log.debug(f'Info Server closed')
        listener.close()

    def client_init(self):
        self.client = Client(self.controller_address, authkey=b'controllerside')
    
    # can also send arbitrary objects:
    # conn.send(['a', 2.5, None, int, sum])
    def __del__(self):
        if self.client is not None:
            self.client.close()

class feedBackController():
    def __init__(self, rank, world_size, localport) -> None:
        self.rank = rank
        self.world_size = world_size
        self.dataloader_address = ('localhost', localport)     # family is deduced to be 'AF_INET'
        self.client = None
        
    def client_init(self):
        self.client = Client(self.dataloader_address, authkey=b'dataloaderside')
        
    def server(self):
        self.server = Listener(self.dataloader_address, authkey=b'dataloaderside')
        conn = self.server.accept()
        print('connection accepted from', self.server.last_accepted)
        while True:
            msg = conn.recv()
            # do something with msg
            if msg == 'close':
                conn.close()
                break
            
    def __del__(self):
        if self.client is not None:
            self.client.close()

def globalManagerHelper():
    def __init__(self):
        MPI.Init()



def globalManager_init(rank, num_replicas, batch_size):
    workerinfo = workerInfoController(rank, num_replicas)
    localport = 60000
    feedBackControllers=[]
    
    
    workerinfo_process = Process(
        target = workerinfo.server,
        args=(feedBackControllers,),
        daemon=True
    )
    # print("globalManager start_process")
    workerinfo_process.start()
    
    # print("before client")
    # for rank_id in range(num_replicas):
    #     fbc=feedBackController(rank_id, num_replicas, localport+rank_id)
    #     fbc.client_init()
    #     feedBackControllers.append(fbc.client)
    # print("after client")
    
    