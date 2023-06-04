from multiprocessing import Process
import logging
import mpi4py
# mpi4py.rc(initialize=False, thread_level = 'funneled')
from mpi4py import MPI
# 
SERVICE="globalManager"
INFO = MPI.Info.Create()
INFO.Set("ompi_global_scope", "true")
# if __debug__:
logging.basicConfig(
    level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

class globalManager():
    def __init__(self, batch_size, worker_batch_size, worker_num, world_size) -> None:
        port=MPI.Open_port()
        log.info("Open port")
        MPI.Publish_name(SERVICE, INFO, port)
        log.info("Open Publish")
        comms = []
        for i in range(world_size):
            log.info(f"Wait World size: {world_size}")
            comms.append( MPI.COMM_SELF.Accept(port) )
            log.info(f"Accept World size: {world_size}")
        log.info("Accept comm done")
        
        rank = comms[0].Get_rank()
        _world_size = comms[0].Get_size()

        assert _world_size == world_size
        
        self.PORT = port
        self.rank = rank
        self.world_size = _world_size

        self.worker_info = {}
        self.batch_size = batch_size
        self.worker_batch_size = worker_batch_size

        for i in range(self.rank):
            self.worker_info[i] = worker_num
            self.progress_info[i] = None
        self.comms = comms

    def Get_comm(self):
        return self.comms

    def Server(self):
        log.debug("Execute Info Server", flush=True)
        log.debug(f"world_size: {self.world_size}, rank: {self.rank}", flush=True)

        while True:
            log.debug(f'globalManager Server recv', flush=True)

            req = self.comm.irecv(source=MPI.ANY_SOURCE, tag=0)
            
            log.debug(
                f'globalManager Server wait for rank{i} info', flush=True)
            msg = req.wait()

            log.debug(f'globalManager Server get rank{i} info', flush=True)
            # do something with msg
            
            if msg == 'close':
                log.debug(f'globalManager Client connection closed', flush=True)
                self.world_size -= 1
                if self.world_size == 1:
                    break
            else:
                log.debug('globalManager Server processinfo', flush=True)
                data = msg

                self.progess_info.update(data)
                log.debug(f'{self.progess_info}', flush=True)
                
                # TODO: Add feedback function
                    
    def __del__(self):
        try:
            MPI.Close_port(self.PORT)
            MPI.Unpublish_name(SERVICE, self.PORT)
            self.comm.Disconnect()
        except AttributeError:
            pass


class globalManagerHelper():
    def __init__(self, root = 0) -> None:
        log.info("Lookup Service")
        
        print(SERVICE)
        port = MPI.Lookup_name(SERVICE)
        log.info("Waiting for connect")
        comm = MPI.COMM_WORLD.Connect(port, INFO, root)
        log.info("Connect Ready")
        rank = comm.Get_rank()
        world_size = comm.Get_size()
        self.ROOT = root
        self.PORT = port
        self.rank = rank
        self.world_size = world_size
        self.comm = comm
        log.debug(
            f'globalManager Helper Created for rank {self.rank}')

    def Send(self, data):
        return self.comm.isend(data, dest=self.ROOT, tag=self.ROOT)

    def Recv(self):
        return self.comm.irecv(source=0, tag=self.ROOT+1)
    
    def Close(self):
        return self.comm.isend("close", dest=self.ROOT, tag=self.ROOT)
    
    def __del__(self):
        try:
            req = self.Close()
            req.wait()
            self.comm.Disconnect()

def globalManagerStart(batch_size, worker_batch_size, worker_num, world_size):
    log.debug("Start GC Server!!", flush=True)


def globalManager_init(batch_size, worker_batch_size, worker_num, rank=0, world_size=1):
    log.debug(f"My rank is {rank}")
    if rank == 0:
        log.debug(f"create GC")
        gc = globalManager(batch_size, worker_batch_size, worker_num, world_size)
        gc_process = Process(
            target=globalControllerStart,
            args=(batch_size, worker_batch_size, worker_num, world_size,)
        )
        gc_process.daemon = True
        log.debug(f"start GC")
        gc_process.start()
        return gc_process
    else:
        return None
    return FAIL

def globalManager_del():
    pass
    # MPI.Finalize()
