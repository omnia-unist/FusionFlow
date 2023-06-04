from multiprocessing import Process
import mpi
import time
import logging
if __debug__:
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger(__name__)

# MPI.Init()

gc_process = mpi.globalController_init(256,4,4,0,1)


# mpi.globalControllerStart(256,4,4,1)
# log.debug('time to sleep 10 sec')
# time.sleep(10)

gcHelper = mpi.globalControllerHelper()
gcHelper.Send({0:4,1:4,2:4,3:4})
gcHelper.Close()

gc_process.join(timeout=3)