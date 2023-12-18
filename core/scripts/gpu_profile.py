import subprocess
import time
import argparse
import signal
import psutil
                
def log_gpu_stats(fname, interval=0.1):
    """
    Logs GPU utilization and memory consumption.
    
    Arguments:
    interval -- the logging interval in seconds (default is 0.1 seconds)
    """
    def signal_handler(signum, frame):
        print("Terminating logging process...")
        f.close()
        exit(0)
    
    with open(fname, 'w') as f:
        f.write('time,gpu_id,gpu_utilization,memory_usage,cpu_usage\n')
        signal.signal(signal.SIGTERM, signal_handler)
        while True:
            try:
                output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--id=0' , '--format=csv,noheader'])
                output = output.decode('utf-8')
                lines = output.strip().split('\n')
                for line in lines:
                    values = line.split(',')
                    gpu_id = int(values[0].strip())
                    gpu_utilization = values[1].strip()
                    memory_usage = values[2].strip().split()[0]
                    current_time = time.time()
                    cpu_usage = psutil.cpu_percent()
                    f.write('{:.2f},{:d},{},{:d}, {},\n'.format(current_time, gpu_id, gpu_utilization, int(memory_usage), cpu_usage))
                f.flush()
                time.sleep(interval)
            except subprocess.CalledProcessError:
                break
        

def main(log_file, interval=0.1):
    log_gpu_stats(log_file, interval)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Log GPU statistics.')
    parser.add_argument('log_file', type=str, help='the name of the file to log to')
    parser.add_argument('--interval', type=int, default=1, help='the logging interval in seconds (default is 1 second)')
    args = parser.parse_args()

    main(args.log_file, args.interval)