from localManagerCPUGPUDirectAggrOffloadTaskExecution import run_worker

# FastFlow Only
if __name__ == '__main__':
    host="0.0.0.0"
    port=51234
    is_intra_batching=True
    run_worker(host=host, port=port, parallel=4, is_intra_batching=is_intra_batching, cpu_to_focus=31)



