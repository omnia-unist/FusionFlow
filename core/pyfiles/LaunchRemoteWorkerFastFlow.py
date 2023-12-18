import argparse
from RPCDataLoader import run_worker

parser = argparse.ArgumentParser(description='Launching remote workers')
parser.add_argument('-host', default="0.0.0.0", nargs='?', help='host address')
parser.add_argument('-port', default=45123, nargs='?', type=int, help='port address')
parser.add_argument('-parallel', type=int, nargs='?', default=1, help='number of concurrent tasks')
parser.add_argument('-cpu_id', type=int, nargs='?', default=31, 
                    help='exact cpu to pin, (only useful only when "parallel" = 1)')
parser.add_argument('--no-intra-batching', dest="no_intra_batching",
                    action='store_true', default=False, help='Turn on/off intra batching')

# FastFlow Only
if __name__ == '__main__':
    args = parser.parse_args()
    
    host = args.host
    port = args.port
    is_intra_batching = not args.no_intra_batching
    parallel = args.parallel
    cpu_to_focus = args.cpu_id

    run_worker(host=host, port=port, parallel=parallel, is_intra_batching=is_intra_batching, cpu_to_focus=cpu_to_focus)


