# no intra batching, each worker has a connection
NUM_WORKERS=4
i=0
port=45123
while [ "${i}" -ne "${NUM_WORKERS}" ];
do
  python LaunchRemoteWorkerFastFlow.py --no-intra-batching -cpu_id "${i}" -port "${port}" &
  i=$((i + 1))
  port=$((port + 1))
done
wait