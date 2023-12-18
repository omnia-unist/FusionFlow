#!/bin/bash
free -m
sudo sync && echo 3 > /proc/sys/vm/drop_caches
free -m

pid=$(ps -ef | grep "python ./scripts/gpu_profile.py" | grep -v grep | awk '{print $2}')

# Kill the process if it's running
if [[ -n $pid ]]; then
  echo "Killing process $pid"
  kill $pid
else
  echo "Process not found"
fi