#!/bin/bash
PY_FILE="./pyfiles/"
NUM_PROC=$1
shift
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC $PY_FILE/train.py "$@"
