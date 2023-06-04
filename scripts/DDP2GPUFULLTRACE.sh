#!/bin/bash

CUR_DIR="./"

FILENAME=$1
DATA_DIR=$2
MODEL=$3
EPOCH=$4
BATCH=$5
WORKERS=$6
THREADS=$7
AUG=$8
PY_FILE="pyfiles/$FILENAME.py"
echo "$MODEL, $#"

if [ $# -gt 9 ]; then
    echo "TraceAug"
    CUDA_VISIBLE_DEVICES=0,1 python $CUR_DIR/$PY_FILE $DATA_DIR -a $MODEL -j $WORKERS -b $BATCH --epochs $EPOCH --augs $AUG -p 1 --dist-url 'tcp://127.0.0.1:1234' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 -t $THREADS --fulltrace --tracefetch --traceaug --seed 49105
    # TRACEAUG=$10
    
elif [ $# -gt 8 ]; then
    echo "TraceFetch"
    CUDA_VISIBLE_DEVICES=0,1 python $CUR_DIR/$PY_FILE $DATA_DIR -a $MODEL -j $WORKERS -b $BATCH --epochs $EPOCH --augs $AUG -p 1 --dist-url 'tcp://127.0.0.1:1234' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 -t $THREADS --fulltrace --tracefetch --seed 49105
    # TRACEFETCH=$9
else
    echo "default"
    CUDA_VISIBLE_DEVICES=0,1 python $CUR_DIR/$PY_FILE $DATA_DIR -a $MODEL -j $WORKERS -b $BATCH --epochs $EPOCH --augs $AUG -p 1 --dist-url 'tcp://127.0.0.1:1234' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 -t $THREADS --fulltrace --seed 49105
fi
