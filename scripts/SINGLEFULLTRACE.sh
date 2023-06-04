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



if [ $# -gt 8 ]; then
    python $CUR_DIR/$PY_FILE $DATA_DIR -a $MODEL -j $WORKERS -b $BATCH -t $THREADS --epochs $EPOCH --augs $AUG -p 1 --gpu 0 --fulltrace --tracefetch
    # TRACEFETCH=$9
elif [ $# -gt 9 ]; then
    python $CUR_DIR/$PY_FILE $DATA_DIR -a $MODEL -j $WORKERS -b $BATCH -t $THREADS --epochs $EPOCH --augs $AUG -p 1 --gpu 0 --fulltrace --tracefetch --traceaug
    # TRACEAUG=$10
else
    python $CUR_DIR/$PY_FILE $DATA_DIR -a $MODEL -j $WORKERS -b $BATCH -t $THREADS --epochs $EPOCH --augs $AUG -p 1 --gpu 0 --fulltrace
fi
