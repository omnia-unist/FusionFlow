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


# python $CUR_DIR/$PY_FILE $DATA_DIR -a $MODEL -j $WORKERS -b $BATCH --epochs $EPOCH --augs $AUG -p 1 --gpu 0 -t $THREADS --fulltrace --tracefetch --traceaug 

# python -O $CUR_DIR/$PY_FILE $DATA_DIR -a $MODEL -j $WORKERS -b $BATCH --epochs $EPOCH --augs $AUG -p 1 --gpu 0 -t $THREADS
# CUDA_VISIBLE_DEVICES=0 python $CUR_DIR/$PY_FILE $DATA_DIR -a $MODEL -j $WORKERS -b $BATCH --epochs $EPOCH --augs $AUG -p 1 --gpu 0 -t $THREADS

# CUDA_VISIBLE_DEVICES=0 python -O $CUR_DIR/$PY_FILE $DATA_DIR -a $MODEL -j $WORKERS -b $BATCH --epochs $EPOCH --augs $AUG -p 1 --gpu 0 -t $THREADS

if [ ${FILENAME} == "DALIBaseline" ]; then
    echo "DALI baseline"
    # CUDA_VISIBLE_DEVICES=0 python -O $CUR_DIR/$PY_FILE $DATA_DIR -a $MODEL -j $WORKERS -b $BATCH --epochs $EPOCH --augs $AUG -p 1 --gpu 0 -t $THREADS --prof 50
    CUDA_VISIBLE_DEVICES=0 python -O $CUR_DIR/$PY_FILE $DATA_DIR -a $MODEL -j $WORKERS -b $BATCH --epochs $EPOCH --augs $AUG -p 1 --gpu 0 -t $THREADS
else
    # CUDA_VISIBLE_DEVICES=0 python -O $CUR_DIR/$PY_FILE $DATA_DIR -a $MODEL -j $WORKERS -b $BATCH --epochs $EPOCH --augs $AUG -p 1 --gpu 0 -t $THREADS
    CUDA_VISIBLE_DEVICES=0 python $CUR_DIR/$PY_FILE $DATA_DIR -a $MODEL -j $WORKERS -b $BATCH --epochs $EPOCH --augs $AUG -p 1 --gpu 0 -t $THREADS
fi