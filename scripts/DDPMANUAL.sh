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
echo "$MODEL"

python -O $CUR_DIR/$PY_FILE $DATA_DIR -a $MODEL -j $WORKERS -b $BATCH --epochs $EPOCH --augs $AUG -p 1 --ddpmanual --seed 49105