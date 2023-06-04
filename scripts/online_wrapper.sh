#!/bin/bash

CUR_DIR="./"
LOG_DIR="$CUR_DIR/log/test"
SCRIPT_DIR=$CUR_DIR/scripts
TYPE=$1
FILENAME=$2
DATA_DIR=$3
MODEL=$4
EPOCH=$5
BATCH=$6
WORKERS=$7
THREADS=$8
DATAFOLDER=`echo "$DATA_DIR" | rev | cut -d '/' -f1 | rev` # Hard coded to find the datafolder

AUG="default"

if [ $# -gt 8 ]; then
    AUG=$9
fi

PY_FILE="pyfiles/$FILENAME.py"

python -O $CUR_DIR/$PY_FILE $DATA_DIR -a $MODEL -j $WORKERS -b $BATCH --epochs $EPOCH --augs $AUG -p 1 --gpu 0 -t $THREADS

echo "Done!"