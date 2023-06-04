#!/bin/bash

CUR_DIR="."

FILENAME=$1
DATA_DIR=$2
MODEL=$3
EPOCH=$4
BATCH=$5
WORKERS=$6
THREADS=$7
AUG=$8
PY_FILE="pyfiles/$FILENAME.py"
DATAFOLDER=`echo "$DATA_DIR" | rev | cut -d '/' -f1 | rev` # Hard coded to find the datafolder
LOG_DIR="$CUR_DIR/log"
NEW_LOG_DIR="$LOG_DIR/NVPROFDDPNosync/$FILENAME/$DATAFOLDER/$AUG/$MODEL/epoch${EPOCH}/b${BATCH}/worker${WORKERS}/thread${THREADS}"

nvprof --log-file $NEW_LOG_DIR/nvprof1.%p.txt --export-profile $NEW_LOG_DIR/nvprof1.%p.nvvp --print-gpu-trace --profile-child-processes python -O $CUR_DIR/$PY_FILE $DATA_DIR -a $MODEL -j $WORKERS -b $BATCH --epochs $EPOCH --augs $AUG -p 1 --ddpmanual
