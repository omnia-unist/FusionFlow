#!/bin/bash
SHELL=`basename "$0"`

if [ $# != 0 ]; then
    echo "$SHELL: USAGE: $SHELL"
    exit 1
fi
CUR_DIR="./"
SCRIPT_DIR="$CUR_DIR/scripts"
DATADIR="/data/opensets"
SCRIPT="thru_py_wrapper.sh"

#############
#           #
# 6GPUInter #
#           #
#############

TYPE="DDP4GPU"; NUM_GPU=4; EPOCH=2; WORKERS=12; 
WORKERS=$(($WORKERS*$NUM_GPU)) # GPU scaling

DATAFOLDER="imagenet-pytorch"; EPOCH=180 # for fast test
DATASET=$DATADIR/$DATAFOLDER

BATCH=256
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
MODEL="resnet50"

$SCRIPT_DIR/$SCRIPT $TYPE baselineConverge $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 croprandaugment
$SCRIPT_DIR/$SCRIPT $TYPE OursConverge $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 croprandaugment
