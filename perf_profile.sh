#!/bin/bash
######################################################
# Perf profiling ( ex L1 L2 L3 DRAM, ...            )#
######################################################

SHELL=`basename "$0"`

if [ $# != 0 ]; then
    echo "$SHELL: USAGE: $SHELL"
    # echo "$SHELL: e.g. (upload bandwidth(Kbps): 10000, 20000, ..."
    # echo "$SHELL: e.g. (the number of target edge group):  1, 2, 3, ..."
    exit 1
fi

CUR_DIR="./"
SCRIPT_DIR="$CUR_DIR/scripts"

# Start with cleanup the all logging executing processes
$SCRIPT_DIR/cleanup.sh

DATADIR="/data"
DATAFOLDER="size10"
DATASET=$DATADIR/$DATAFOLDER


MODEL="resnet18"
SCRIPT="perf_py_wrapper.sh"

BATCH=512
WORKERS=4
THREADS=0

##########
# Single #
##########
# TYPE="SINGLEFULLTRACESEED"; EPOCH=1
# TYPE="SINGLEFULLTRACE"; EPOCH=1
TYPE="SINGLE"; EPOCH=5
# TYPE="SINGLE"; EPOCH=255

DATAFOLDER="size10"
DATASET=$DATADIR/$DATAFOLDER
$SCRIPT_DIR/$SCRIPT $TYPE prepNloadNtrainPMPWPinOverwriteSampler $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS norandom

#######
# DDP #
#######
TYPE="DDP"; EPOCH=5
BATCH=$(($BATCH*4))
WORKERS=$(($WORKERS*4))
THREADS=0

DATAFOLDER="size10"
DATASET=$DATADIR/$DATAFOLDER
sleep 5m
$SCRIPT_DIR/$SCRIPT $TYPE prepNloadNtrainPMPWPinOverwriteSampler $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS norandom

echo "Done!"
