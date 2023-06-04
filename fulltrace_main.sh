#!/bin/bash
SHELL=`basename "$0"`

if [ $# != 0 ]; then
    echo "$SHELL: USAGE: $SHELL"
    exit 1
fi

CUR_DIR="./"
SCRIPT_DIR="$CUR_DIR/scripts"
# DATASET DIR
DATADIR="/data"


# Start with cleanup the all logging executing processes
$SCRIPT_DIR/cleanup.sh
SCRIPT="thru_py_wrapper.sh"


###########
# DDP6GPU #
###########
# TYPE="DDP6GPUInterFULLTRACE"; NUM_GPU=4; EPOCH=1; WORKERS=4; BATCH=256
# BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling
# WORKERS=$(($WORKERS*$NUM_GPU)) # GPU scaling

# DATAFOLDER="size2"; EPOCH=1  # for fast test
# DATASET=$DATADIR/$DATAFOLDER
# MODEL="resnet50"
# BATCH=240
# BATCH=$(($BATCH*$NUM_GPU))
# # $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicyGPU $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment


# TYPE="DDP6GPUInterFULLTRACEFETCH"; NUM_GPU=6; EPOCH=1; WORKERS=4; BATCH=256
# BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling
# WORKERS=$(($WORKERS*$NUM_GPU)) # GPU scaling

# DATAFOLDER="openimage"; EPOCH=1  # for fast test
# DATASET=$DATADIR/$DATAFOLDER
# MODEL="resnet50"
# BATCH=240
# BATCH=$(($BATCH*$NUM_GPU))
# # $SCRIPT_DIR/$SCRIPT $TYPE origin_main_pin $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# # $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicyGPUDirectAggrOffloadSingleSamp $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment

# # $SCRIPT_DIR/$SCRIPT $TYPE origin_main_pin $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment

# # BATCH=116
# # BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
# # MODEL="efficientnet-b3"
# # $SCRIPT_DIR/$SCRIPT $TYPE origin_main_pin $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment

# # BATCH=56
# # BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
# # MODEL="efficientnet-b4"
# # $SCRIPT_DIR/$SCRIPT $TYPE origin_main_pin $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment

# # BATCH=56
# # BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
# # MODEL="efficientnet-b4"
# # $SCRIPT_DIR/$SCRIPT $TYPE origin_main_pin $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment

# # BATCH=116
# # BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
# # MODEL="vit-base"
# # $SCRIPT_DIR/$SCRIPT $TYPE origin_main_pin $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 autoaugment





# TYPE="DDP6GPUInterFULLTRACE"; NUM_GPU=6; EPOCH=1; WORKERS=4;
# WORKERS=$(($WORKERS*$NUM_GPU)) # GPU scaling
# MODEL="resnet50"
# DATAFOLDER="openimage"
# DATASET=$DATADIR/$DATAFOLDER
# BATCH=240
# BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling
# # $SCRIPT_DIR/$SCRIPT $TYPE origin_main_pin $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 autoaugment
# # $SCRIPT_DIR/$SCRIPT $TYPE origin_main_pin $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment



TYPE="SINGLEFULLTRACE"; NUM_GPU=1; EPOCH=1; WORKERS=4;
WORKERS=$(($WORKERS*$NUM_GPU)) # GPU scaling
MODEL="resnet50"
DATAFOLDER="size2"
DATASET=$DATADIR/$DATAFOLDER
BATCH=240
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling
# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicyGPUDirectAggrOffload $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 autoaugment
# $SCRIPT_DIR/$SCRIPT $TYPE GPUonlyNaive $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 autoaugment
$SCRIPT_DIR/$SCRIPT $TYPE CPUPolicyGPUDirectAggrOffloadRefurbish $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 autoaugment
