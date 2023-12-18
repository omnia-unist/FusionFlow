#!/bin/bash
ipcrm --all
SHELL=`basename "$0"`

if [ $# != 0 ]; then
    echo "$SHELL: USAGE: $SHELL"
    exit 1
fi
CUR_DIR="./"
SCRIPT_DIR="$CUR_DIR/scripts"
DATADIR="/srv"
SCRIPT="thru_py_wrapper.sh"

############
#          #
#  SINGLE  #
#          #
############

# TYPE="SINGLE"; NUM_GPU=1; EPOCH=1; WORKERS=4;
TYPE="DDP6GPUInter"; NUM_GPU=6; EPOCH=1; WORKERS=4;

WORKERS=$(($WORKERS*$NUM_GPU)) # GPU scaling

# DATAFOLDER="size2"; EPOCH=1 # for fast test
DATAFOLDER="openimage"; EPOCH=1 # for fast test
# DATAFOLDER="openimage"; EPOCH=1 # for fast test
DATASET=$DATADIR/$DATAFOLDER

MODEL="resnet18";
# MODEL="resnet50";
AUGS="randaugment"

# BATCH=600
BATCH=500
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling

# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicyGPUAoT $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 $AUGS
# sleep 30s

# $SCRIPT_DIR/$SCRIPT $TYPE NaiveGPUExecution $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 30s


MODEL="resnet50";
BATCH=240
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling

# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicyGPUAoT $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 $AUGS
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE NaiveGPUExecution $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 30s



MODEL="vit-base"
BATCH=80
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling

# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicyGPUAoT $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 $AUGS
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE NaiveGPUExecution $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 30s

# sleep 30s
AUGS="autoaugment"

MODEL="resnet18";
BATCH=500
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling

# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicyGPUAoT $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 $AUGS
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE NaiveGPUExecution $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 autoaugment
# sleep 30s


MODEL="resnet50";
BATCH=240
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling

# $SCRIPT_DIR/$SCRIPT $TYPE NaiveGPUExecution $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 autoaugment
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicyGPUAoT $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 $AUGS
# sleep 30s


MODEL="vit-base"
BATCH=80
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling

# $SCRIPT_DIR/$SCRIPT $TYPE NaiveGPUExecution $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 autoaugment
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicyGPUAoT $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 $AUGS
# sleep 30s

AUGS="deepautoaugment"

MODEL="resnet18";
BATCH=500
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling

# $SCRIPT_DIR/$SCRIPT $TYPE NaiveGPUExecution $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 deepautoaugment
# sleep 30s


# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicyGPUAoT $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 $AUGS
# sleep 30s

# $SCRIPT_DIR/$SCRIPT $TYPE Baseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 $AUGS
# sleep 30s

MODEL="resnet50";
BATCH=240
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling

# $SCRIPT_DIR/$SCRIPT $TYPE NaiveGPUExecution $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 deepautoaugment
# sleep 30s

# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicyGPUAoT $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 $AUGS
# sleep 30s

MODEL="vit-base"
BATCH=80
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling

# $SCRIPT_DIR/$SCRIPT $TYPE NaiveGPUExecution $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 deepautoaugment
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicyGPUAoT $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 $AUGS
# sleep 30s