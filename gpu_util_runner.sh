#!/bin/bash
SHELL=`basename "$0"`

if [ $# != 0 ]; then
    echo "$SHELL: USAGE: $SHELL"
    exit 1
fi
CUR_DIR="./"
SCRIPT_DIR="$CUR_DIR/scripts"
DATADIR="/data"
SCRIPT="gpu_py_wrapper.sh"


########
#      #
# 8GPU #
#      #
########

TYPE="DDP"; NUM_GPU=8; EPOCH=2; WORKERS=3; BATCH=512
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling
WORKERS=$(($WORKERS*$NUM_GPU)) # GPU scaling
MODEL="resnet50"
DATAFOLDER="openimage"; EPOCH=1 # for fast test
DATASET=$DATADIR/$DATAFOLDER
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment
# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrainMicro $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment
# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrainMicro $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrainAdaptiveBackIntraIterGM $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment
# sleep 2m
# sleep 2m
#############
#           #
# 4GPUInter #
#           #
#############

TYPE="DDP4GPUInter"; NUM_GPU=4; EPOCH=2; WORKERS=4; BATCH=128
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling
WORKERS=$(($WORKERS*$NUM_GPU)) # GPU scaling

MODEL="resnet50"
DATAFOLDER="size2"
DATASET=$DATADIR/$DATAFOLDER
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment

MODEL="resnet50"
DATAFOLDER="imagenet-pytorch"; EPOCH=1; BATCH=128 # for fast test
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
DATASET=$DATADIR/$DATAFOLDER
$SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrainAdaptiveBackIntraIterGM $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment



# MODEL="resnet50"
# DATAFOLDER="openimage"; EPOCH=1 # for fast test
# DATASET=$DATADIR/$DATAFOLDER
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment
# # sleep 2m
# # $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrainMicro $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment
# # sleep 2m
# # $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrainMicro $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrainAdaptiveBackIntraIterGM $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment
# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrainAdaptiveBackIntraIterGM $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# # sleep 2m
# # $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrainBackIntraIterGM $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment
# # sleep 2m
# # $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrainBackIntraIterGM $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# # sleep 2m

########
#      #
# 2GPU #
#      #
########

TYPE="DDP2GPU"; NUM_GPU=2; EPOCH=2; WORKERS=4; BATCH=512
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling
WORKERS=$(($WORKERS*$NUM_GPU)) # GPU scaling
MODEL="resnet50"

# Performance Test
DATAFOLDER="size2"; EPOCH=2
DATASET=$DATADIR/$DATAFOLDER
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrainBackFIFOPolicy $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment

# Performance Test
DATAFOLDER="size10"
DATASET=$DATADIR/$DATAFOLDER
# $SCRIPT_DIR/$SCRIPT $TYPE fordebug $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 norandom
# $SCRIPT_DIR/$SCRIPT $TYPE fordebugFIFOPolicyGlobalController $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 norandom
# $SCRIPT_DIR/$SCRIPT $TYPE fordebugBackworkerFIFOPolicyGlobalController $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment

# Real Dataset: ImageNet
MODEL="resnet50"
DATAFOLDER="imagenet"; EPOCH=1 # for fast test
DATASET=$DATADIR/$DATAFOLDER
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrainBackFIFOPolicy $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment
MODEL="resnet50"
DATAFOLDER="openimage"; EPOCH=1 # for fast test
DATASET=$DATADIR/$DATAFOLDER
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrainBackFIFOPolicy $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrainBackFIFOPolicy $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrain075thresBackFIFOPolicy $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrain05thresBackFIFOPolicy $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrain2thresBackFIFOPolicy $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment




##########
# Sanity #
##########

CUR_DIR="./"
SCRIPT_DIR="$CUR_DIR/scripts"
# Start with cleanup the all logging executing processes
$SCRIPT_DIR/cleanup.sh
TYPE="SINGLE"
# TYPE="DDP4GPU"
DATADIR="/data"
DATAFOLDER="size2"
DATASET=$DATADIR/$DATAFOLDER

MODEL="resnet34"
SCRIPT="start_py_wrapper.sh"
EPOCH=5
BATCH=64
WORKERS=4
EPOCH=1
# $SCRIPT_DIR/$SCRIPT $TYPE sanity_test $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 none
# $SCRIPT_DIR/$SCRIPT $TYPE sanity_global $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 none



TYPE="DDP2GPUNoDiskSocket"; NUM_GPU=2; EPOCH=2; WORKERS=5; BATCH=512
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling
WORKERS=$(($WORKERS*$NUM_GPU)) # GPU scaling
MODEL="resnet50"

DATAFOLDER="openimage"; EPOCH=1 # for fast test
DATASET=$DATADIR/$DATAFOLDER
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrainTwoGPUdiffsocket $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrainBackFIFOPolicydiffsocket $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment



##############
#            #
#            #
#            #
# DEPRECATED #
#            #
#            #
#            #
##############
# WORKERS=5; BATCH=512;
# TYPE="DDP"; EPOCH=5
# NUM_GPU=8
# BATCH=$(($BATCH*$NUM_GPU))
# WORKERS=$(($WORKERS*$NUM_GPU))
# THREADS=0
# MODEL="resnet50"
# DATAFOLDER="imagenet-pytorch"
# DATASET=$DATADIR/$DATAFOLDER
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrainMicro $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrainFIFOPolicy $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrainNoPinCPU $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS default
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS randaugment

# DATAFOLDER="size80iter"
# DATASET=$DATADIR/$DATAFOLDER
# $SCRIPT_DIR/$SCRIPT $TYPE prepNloadNtrainPMPWPinOverwriteSampler $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS norandom

# DATAFOLDER="size10"
# DATASET=$DATADIR/$DATAFOLDER
# sleep 5m
# $SCRIPT_DIR/$SCRIPT $TYPE train $DATASET $MODEL $EPOCH $BATCH 4 $THREADS norandom
# sleep 5m
# $SCRIPT_DIR/$SCRIPT $TYPE trainNoAMP $DATASET $MODEL $EPOCH $BATCH 2 $THREADS norandom
# sleep 5m
# $SCRIPT_DIR/$SCRIPT $TYPE loadNtrainPWPMPin $DATASET $MODEL $EPOCH $BATCH 4 $THREADS norandom

# MODEL="resnet50"
# DATAFOLDER="Imagenet"
# DATASET=$DATADIR/$DATAFOLDER
# # $SCRIPT_DIR/$SCRIPT $TYPE fetchNfsNprepNloadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS default
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS default



# sleep 5m
# $SCRIPT_DIR/$SCRIPT $TYPE loadNtrainPWPMPinNoAMP $DATASET $MODEL $EPOCH $BATCH 2 $THREADS norandom
# sleep 5m
# $SCRIPT_DIR/$SCRIPT $TYPE prepNloadNtrainPMPWPinOverwriteSampler $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS norandom
# EPOCH=3
# sleep 5m
# $SCRIPT_DIR/$SCRIPT $TYPE fetchNfsNprepNloadNtrain $DATADIR/size320 $MODEL $EPOCH $BATCH $WORKERS $THREADS norandom



##########
# Single #
##########

# TYPE="SINGLE"; EPOCH=5
# # TYPE="SINGLE"; EPOCH=255
# DATAFOLDER="size10"
# DATASET=$DATADIR/$DATAFOLDER

# BATCH64_MODELS=(
#     resnext101_32x8d
#     wide_resnet101_2
#     resnet152
#     densenet161
#     vgg16
#     inception_v3
# )

# BATCH128_MODELS=(
#     resnet50
#     resnet101
#     resnext50_32x4d
#     wide_resnet50_2
# )

# BATCH256_MODELS=(
#     shufflenet_v2_x1_0
#     mnasnet1_0
#     resnet18
#     squeezenet1_1
#     mobilenet_v3_small
#     mobilenet_v3_large
# )
# BATCH=512
# for _MODEL in ${BATCH128_MODELS[@]}; do
#     $SCRIPT_DIR/$SCRIPT $TYPE loadNtrainPWPMPin $DATASET ${_MODEL} $EPOCH $BATCH 2 $THREADS norandom
#     sleep 3m
# done
# BATCH=64
# for _MODEL in ${BATCH64_MODELS[@]}; do
#     $SCRIPT_DIR/$SCRIPT $TYPE loadNtrainPWPMPin $DATASET ${_MODEL} $EPOCH $BATCH 2 $THREADS norandom
#     sleep 3m
# done

# $SCRIPT_DIR/$SCRIPT $TYPE loadNtrainPWPMPin $DATASET resnet50 $EPOCH $BATCH 2 $THREADS norandom
# $SCRIPT_DIR/$SCRIPT $TYPE trainNoAMP $DATASET $MODEL $EPOCH $BATCH 2 $THREADS norandom
# sleep 1m
# $SCRIPT_DIR/$SCRIPT $TYPE loadNtrainPWPMPinNoAMP $DATASET $MODEL $EPOCH $BATCH 2 $THREADS norandom
# sleep 1m
# $SCRIPT_DIR/$SCRIPT $TYPE prepNloadNtrainPMPWPinOverwriteSampler $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS norandom
# sleep 3m
# $SCRIPT_DIR/$SCRIPT $TYPE fsNprepNloadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS norandom
# sleep 3m
# $SCRIPT_DIR/$SCRIPT $TYPE fetchNfsNprepNloadNtrain $DATADIR/size512 $MODEL $EPOCH $BATCH $WORKERS $THREADS norandom
# EPOCH=3
# sleep 5m
# $SCRIPT_DIR/$SCRIPT $TYPE fetchNfsNprepNloadNtrain $DATADIR/size512 $MODEL $EPOCH $BATCH $WORKERS $THREADS norandom

# THREADS=0
# WORKERS=3P
# $SCRIPT_DIR/$SCRIPT $TYPE fordebugIncrease $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 norandom
# WORKERS=8
# $SCRIPT_DIR/$SCRIPT $TYPE fordebugDecrease $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 norandom
# WORKERS=3
# $SCRIPT_DIR/$SCRIPT $TYPE fordebug $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 norandom

# for ((i=3;i<=6;i++)); do
#     WORKERS=$(($i))
#     $SCRIPT_DIR/$SCRIPT $TYPE fordebug $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 norandom
# done
