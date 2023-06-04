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
########\
#      #
# 8GPU #
#      #
########

TYPE="DDP"; NUM_GPU=8; EPOCH=2; WORKERS=3; BATCH=256 
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling
WORKERS=$(($WORKERS*$NUM_GPU)) # GPU scaling

# Performance Test
MODEL="resnet50"
DATAFOLDER="size10"
DATASET=$DATADIR/$DATAFOLDER


MODEL="resnet50"
DATAFOLDER="imagenet-pytorch"; EPOCH=2 # for fast test
DATASET=$DATADIR/$DATAFOLDER
# $SCRIPT_DIR/$SCRIPT $TYPE baseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE tfdata $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment

# $SCRIPT_DIR/$SCRIPT $TYPE baseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicy $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicyGPU $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE tfdata $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment


MODEL="resnet50"
DATAFOLDER="Imagenet"; EPOCH=5; BATCH=256 # for fast test
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
DATASET=$DATADIR/$DATAFOLDER
# $SCRIPT_DIR/$SCRIPT $TYPE baseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE tfdata $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment

#############
#           #
# 6GPUInter #
#           #
#############

TYPE="DDP6GPUInter"; NUM_GPU=6; EPOCH=2; WORKERS=4; BATCH=256
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling
WORKERS=$(($WORKERS*$NUM_GPU)) # GPU scaling


DATAFOLDER="size2"; EPOCH=1; BATCH=256 # for fast test
DATASET=$DATADIR/$DATAFOLDER
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
MODEL="resnet50"
# $SCRIPT_DIR/$SCRIPT $TYPE loadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# MODEL="efficientnet-b0"
# $SCRIPT_DIR/$SCRIPT $TYPE loadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# MODEL="efficientnet-b1"
# $SCRIPT_DIR/$SCRIPT $TYPE loadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s



# BATCH=116
# BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
# MODEL="efficientnet-b3"
# $SCRIPT_DIR/$SCRIPT $TYPE loadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment

# BATCH=56
# BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
# MODEL="efficientnet-b4"
# $SCRIPT_DIR/$SCRIPT $TYPE loadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment


# BATCH=116
# BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
# MODEL="vit-base"
# $SCRIPT_DIR/$SCRIPT $TYPE loadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment



# BATCH=120; BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
# MODEL="efficientnet-b2"
# $SCRIPT_DIR/$SCRIPT $TYPE loadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# MODEL="efficientnet-b3"
# $SCRIPT_DIR/$SCRIPT $TYPE loadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS 0 randaugment
# sleep 30s
# BATCH=60
# BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
# MODEL="efficientnet-b4"
# $SCRIPT_DIR/$SCRIPT $TYPE loadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS 0 randaugment

# sleep 30s
# MODEL="efficientnet-b5"
# $SCRIPT_DIR/$SCRIPT $TYPE loadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# BATCH=16; BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
# MODEL="efficientnet-b6"
# $SCRIPT_DIR/$SCRIPT $TYPE loadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# MODEL="efficientnet-b7"
# $SCRIPT_DIR/$SCRIPT $TYPE loadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s

# sleep 30s
BATCH=120; BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
MODEL="vit-base"



# BATCH=32; BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
# MODEL="vit-large"
# $SCRIPT_DIR/$SCRIPT $TYPE loadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# BATCH=16; BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
# MODEL="vit-huge"
# $SCRIPT_DIR/$SCRIPT $TYPE loadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s

# $SCRIPT_DIR/$SCRIPT $TYPE loadNtrain $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
DATAFOLDER="imagenet-pytorch"; EPOCH=1; BATCH=256 # for fast test
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
DATASET=$DATADIR/$DATAFOLDER

# BATCH=256
# BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
# MODEL="resnet50"
# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicyGPUDirectAggr $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment

# BATCH=120
# BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
# MODEL="vit-base"
# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicyGPUDirectAggr $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment

# MODEL="efficientnet-b3"
# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicyGPUDirectAggr $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment

# BATCH=60
# BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
# MODEL="efficientnet-b4"
# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicyGPUDirectAggr $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment


MODEL="resnet50"
# $SCRIPT_DIR/$SCRIPT $TYPE baseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE baseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE GPUonly $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE GPUonly $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE GPUonlyNaive $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE GPUonlyNaive $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 30s
# # $SCRIPT_DIR/$SCRIPT $TYPE tfdata $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# # sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE tfdata $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 30s
# # $SCRIPT_DIR/$SCRIPT $TYPE dalibaseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE GPUonly $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicy $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicyGPUDirectAggr $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicyGPUDirectAggr $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicyGPU $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment

MODEL="vit-base"
# $SCRIPT_DIR/$SCRIPT $TYPE baseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE baseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE tfdata $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE tfdata $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE dalibaseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE dalibaseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment

# BATCH=16
# BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling
# MODEL="efficientnet-b7"
# $SCRIPT_DIR/$SCRIPT $TYPE baseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE baseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE tfdata $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE tfdata $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE dalibaseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE dalibaseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment

# DATAFOLDER="imagenet-pytorch"; EPOCH=2; BATCH=256 # for fast test
# BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
# DATASET=$DATADIR/$DATAFOLDER
# MODEL="resnet50"
# $SCRIPT_DIR/$SCRIPT $TYPE baseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE baseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE tfdata $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE tfdata $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE dalibaseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE dalibaseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment

# MODEL="vit"
# $SCRIPT_DIR/$SCRIPT $TYPE baseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE baseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE tfdata $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE tfdata $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE dalibaseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE dalibaseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment

# MODEL="efficientnet-b7"
# $SCRIPT_DIR/$SCRIPT $TYPE baseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE baseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE tfdata $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE tfdata $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE dalibaseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE dalibaseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment

########
#      #
# 4GPU #
#      #
########

TYPE="DDP4GPU"; NUM_GPU=4; EPOCH=2; WORKERS=3; BATCH=512
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling
WORKERS=$(($WORKERS*$NUM_GPU)) # GPU scaling

# Performance Test
MODEL="resnet50"
DATAFOLDER="size10"
DATASET=$DATADIR/$DATAFOLDER

MODEL="resnet50"
DATAFOLDER="size10"
DATASET=$DATADIR/$DATAFOLDER


MODEL="resnet50"
DATAFOLDER="imagenet-pytorch"; EPOCH=1 # for fast test
DATASET=$DATADIR/$DATAFOLDER

#############
#           #
# 4GPUInter #
#           #
#############

TYPE="DDP4GPUInter"; NUM_GPU=4; EPOCH=2; WORKERS=4; BATCH=256
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling
WORKERS=$(($WORKERS*$NUM_GPU)) # GPU scaling

# MODEL="resnet50"
# DATAFOLDER="Imagenet"; EPOCH=2; BATCH=256 # for fast test
# BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
# DATASET=$DATADIR/$DATAFOLDER
# $SCRIPT_DIR/$SCRIPT $TYPE baseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicy $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE tfdata $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment

MODEL="resnet50"
DATAFOLDER="Imagenet"; EPOCH=5; BATCH=256 # for fast test
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
DATASET=$DATADIR/$DATAFOLDER
# $SCRIPT_DIR/$SCRIPT $TYPE baseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE tfdata $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment

MODEL="resnet50"
DATAFOLDER="imagenet-pytorch"; EPOCH=5; BATCH=256 # for fast test
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
DATASET=$DATADIR/$DATAFOLDER

# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicy $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE CPUGPUNoWorkerStealing $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment
# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicyGPU $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 randaugment


########
#      #
# 3GPU #
#      #
########

TYPE="DDP3GPU"; NUM_GPU=3; EPOCH=2; WORKERS=4; BATCH=512
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling
WORKERS=$(($WORKERS*$NUM_GPU)) # GPU scaling

# Performance Test
MODEL="resnet50"
DATAFOLDER="size10"
DATASET=$DATADIR/$DATAFOLDER


MODEL="resnet50"
DATAFOLDER="imagenet-pytorch"; EPOCH=1; BATCH=256 # for fast test
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling 
DATASET=$DATADIR/$DATAFOLDER
# $SCRIPT_DIR/$SCRIPT $TYPE CPUPolicy $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment

########
#      #
# 2GPU #
#      #
########

TYPE="DDP2GPU"; NUM_GPU=2; EPOCH=2; WORKERS=3; BATCH=512
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling
WORKERS=$(($WORKERS*$NUM_GPU)) # GPU scaling
MODEL="resnet50"

# Performance Test
DATAFOLDER="size2"; EPOCH=2
DATASET=$DATADIR/$DATAFOLDER

# Performance Test
DATAFOLDER="size10"
DATASET=$DATADIR/$DATAFOLDER

# Real Dataset: ImageNet
MODEL="resnet50"
DATAFOLDER="imagenet"; EPOCH=1 # for fast test
DATASET=$DATADIR/$DATAFOLDER


MODEL="resnet50"
DATAFOLDER="imagenet-pytorch"; EPOCH=1 # for fast test
DATASET=$DATADIR/$DATAFOLDER
BATCH=256; WORKERS=5
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling
# WORKERS=5
# WORKERS=$(($WORKERS*$NUM_GPU)) # GPU scaling


##########
# Sanity #
##########

TYPE="SINGLE"
DATADIR="/data/opensets"
DATAFOLDER="size2"
DATASET=$DATADIR/$DATAFOLDER
MODEL="resnet34"
BATCH=256; EPOCH=1; 
# WORKERS=1
# $SCRIPT_DIR/$SCRIPT $TYPE fordebug $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# WORKERS=2
# $SCRIPT_DIR/$SCRIPT $TYPE fordebug $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# WORKERS=4
# $SCRIPT_DIR/$SCRIPT $TYPE fordebug $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default
# WORKERS=8
# $SCRIPT_DIR/$SCRIPT $TYPE fordebug $DATASET $MODEL $EPOCH $BATCH $WORKERS 4 default


# $SCRIPT_DIR/$SCRIPT $TYPE sanity_test $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 none
# $SCRIPT_DIR/$SCRIPT $TYPE sanity_global $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 none



TYPE="DDP2GPUNoDiskSocket"; NUM_GPU=2; EPOCH=2; WORKERS=5; BATCH=512
BATCH=$(($BATCH*$NUM_GPU)) # GPU scaling
WORKERS=$(($WORKERS*$NUM_GPU)) # GPU scaling
MODEL="resnet50"

DATAFOLDER="imagenet-pytorch"; EPOCH=1 # for fast test
DATASET=$DATADIR/$DATAFOLDER
# $SCRIPT_DIR/$SCRIPT $TYPE baselineTwoGPUdiffsocket $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment
# $SCRIPT_DIR/$SCRIPT $TYPE baselineBackFIFOPolicydiffsocket $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment



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
# $SCRIPT_DIR/$SCRIPT $TYPE baseline $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment
# $SCRIPT_DIR/$SCRIPT $TYPE baselineMicro $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment
# $SCRIPT_DIR/$SCRIPT $TYPE baselineFIFOPolicy $DATASET $MODEL $EPOCH $BATCH $WORKERS 8 randaugment
# $SCRIPT_DIR/$SCRIPT $TYPE baselineNoPinCPU $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS default
# $SCRIPT_DIR/$SCRIPT $TYPE baseline $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS randaugment

# DATAFOLDER="size80iter"
# DATASET=$DATADIR/$DATAFOLDER
# $SCRIPT_DIR/$SCRIPT $TYPE prepNloadNtrainPMPWPinOverwriteSampler $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS norandom

# DATAFOLDER="size10"
# DATASET=$DATADIR/$DATAFOLDER
# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE train $DATASET $MODEL $EPOCH $BATCH 4 $THREADS norandom
# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE trainNoAMP $DATASET $MODEL $EPOCH $BATCH 2 $THREADS norandom
# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE loadNtrainPWPMPin $DATASET $MODEL $EPOCH $BATCH 4 $THREADS norandom

# MODEL="resnet50"
# DATAFOLDER="Imagenet"
# DATASET=$DATADIR/$DATAFOLDER
# # $SCRIPT_DIR/$SCRIPT $TYPE fetchNbaseline $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS default
# $SCRIPT_DIR/$SCRIPT $TYPE baseline $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS default



# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE loadNtrainPWPMPinNoAMP $DATASET $MODEL $EPOCH $BATCH 2 $THREADS norandom
# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE prepNloadNtrainPMPWPinOverwriteSampler $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS norandom
# EPOCH=3
# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE fetchNbaseline $DATADIR/size320 $MODEL $EPOCH $BATCH $WORKERS $THREADS norandom



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
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE loadNtrainPWPMPinNoAMP $DATASET $MODEL $EPOCH $BATCH 2 $THREADS norandom
# sleep 30s
# $SCRIPT_DIR/$SCRIPT $TYPE prepNloadNtrainPMPWPinOverwriteSampler $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS norandom
# sleep 3m
# $SCRIPT_DIR/$SCRIPT $TYPE baseline $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS norandom
# sleep 3m
# $SCRIPT_DIR/$SCRIPT $TYPE fetchNbaseline $DATADIR/size512 $MODEL $EPOCH $BATCH $WORKERS $THREADS norandom
# EPOCH=3
# sleep 2m
# $SCRIPT_DIR/$SCRIPT $TYPE fetchNbaseline $DATADIR/size512 $MODEL $EPOCH $BATCH $WORKERS $THREADS norandom

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
