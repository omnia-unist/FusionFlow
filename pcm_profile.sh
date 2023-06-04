
#!/bin/bash
######################################################
# Pcm profiling ( ex IPC, memory bandwidth, request) #
######################################################

CUR_DIR="./"
SCRIPT_DIR="$CUR_DIR/scripts"

# Start with cleanup the all logging executing processes
$SCRIPT_DIR/cleanup.sh

DATADIR="/data"
DATAFOLDER="size10"
DATASET=$DATADIR/$DATAFOLDER



MODEL="resnet18"
SCRIPT="pcm_py_wrapper.sh"

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

BATCH=512
WORKERS=8
THREADS=0

TYPE="DDP2GPU"; EPOCH=5
DATAFOLDER="size10"
DATASET=$DATADIR/$DATAFOLDER
# $SCRIPT_DIR/$SCRIPT $TYPE prepNloadNtrainPMPWPinOverwriteSampler $DATASET $MODEL $EPOCH 512 8 $THREADS norandom
# $SCRIPT_DIR/$SCRIPT $TYPE fetchNfsNprepNloadNtrain $DATADIR/size320 $MODEL $EPOCH 512 8 $THREADS norandom

TYPE="DDP2GPUTwoSocket"; EPOCH=5
DATAFOLDER="size10"
DATASET=$DATADIR/$DATAFOLDER
# sleep 15m
# $SCRIPT_DIR/$SCRIPT $TYPE prepNloadNtrainPMPWPinOverwriteSamplerTwoGPU $DATASET $MODEL $EPOCH 512 8 $THREADS norandom
# $SCRIPT_DIR/$SCRIPT $TYPE fetchNfsNprepNloadNtrainTwoGPU $DATADIR/size320 $MODEL $EPOCH 512 8 $THREADS norandom

TYPE="DDP2GPUNoDiskSocket"; EPOCH=5
DATAFOLDER="size10"
DATASET=$DATADIR/$DATAFOLDER
# sleep 15m
# $SCRIPT_DIR/$SCRIPT $TYPE prepNloadNtrainPMPWPinOverwriteSamplerTwoGPUdiffsocket $DATASET $MODEL $EPOCH 512 8 $THREADS norandom
# sleep 15m
# $SCRIPT_DIR/$SCRIPT $TYPE fetchNfsNprepNloadNtrainTwoGPUdiffSocket $DATADIR/size320 $MODEL $EPOCH 512 8 $THREADS norandom



