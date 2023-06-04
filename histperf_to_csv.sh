#!/bin/bash
######################################################
# Perf profiling ( ex L1 L2 L3 DRAM, ...            )#
######################################################

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
# $SCRIPT_DIR/$SCRIPT $TYPE prepNloadNtrainPMPWPinOverwriteSampler $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS norandom
$SCRIPT_DIR/$SCRIPT $TYPE fetchNfsNprepNloadNtrainPure $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS norandom

#######
# DDP #
#######
TYPE="DDP2GPU"; EPOCH=5
BATCH=$(($BATCH*2))
WORKERS=$(($WORKERS*2))
THREADS=0

DATAFOLDER="size10"
DATASET=$DATADIR/$DATAFOLDER
sleep 3m
$SCRIPT_DIR/$SCRIPT $TYPE prepNloadNtrainPMPWPinOverwriteSampler $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS norandom
sleep 3m
$SCRIPT_DIR/$SCRIPT $TYPE fetchNfsNprepNloadNtrainPure $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS norandom

echo "Done!"

TYPE="DDP"; EPOCH=5
BATCH=$(($BATCH*4))
WORKERS=$(($WORKERS*4))
THREADS=0

DATAFOLDER="size10"
DATASET=$DATADIR/$DATAFOLDER
# sleep 5m
# $SCRIPT_DIR/$SCRIPT $TYPE prepNloadNtrainPMPWPinOverwriteSampler $DATASET $MODEL $EPOCH $BATCH $WORKERS $THREADS norandom

echo "Done!"
