#!/bin/bash

CUR_DIR="."
LOG_DIR="$CUR_DIR/log"

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
    OPTION=${10}
else
    OPTION=$9
fi

if [ ${OPTION} == "*test*" ]; then
    LOG_DIR="$CUR_DIR/log/test"
elif [ ${OPTION} == "*shared*" ]; then
    LOG_DIR="$CUR_DIR/log/shared"
fi

$SCRIPT_DIR/cleanup.sh
echo "Clear caches"
sudo $SCRIPT_DIR/clearCaches.sh
NEW_LOG_DIR="$LOG_DIR/$TYPE/$FILENAME/$DATAFOLDER/$AUG/$MODEL/epoch${EPOCH}/b${BATCH}/worker${WORKERS}/thread${THREADS}"
echo "Start $TYPE, $FILENAME at $NEW_LOG_DIR"

if [ -d $NEW_LOG_DIR ]; then
    rm -r $NEW_LOG_DIR
    echo "Remove pre-existed folder"
fi
mkdir -p $NEW_LOG_DIR

$SCRIPT_DIR/$TYPE.sh $FILENAME $DATA_DIR $MODEL $EPOCH $BATCH $WORKERS $THREADS $AUG &> ${NEW_LOG_DIR}/output.log &
python_pid=$!
# $SCRIPT_DIR/cpu_temp_logger.sh &> ${NEW_LOG_DIR}/cpu_temp_log.csv &
wait $python_pid
$SCRIPT_DIR/cleanup.sh
echo "Done!"