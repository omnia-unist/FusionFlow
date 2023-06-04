#!/bin/bash

CUR_DIR="./"
LOG_DIR="$CUR_DIR/log/test"
SCRIPT_DIR=$CUR_DIR/scripts
TYPE=$1; shift
FILENAME=$1; shift
DATA_DIR=$1; shift
MODEL=$1; shift
EPOCH=$1; shift
BATCH=$1; shift
WORKERS=$1; shift
THREADS=$1; shift
AUG=$1; shift
DATAFOLDER=`echo "$DATA_DIR" | awk -F'/' '{print $3}'` # Hard coded to find the datafolder

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
$SCRIPT_DIR/$TYPE.sh $FILENAME $DATA_DIR $MODEL $EPOCH $BATCH $WORKERS $THREADS $AUG $@ &> ${NEW_LOG_DIR}/output.log &
python_pid=$!
free -h -s 1 &> ${NEW_LOG_DIR}/memory.log &
memory_pid=$!
iostat -p /dev/sdc 1 >> ${NEW_LOG_DIR}/ssd_io.log &
iostat_pid=$!
cpustat >> ${NEW_LOG_DIR}/pid.log &
pidstat_pid=$!
sudo $SCRIPT_DIR/cachestat -t >> ${NEW_LOG_DIR}/cache.log &

wait $python_pid
$SCRIPT_DIR/cleanup.sh

echo "Done!"