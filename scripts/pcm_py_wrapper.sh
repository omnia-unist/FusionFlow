#!/bin/bash

CUR_DIR="./"
LOG_DIR="$CUR_DIR/log/pcm/"
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
fi

$SCRIPT_DIR/cleanup.sh
echo "Clear caches"
sudo $SCRIPT_DIR/clearCaches.sh
NEW_LOG_DIR="$LOG_DIR/$TYPE/$FILENAME/$DATAFOLDER/$AUG/$MODEL/epoch${EPOCH}/b${BATCH}/worker${WORKERS}/thread${THREADS}"
echo "Start $TYPE, $FILENAME at $NEW_LOG_DIR"

if [ -d $NEW_LOG_DIR ]; then
    sudo chown -R chanho:chanho ${NEW_LOG_DIR}
    rm -r $NEW_LOG_DIR
    echo "Remove pre-existed folder"
fi
mkdir -p $NEW_LOG_DIR

$SCRIPT_DIR/$TYPE.sh $FILENAME $DATA_DIR $MODEL $EPOCH $BATCH $WORKERS $THREADS $AUG &> ${NEW_LOG_DIR}/output.log &
python_pid=$!
sudo pcm -csv=${NEW_LOG_DIR}/pcm.csv 1 &
sudo pcm-memory -csv=${NEW_LOG_DIR}/pcm_memory.csv 1 &
# sudo pcm-core -csv=${NEW_LOG_DIR}/pcm_cpu.csv 1 &

wait $python_pid
$SCRIPT_DIR/cleanup.sh
sudo chown -R chanho:chanho ${NEW_LOG_DIR}

echo "Done!"