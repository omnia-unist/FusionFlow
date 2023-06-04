#!/bin/bash
ODE=$1; shift
TYPE=$1; shift # dataloader name
TRAININFO=$1; shift
DATA_DIR=$1; shift
ODEL=$1; shift
EPOCH=$1; shift
BATCH=$1; shift
WORKERS=$1; shift
THREADS=$1; shift
AUG=$1; shift

CUR_DIR="./"
LOG_DIR="$CUR_DIR/log/$MODE"
SCRIPT_DIR=$CUR_DIR/scripts
DATAFOLDER=`echo "$DATA_DIR" | rev | cut -d '/' -f1 | rev` # Hard coded to find the datafolder

command="$DATA_DIR -a $MODEL -j $WORKERS -b $BATCH --epochs $EPOCH $@ "
GPU_NUM=""


if [[ $TYPE == *"2GPU"* ]]; then
    GPU_NUM="2 $command"
elif [[ $TYPE == *"DDP"* ]]; then
    GPU_NUM="4 $command"
else
    echo "$0: Pointless training with over 4GPU or 1GPU"
fi

if [[ $TRAININFO == *"PinMem"* ]]; then
    command="$command --pin-mem"
fi

if [[ $TRAININFO == *"AMP"* ]]; then
    # Use native!!
    # https://discuss.pytorch.org/t/torch-cuda-amp-vs-nvidia-apex/74994
    command="$command --native-amp"
fi

$SCRIPT_DIR/cleanup.sh
echo "Clear caches"
sudo $SCRIPT_DIR/clearCaches.sh
NEW_LOG_DIR="$LOG_DIR/$TYPE/$TRAININFO/$DATAFOLDER/$AUG/$MODEL/epoch${EPOCH}/b${BATCH}/worker${WORKERS}/thread${THREADS}"
echo "Start $TYPE, $TRAININFO at $NEW_LOG_DIR"

if [ -d $NEW_LOG_DIR ]; then
    # printf "Remove old folder? [y/N]: "
    # if read -q; then
    rm -r $NEW_LOG_DIR
    echo "Remove pre-existed folder"
    # fi
fi

# Logs
mkdir -p $NEW_LOG_DIR

$SCRIPT_DIR/$TRAININFO.sh ${command} &> ${NEW_LOG_DIR}/output.log &
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