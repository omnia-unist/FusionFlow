#!/bin/bash

CUR_DIR="./"
LOG_DIR="$CUR_DIR/log/perf/"
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
script_id=$!
sleep 1s
python_pid=`ps aux | grep -i 'python -O .//' | head -n 1 | awk -F ' ' '{print $2}'`
echo "$python_pid"
ps aux | grep -i 'python -O .//'
metrics='l1d.replacement,l1d_pend_miss.fb_full,l2_rqsts.miss,l2_rqsts.references,l2_rqsts.all_demand_data_rd,l2_rqsts.all_demand_miss,l2_rqsts.code_rd_hit,l2_rqsts.code_rd_miss,l2_rqsts.demand_data_rd_hit,l2_rqsts.demand_data_rd_miss,longest_lat_cache.miss,longest_lat_cache.reference,cycle_activity.cycles_l3_miss,cycle_activity.stalls_l3_miss,offcore_requests.l3_miss_demand_data_rd,offcore_requests_outstanding.cycles_with_l3_miss_demand_data_rd,offcore_requests_outstanding.l3_miss_demand_data_rd,offcore_requests_outstanding.all_data_rd,offcore_requests_buffer.sq_full,dtlb_load_misses.miss_causes_a_walk,dtlb_load_misses.stlb_hit,dtlb_store_misses.miss_causes_a_walk,dtlb_store_misses.stlb_hit'
# l1d_pend_miss.pending_cycles,
sudo perf record -o ${NEW_LOG_DIR}/perf.data -p $python_pid -e ${metrics} -a &

wait $script_id
$SCRIPT_DIR/cleanup.sh
sudo chown -R chanho:chanho ${NEW_LOG_DIR}

echo "Done!"


