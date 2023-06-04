#!/bin/bash

script="decode_test.py"
SCRIPT_DIR="../scripts"
END=9
for ((i=0;i<=${END};i++)); do
    python $script &> ./log/decode${i}.log
done


script="io_test.py"

END=9
for ((i=0;i<=${END};i++)); do
    sudo $SCRIPT_DIR/clearCaches.sh
    python $script &> ./log/io${i}.log
done
