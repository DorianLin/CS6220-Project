#!/bin/bash

# Optional: Set a default value for IMAGE_SIZE if needed
rm -rf ./logs
export IMAGE_SIZE=256  
export LOG_DIRECTORY=./logs
for BATCH_SIZE in 1 2 4 8 16 32 64 128
do
    echo "Running with BATCH_SIZE=${BATCH_SIZE}"
    export BATCH_SIZE
    python benchmark.py
done