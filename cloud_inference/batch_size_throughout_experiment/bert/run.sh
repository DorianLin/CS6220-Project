#!/bin/bash

rm -rf ./logs
export MAX_LENGTH=128  
export LOG_DIRECTORY=./logs
export MODEL_NAME='bert-base-multilingual-cased'

echo "Compiling model ${MODEL_NAME}..."
python compile_model.py
for BATCH_SIZE in 1 2 4 8 16 32 64 128
do
    echo "Running with BATCH_SIZE=${BATCH_SIZE}"
    export BATCH_SIZE
    python benchmark.py
done