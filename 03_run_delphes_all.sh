#!/bin/bash

eval "$(/home/hep/us322/miniforge3/bin/conda shell.bash hook)"
conda activate madminer_env

# Arguments: process_code batch_index [supp_id]
PROCESS_CODE=$1
BATCH_INDEX=$2
SUPP_ID=$3

if [ "$PROCESS_CODE" = "signal_supp" ]; then
    python 03a_read_delphes.py -p $PROCESS_CODE -b $BATCH_INDEX -supp_id $SUPP_ID -start 1 -stop 2
else
    python 03a_read_delphes.py -p $PROCESS_CODE -b $BATCH_INDEX -start 1 -stop 2
fi