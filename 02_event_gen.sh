#!/bin/bash

eval "$(/home/hep/us322/miniforge3/bin/conda shell.bash hook)"
conda activate madminer_env
#export TMPDIR=/vols/cms/us322/tmp

# Parse command line arguments
GENERATION_TYPE=$1
SUPP_ID=$2

# Run the event generation script based on the generation type
case $GENERATION_TYPE in
    "signal")
        echo "Running signal generation (-sm)"
        python 02_generate_events.py -sm
        ;;
    "background")
        echo "Running background generation (-b)"
        python 02_generate_events.py -b
        ;;
    "bsm")
        if [ -z "$SUPP_ID" ]; then
            echo "Error: BSM generation requires a supp_id (1-9)"
            exit 1
        fi
        echo "Running BSM generation with supp_id $SUPP_ID"
        python 02_generate_events.py -supp -supp_id $SUPP_ID
        ;;
    "bsm_loop")
        echo "Running BSM generation loop for supp_id 1-9"
        for i in {1..9}; do
            echo "Running BSM generation with supp_id $i"
            python 02_generate_events.py -supp -supp_id $i
        done
        ;;
    *)
        exit 1
        ;;
esac