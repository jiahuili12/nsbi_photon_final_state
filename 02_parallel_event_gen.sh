#!/bin/bash

# Fix the mozilla directory creation error by setting HOME to a writable location
export HOME="$TMPDIR"

# Set environment variables to prevent display/browser errors in headless environments
export MOZILLA_HOME="$TMPDIR/.mozilla"

# Create the mozilla directory in the temp directory if it doesn't exist
mkdir -p "$MOZILLA_HOME" 2>/dev/null || true

# Use Condor's scratch directory for temporary files
SCRATCH_DIR="${_CONDOR_SCRATCH_DIR:-$TMPDIR}"
echo "Using scratch directory: $SCRATCH_DIR"

# Store the original working directory (where the job is submitted from)
ORIGINAL_DIR=$(pwd)
echo "Original working directory: $ORIGINAL_DIR"

# Extract MadGraph installation to scratch directory
echo "Extracting MadGraph installation..."
cd "$SCRATCH_DIR"
tar -xzf "$ORIGINAL_DIR/madgraph.tar.gz"
export MADGRAPH_DIR="$SCRATCH_DIR/mg5amcnlo"
echo "MadGraph extracted to: $MADGRAPH_DIR"

# Return to original directory for running the Python script
cd "$ORIGINAL_DIR"
echo "Returned to working directory: $(pwd)"

eval "$(/home/hep/us322/miniforge3/bin/conda shell.bash hook)"
conda activate madminer_env

# Parse command line arguments
GENERATION_TYPE=$1

# For BSM jobs, get supp_id and queue_line from command line arguments
if [ "$GENERATION_TYPE" = "bsm" ]; then
    SUPP_ID=$2
    QUEUE_LINE=$3
    echo "BSM job: using supp_id from command line argument: $SUPP_ID, queue_line: $QUEUE_LINE"
else
    SUPP_ID=$2
fi

# Generate a unique job ID based on queue line number
# Use CLUSTER to identify the queue line, PROCESS for individual job within that line
if [ -n "$CLUSTER" ]; then
    # For BSM jobs, we want each queue line to create one folder
    if [ "$GENERATION_TYPE" = "bsm" ]; then
        # Use the queue_line argument directly as job ID
        JOB_ID=$QUEUE_LINE
    else
        # For other jobs, use PROCESS + 1 (PROCESS starts from 0)
        JOB_ID=$((PROCESS + 1))
    fi
else
    JOB_ID=1
fi
echo "Generated job ID: $JOB_ID (CLUSTER=$CLUSTER, PROCESS=$PROCESS)"

# Use Condor's scratch directory for temporary files
TEMP_DIR="$SCRATCH_DIR"
echo "Using Condor scratch directory: $TEMP_DIR"

# Function to generate random seed
generate_random_seed() {
    # Generate a random number between 1 and 999999
    echo $((RANDOM % 999999 + 1))
}

# Function to modify run card with random seed
modify_run_card() {
    local run_card_path=$1
    local temp_run_card_path=$2
    local random_seed=$3
    
    # Copy the original run card to temp directory
    cp "$run_card_path" "$temp_run_card_path"
    
    # Replace XXX with the random seed
    sed -i "s/XXX/${random_seed}/g" "$temp_run_card_path"
    
    echo "Modified run card with seed: $random_seed"
}

# No cleanup needed - Condor handles temporary directory cleanup automatically

# Run the event generation script based on the generation type
case $GENERATION_TYPE in
    "signal")
        echo "Running signal generation (-sm) with random seed"
        
        # Use 14TeV signal run card
        RUN_CARD="./cards/run_cards/run_card_signal_14TeV.dat"
        TEMP_RUN_CARD="$TEMP_DIR/run_card_signal.dat"
        RANDOM_SEED=$(generate_random_seed)
        modify_run_card "$RUN_CARD" "$TEMP_RUN_CARD" "$RANDOM_SEED"
        
        # Run with modified run card and extracted MadGraph directory
        MADGRAPH_DIR="$MADGRAPH_DIR" python 02_generate_events_parallel.py -sm -run_card "$TEMP_RUN_CARD" -job_id "$JOB_ID"
        ;;
        
    "background")
        echo "Running background generation (-b) with random seed"
        
        # Use 14TeV background run card
        RUN_CARD="./cards/run_cards/run_card_background_14TeV.dat"
        TEMP_RUN_CARD="$TEMP_DIR/run_card_background.dat"
        RANDOM_SEED=$(generate_random_seed)
        modify_run_card "$RUN_CARD" "$TEMP_RUN_CARD" "$RANDOM_SEED"
        
        # Run with modified run card and extracted MadGraph directory
        MADGRAPH_DIR="$MADGRAPH_DIR" python 02_generate_events_parallel.py -b -run_card "$TEMP_RUN_CARD" -job_id "$JOB_ID"
        ;;
        
    "bsm")
        if [ -z "$SUPP_ID" ]; then
            echo "Error: BSM generation requires a supp_id (1-9)"
            exit 1
        fi
        echo "Running BSM generation with supp_id $SUPP_ID and random seed"
        
        # Use 14TeV signal run card for BSM (same as signal generation)
        RUN_CARD="./cards/run_cards/run_card_signal_14TeV.dat"
        TEMP_RUN_CARD="$TEMP_DIR/run_card_signal.dat"
        RANDOM_SEED=$(generate_random_seed)
        modify_run_card "$RUN_CARD" "$TEMP_RUN_CARD" "$RANDOM_SEED"
        
        # Run with modified run card and extracted MadGraph directory
        MADGRAPH_DIR="$MADGRAPH_DIR" python 02_generate_events_parallel.py -supp -supp_id $SUPP_ID -run_card "$TEMP_RUN_CARD" -job_id "$JOB_ID"
        ;;
        
    *)
        echo "Error: Unknown generation type: $GENERATION_TYPE"
        exit 1
        ;;
esac

echo "Job completed successfully" 