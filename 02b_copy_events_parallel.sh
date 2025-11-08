#!/bin/bash

# Script to copy events from parallel job directories to the correct directory structure for Delphes processing
# Based on the structure expected by 03a_read_delphes.py
# Handles parallel structure: signal_sm_1, signal_sm_2, etc. -> run_01, run_02, etc.

set -e  # Exit on any error

echo "Setting up directory structure for Delphes processing from parallel jobs..."

# Create base directory
FINAL_EVENTS_DIR="/vols/cms/us322/events_4b/02b_final_events_4b"
SOURCE_DIR="/vols/cms/us322/events_4b/02_event_generation_4b"

mkdir -p "$FINAL_EVENTS_DIR"

echo "Creating directory structure..."

# Function to find and copy signal_sm jobs
copy_signal_sm_jobs() {
    echo "Copying signal_sm events from parallel jobs..."
    
    # Find all signal_sm_* directories, sort them numerically
    job_dirs=($(find "$SOURCE_DIR/mg_processes" -maxdepth 1 -name "signal_sm_*" -type d | sort -V))
    total_jobs=${#job_dirs[@]}
    
    echo "Found $total_jobs signal_sm jobs"
    
    # Calculate number of batches needed (2 runs per batch)
    batch_size=2
    num_batches=$(( (total_jobs + batch_size - 1) / batch_size ))
    
    echo "Creating $num_batches batches with $batch_size runs each"
    
    job_count=1
    batch_index=0
    
    for job_dir in "${job_dirs[@]}"; do
        if [ -d "$job_dir" ]; then
            # Calculate which batch this job belongs to
            batch_index=$(( (job_count - 1) / batch_size ))
            run_in_batch=$(( (job_count - 1) % batch_size + 1 ))
            
            mkdir -p "$FINAL_EVENTS_DIR/signal_sm/batch_$batch_index"
            
            echo "Copying from $job_dir to batch_$batch_index/run_$(printf "%02d" $run_in_batch)..."
            cp -r "$job_dir/Events/run_01" "$FINAL_EVENTS_DIR/signal_sm/batch_$batch_index/run_$(printf "%02d" $run_in_batch)"
            cp -r "$job_dir/Events/run_01_decayed_1" "$FINAL_EVENTS_DIR/signal_sm/batch_$batch_index/run_$(printf "%02d" $run_in_batch)_decayed_1"
            ((job_count++))
        fi
    done
    
    echo "Copied $((job_count-1)) signal_sm jobs into $((batch_index+1)) batches"
}

# Function to find and copy background jobs
copy_background_jobs() {
    echo "Copying background events from parallel jobs..."
    
    # Find all background_* directories, sort them numerically
    job_dirs=($(find "$SOURCE_DIR/mg_processes_2" -maxdepth 1 -name "background_*" -type d | sort -V))
    total_jobs=${#job_dirs[@]}
    
    echo "Found $total_jobs background jobs"
    
    # Calculate number of batches needed (2 runs per batch)
    batch_size=2
    num_batches=$(( (total_jobs + batch_size - 1) / batch_size ))
    
    echo "Creating $num_batches batches with $batch_size runs each"
    
    job_count=1
    batch_index=0
    
    for job_dir in "${job_dirs[@]}"; do
        if [ -d "$job_dir" ]; then
            # Calculate which batch this job belongs to
            batch_index=$(( (job_count - 1) / batch_size ))
            run_in_batch=$(( (job_count - 1) % batch_size + 1 ))
            
            mkdir -p "$FINAL_EVENTS_DIR/background/batch_$batch_index"
            
            echo "Copying from $job_dir to batch_$batch_index/run_$(printf "%02d" $run_in_batch)..."
            cp -r "$job_dir/Events/run_01" "$FINAL_EVENTS_DIR/background/batch_$batch_index/run_$(printf "%02d" $run_in_batch)"
            ((job_count++))
        fi
    done
    
    echo "Copied $((job_count-1)) background jobs into $((batch_index+1)) batches"
}

# Function to find and copy BSM jobs
copy_bsm_jobs() {
    echo "Copying signal_supp events from parallel jobs..."
    mkdir -p "$FINAL_EVENTS_DIR/signal_supp"
    
    # For each morphing basis vector (1-9), find all parallel jobs and copy them
    for mb_vector in {1..9}; do
        echo "Processing morphing_basis_vector_$mb_vector..."
        
        # Find all directories for this morphing basis vector, sort them numerically
        job_dirs=($(find "$SOURCE_DIR/mg_processes" -maxdepth 2 -path "*/signal_supp_*/morphing_basis_vector_${mb_vector}" -type d | sort -V))
        total_jobs=${#job_dirs[@]}
        
        if [ $total_jobs -eq 0 ]; then
            echo "No jobs found for morphing_basis_vector_$mb_vector, skipping..."
            continue
        fi
        
        echo "Found $total_jobs jobs for morphing_basis_vector_$mb_vector"
        
        # Calculate number of batches needed (2 runs per batch)
        batch_size=2
        num_batches=$(( (total_jobs + batch_size - 1) / batch_size ))
        
        echo "Creating $num_batches batches with $batch_size runs each"
        
        job_count=1
        batch_index=0
        
        for job_dir in "${job_dirs[@]}"; do
            if [ -d "$job_dir" ]; then
                # Calculate which batch this job belongs to
                batch_index=$(( (job_count - 1) / batch_size ))
                run_in_batch=$(( (job_count - 1) % batch_size + 1 ))
                
                mkdir -p "$FINAL_EVENTS_DIR/signal_supp/mb_vector_$mb_vector/batch_$batch_index"
                
                echo "Copying from $job_dir to batch_$batch_index/run_$(printf "%02d" $run_in_batch)..."
                cp -r "$job_dir/Events/run_01" "$FINAL_EVENTS_DIR/signal_supp/mb_vector_$mb_vector/batch_$batch_index/run_$(printf "%02d" $run_in_batch)"
                cp -r "$job_dir/Events/run_01_decayed_1" "$FINAL_EVENTS_DIR/signal_supp/mb_vector_$mb_vector/batch_$batch_index/run_$(printf "%02d" $run_in_batch)_decayed_1"
                ((job_count++))
            fi
        done
        
        echo "Copied $((job_count-1)) jobs for morphing_basis_vector_$mb_vector into $((batch_index+1)) batches"
    done
}

# Execute the copy functions
copy_signal_sm_jobs
copy_background_jobs
copy_bsm_jobs

echo "Directory structure created successfully!"
echo "Final structure:"
echo "  $FINAL_EVENTS_DIR/signal_sm/batch_X/run_01, run_02, ... (2 runs per batch)"
echo "  $FINAL_EVENTS_DIR/background/batch_X/run_01, run_02, ... (2 runs per batch)"
echo "  $FINAL_EVENTS_DIR/signal_supp/mb_vector_X/batch_Y/run_01, run_02, ... (2 runs per batch)" 