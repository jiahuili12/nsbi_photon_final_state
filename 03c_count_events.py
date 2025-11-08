#!/usr/bin/env python3
"""
Script to count events in MadMiner HDF5 datasets.
Usage: python count_events.py <data_directory>

The directory should contain files like:
- delphes_s_shuffled_14TeV.h5 (signal events)
- delphes_b0_shuffled_14TeV.h5 (background events)
"""

import sys
import os
import h5py
import argparse
from pathlib import Path

def count_events_in_file(filepath):
    """Count events in a single HDF5 file."""
    try:
        with h5py.File(filepath, 'r') as f:
            signal_events = f['sample_summary/signal_events'][()]
            background_events = f['sample_summary/background_events'][()]
            
            # Get benchmark names if available
            benchmark_names = []
            if 'benchmarks/names' in f:
                benchmark_names = [name.decode() for name in f['benchmarks/names']]
            
            return signal_events, background_events, benchmark_names
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None, None

def analyze_directory(data_dir):
    """Analyze MadMiner HDF5 files in the directory."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return
    
    # Find signal and background files
    signal_files = list(data_path.glob("delphes_s_shuffled_*.h5"))
    background_files = list(data_path.glob("delphes_b0_shuffled_*.h5"))
    
    if not signal_files and not background_files:
        print(f"No MadMiner HDF5 files found in {data_dir}")
        print("Expected files: delphes_s_shuffled_*.h5 (signal) and delphes_b0_shuffled_*.h5 (background)")
        return
    
    print(f"Analyzing MadMiner HDF5 files in {data_dir}")
    print("=" * 60)
    
    total_signal_events = 0
    total_background_events = 0
    
    # Process signal files
    for filepath in sorted(signal_files):
        print(f"\nSignal file: {filepath.name}")
        print("-" * 40)
        
        signal_events, background_events, benchmark_names = count_events_in_file(filepath)
        
        if signal_events is not None:
            # Print signal events breakdown
            if len(signal_events) > 0 and any(signal_events > 0):
                print("Signal events by benchmark:")
                for i, count in enumerate(signal_events):
                    if count > 0:
                        benchmark_name = benchmark_names[i] if i < len(benchmark_names) else f"benchmark_{i}"
                        print(f"  {benchmark_name}: {count:,} events")
                        total_signal_events += count
                
                print(f"  Total signal in this file: {sum(signal_events):,} events")
            else:
                print("Signal events: 0")
            
            # Print background events (should be 0 for signal files)
            if background_events > 0:
                print(f"Background events: {background_events:,}")
            else:
                print("Background events: 0")
    
    # Process background files
    for filepath in sorted(background_files):
        print(f"\nBackground file: {filepath.name}")
        print("-" * 40)
        
        signal_events, background_events, benchmark_names = count_events_in_file(filepath)
        
        if background_events is not None:
            # Print signal events (should be 0 for background files)
            if len(signal_events) > 0 and any(signal_events > 0):
                print("Signal events: 0 (should be 0 for background files)")
            else:
                print("Signal events: 0")
            
            # Print background events
            if background_events > 0:
                print(f"Background events: {background_events:,}")
                total_background_events += background_events
            else:
                print("Background events: 0")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total signal events: {total_signal_events:,}")
    print(f"Total background events: {total_background_events:,}")
    print(f"Total events: {total_signal_events + total_background_events:,}")

def main():
    parser = argparse.ArgumentParser(description='Count events in MadMiner HDF5 datasets')
    parser.add_argument('data_dir', help='Directory containing MadMiner HDF5 files (delphes_s_shuffled_*.h5 and delphes_b0_shuffled_*.h5)')
    
    args = parser.parse_args()
    
    analyze_directory(args.data_dir)

if __name__ == "__main__":
    main() 