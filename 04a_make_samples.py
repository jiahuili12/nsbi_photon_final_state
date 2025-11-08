#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import argparse
import sys

from madminer.sampling import SampleAugmenter
from madminer import sampling

# MadMiner output
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.INFO
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)

import yaml
with open("workflow.yaml", "r") as file:
    workflow = yaml.safe_load(file)

data_input_dir = workflow["sampling"]["input_dir"]
samples_output_dir = workflow["sampling"]["output_dir"]

def get_test_set_codes(parameter_code):
    """Get test set codes based on parameter code"""
    if parameter_code == "c1":
        test_set_codes = {
            "m20": (0, -20, 0),
            "m16": (0, -16, 0),
            "m12": (0, -12, 0),
            "m8": (0, -8, 0),
            "m4": (0, -4, 0),
            "p4": (0, 4, 0),
            "p8": (0, 8, 0),
            "p12": (0, 12, 0),
            "p16": (0, 16, 0),
        }
    elif parameter_code == "c0":
        test_set_codes = {
            "m12": (-12, 0, 0),
            "m10": (-10, 0, 0),
            "m8": (-8, 0, 0),
            "m6": (-6, 0, 0),
            "m4": (-4, 0, 0),
            "m2": (-2, 0, 0),
            "p2": (2, 0, 0),
            "p1": (1, 0, 0),
        }
    elif parameter_code == "c0c1":
        test_set_codes = {
            "m10p2p0": (-10, 2, 0),
            "p3m2p0": (3, -2, 0),
            "m4p1p0": (-4, 1, 0),
        }
    elif parameter_code == "c0c2":
        test_set_codes = {
            "m10p0p3": (-10, 0, 3),
            "p3p0m2": (3, 0, -2),
            "m4p0p3": (-4, 0, 3),
        }
    elif parameter_code == "c1c2":
        test_set_codes = {
            "p0m2p2": (0, -2, 2),
            "p0m3p1": (0, -3, 1),
            "p0m1p3": (0, -1, 3),
        }
    elif parameter_code == "c2":
        test_set_codes = {
            "m12": (0, 0, -12),
            "m10": (0, 0, -10),
            "m8": (0, 0, -8),
            "m6": (0, 0, -6),
            "m4": (0, 0, -4),
            "m2": (0, 0, -2),
            "p2": (0, 0, 2),
            "p4": (0, 0, 4),
            "p6": (0, 0, 6),
            "p8": (0, 0, 8),
        }
    else:
        raise ValueError(f"Unknown parameter_code: {parameter_code}")
    
    return test_set_codes

def main():
    parser = argparse.ArgumentParser(description='Generate samples for MadMiner analysis')
    parser.add_argument('parameter_code', help='Parameter code (c0, c1, c2, c0c1, c0c2, c1c2)')
    parser.add_argument('--test-split', type=float, default=0.14, help='Test split fraction (default: 0.14)')
    parser.add_argument('--n-samples', type=int, default=10000000, help='Number of samples (default: 10000000)')
    parser.add_argument('--n-test-samples', type=int, default=10000, help='Number of test samples (default: 10000)')
    parser.add_argument('--n-processes', type=int, default=16, help='Number of processes (default: 16)')
    
    args = parser.parse_args()
    
    parameter_code = args.parameter_code
    test_split = args.test_split
    
    print(f"Processing parameter code: {parameter_code}")
    print(f"Test split: {test_split}")
    
    # Get test set codes
    test_set_codes = get_test_set_codes(parameter_code)
    print(f"Test set codes: {list(test_set_codes.keys())}")
    
    printed_codes = []
    for c in test_set_codes.keys():
        printed_codes.append([test_set_codes[c][0]/10.0, test_set_codes[c][1]/10.0, test_set_codes[c][2]/10.0])
    print(f"Parameter values: {printed_codes}")
    
    # Signal Events
    print("\n" + "="*50)
    print("Processing Signal Events")
    print("="*50)
    
    sampler = SampleAugmenter(f'{data_input_dir}/delphes_s_shuffled_14TeV.h5')
    
    # Alternative training set
    print("Generating alternative training set...")
    x, theta, n_effective = sampler.sample_train_plain(
        theta=sampling.random_morphing_points(1000, [("flat", -14, 6), ("flat", -4, 5), ("flat", -5, 7)]),
        n_samples=args.n_samples,
        folder=f'{samples_output_dir}/plain_real/delphes_s/{parameter_code}',
        filename=f"alt_{parameter_code}",
        sample_only_from_closest_benchmark=True,
        n_processes=args.n_processes,
        validation_split=0.0,
        test_split=test_split
    )
    
    # Alternative test sets
    print("Generating alternative test sets...")
    for code in test_set_codes.keys():
        print(f"  Generating test set for {code}...")
        _ = sampler.sample_test(
            theta=sampling.morphing_point(test_set_codes[code]),
            n_samples=args.n_test_samples,
            folder=f'{samples_output_dir}/plain_real/delphes_s/{parameter_code}',
            filename=f"alt_{parameter_code}_{code}_test",
            sample_only_from_closest_benchmark=True,
            validation_split=0.0,
            test_split=test_split
        )
    
    # SM training set
    print("Generating SM training set...")
    x, theta, n_effective = sampler.sample_train_plain(
        theta=sampling.benchmark("sm"),
        n_samples=args.n_samples,
        folder=f'{samples_output_dir}/plain_real/delphes_s/{parameter_code}',
        filename="sm",
        sample_only_from_closest_benchmark=True,
        n_processes=1,
        validation_split=0.0,
        test_split=test_split
    )
    
    # SM test set
    print("Generating SM test set...")
    _ = sampler.sample_test(
        theta=sampling.benchmark("sm"),
        n_samples=100000,
        folder=f'{samples_output_dir}/plain_real/delphes_s/{parameter_code}',
        filename=f"sm_test",
        sample_only_from_closest_benchmark=True,
        validation_split=0.0,
        test_split=test_split
    )
    
    # Background Events
    print("\n" + "="*50)
    print("Processing Background Events")
    print("="*50)
    
    sampler = SampleAugmenter(f'{data_input_dir}/delphes_b0_shuffled_14TeV.h5')
    
    # Background training set
    print("Generating background training set...")
    x, theta, n_effective = sampler.sample_train_plain(
        theta=sampling.benchmark("sm"),
        n_samples=args.n_samples,
        folder=f'{samples_output_dir}/plain_real/delphes_b0/{parameter_code}',
        filename="bkg",
        sample_only_from_closest_benchmark=True,
        n_processes=1,
        validation_split=0.0,
        test_split=test_split
    )
    
    # Background test set
    print("Generating background test set...")
    _ = sampler.sample_test(
        theta=sampling.benchmark("sm"),
        n_samples=100000,
        folder=f'{samples_output_dir}/plain_real/delphes_b0/{parameter_code}',
        filename=f"bkg_test",
        sample_only_from_closest_benchmark=True,
        validation_split=0.0,
        test_split=test_split
    )
    
    print("\n" + "="*50)
    print("Sample generation complete!")
    print("="*50)

if __name__ == "__main__":
    main() 