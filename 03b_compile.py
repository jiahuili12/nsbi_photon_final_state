#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import matplotlib
import math

from madminer.sampling import combine_and_shuffle
import argparse

logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.DEBUG
)

for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)
 
import yaml
with open("workflow.yaml", "r") as file:
    workflow = yaml.safe_load(file)
    
parser = argparse.ArgumentParser()
parser.add_argument("-p","--process_code",help="process_code: signal_sm, signal_bsm, or background",default="Choose signal_sm, signal_bsm, or background")
parser.add_argument("-n","--num_batch",help="num_background_batches",default=80,type=int)

args = parser.parse_args()   

# batch indices for the SM benchmark (10 batches: 0-9)
signal_sm_batches = list(range(20))  # 0,1,2,3,4,5,6,7,8,9

# {morphing basis: batch indices} for non-SM benchmarks (10 batches each: 0-9)
signal_bsm_batches = {1:list(range(10)), 2:list(range(10)), 3:list(range(10)), 4:list(range(10)), 5:list(range(10)), 6:list(range(10)), 7:list(range(10)), 8:list(range(10)), 9:list(range(10))}  # 10 batches each for all morphing points

if args.process_code == "signal_sm":
    # Process only SM signal batches
    to_combine = []
    for i in signal_sm_batches:
        to_combine.append('{long_term_storage_dir}/delphes_signal_sm_batch_{batch_num}.h5'.format(long_term_storage_dir=workflow["delphes"]["long_term_storage_dir"], batch_num=i))

    combine_and_shuffle(
        to_combine,
        '{long_term_storage_dir}/delphes_signal_sm_shuffled_14TeV.h5'.format(long_term_storage_dir=workflow["delphes"]["long_term_storage_dir"])
    )

elif args.process_code == "signal_bsm":
    # Process only BSM signal batches
    to_combine = []
    for supp_id in range(1, 10): # signal supp
        for i in signal_bsm_batches[supp_id]:
            to_combine.append('{long_term_storage_dir}/delphes_signal_supp_{supp_id}_batch_{batch_num}.h5'.format(long_term_storage_dir=workflow["delphes"]["long_term_storage_dir"], batch_num=i, supp_id=supp_id))

    combine_and_shuffle(
        to_combine,
        '{long_term_storage_dir}/delphes_signal_bsm_shuffled_14TeV.h5'.format(long_term_storage_dir=workflow["delphes"]["long_term_storage_dir"])
    )

elif args.process_code == "signal_all":
    # Process both SM and BSM signal batches
    to_combine = []
    # Add SM signal batches
    for i in signal_sm_batches:
        to_combine.append('{long_term_storage_dir}/delphes_signal_sm_batch_{batch_num}.h5'.format(long_term_storage_dir=workflow["delphes"]["long_term_storage_dir"], batch_num=i))
    # Add BSM signal batches
    for supp_id in range(1, 10): # signal supp
        for i in signal_bsm_batches[supp_id]:
            to_combine.append('{long_term_storage_dir}/delphes_signal_supp_{supp_id}_batch_{batch_num}.h5'.format(long_term_storage_dir=workflow["delphes"]["long_term_storage_dir"], batch_num=i, supp_id=supp_id))

    combine_and_shuffle(
        to_combine,
        '{long_term_storage_dir}/delphes_s_shuffled_14TeV.h5'.format(long_term_storage_dir=workflow["delphes"]["long_term_storage_dir"])
    )

elif args.process_code == "background": # i.e. background only
    to_combine = []
    k_factors_background = [1 for x in range(args.num_batch)]  

    print(f"Adding in {args.num_batch} batches of background...")
    for i in range(args.num_batch):
        to_combine.append('{long_term_storage_dir}/delphes_background_batch_{batch_num}.h5'.format(long_term_storage_dir=workflow["delphes"]["long_term_storage_dir"], batch_num=i))
    combine_and_shuffle(
        to_combine,
        '{long_term_storage_dir}/delphes_b0_shuffled_14TeV.h5'.format(long_term_storage_dir=workflow["delphes"]["long_term_storage_dir"]),
        k_factors=k_factors_background
    )

else:
    print("Invalid process_code. Choose from: signal_sm, signal_bsm, signal_all, background") 