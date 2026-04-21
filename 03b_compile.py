#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import argparse

from madminer.sampling import combine_and_shuffle

logging.basicConfig(
    format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
    datefmt="%H:%M",
    level=logging.DEBUG
)

for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)


parser = argparse.ArgumentParser()

parser.add_argument(
    "-p",
    "--process_code",
    help="Choose from: signal_sm, signal_bsm, signal_all, background",
    required=True
)

parser.add_argument(
    "-o",
    "--output_dir",
    help="Directory to save combined shuffled h5 files",
    default="/vols/cms/jl3222/nsbi_backups/combined_samples"
)

args = parser.parse_args()


# ============================================================
# Input file paths: set according to your real folder structure
# ============================================================

signal_sm_file = "/vols/cms/jl3222/nsbi_backups/signal_100k/post_delphes/delphes_signal_sm_batch_0.h5"

signal_bsm_files = {
    1: "/vols/cms/jl3222/nsbi_backups/signal_1_100k/delphes_signal_supp_1_batch_0.h5",
    2: "/vols/cms/jl3222/nsbi_backups/signal2_50k/delphes_signal_supp_2_batch_0.h5",
    3: "/vols/cms/jl3222/nsbi_backups/signal3_50k/delphes_signal_supp_3_batch_0.h5",
    4: "/vols/cms/jl3222/nsbi_backups/signal4_50k/delphes_signal_supp_4_batch_0.h5",
    5: "/vols/cms/jl3222/nsbi_backups/signal5_50k/delphes_signal_supp_5_batch_0.h5",
    6: "/vols/cms/jl3222/nsbi_backups/signal6_50k/delphes_signal_supp_6_batch_0.h5",
    7: "/vols/cms/jl3222/nsbi_backups/signal7_50k/delphes_signal_supp_7_batch_0.h5",
    8: "/vols/cms/jl3222/nsbi_backups/signal8_50k/delphes_signal_supp_8_batch_0.h5",
    9: "/vols/cms/jl3222/nsbi_backups/signal9_50k/delphes_signal_supp_9_batch_0.h5",
}

background_file = "/vols/cms/jl3222/nsbi_backups/background_100k/post_delphes/delphes_background_batch_0.h5"


# ============================================================
# Make output directory
# ============================================================

os.makedirs(args.output_dir, exist_ok=True)


# ============================================================
# Check input files exist
# ============================================================

def check_files(file_list):
    missing = [f for f in file_list if not os.path.isfile(f)]
    if missing:
        print("The following input files are missing:")
        for f in missing:
            print(f"  {f}")
        raise FileNotFoundError("Some input files do not exist.")


# ============================================================
# Main logic
# ============================================================

if args.process_code == "signal_sm":
    to_combine = [signal_sm_file]
    check_files(to_combine)

    print("Combining SM signal file:")
    for f in to_combine:
        print(f"  {f}")

    combine_and_shuffle(
        to_combine,
        os.path.join(args.output_dir, "delphes_signal_sm_shuffled_14TeV.h5")
    )

elif args.process_code == "signal_bsm":
    to_combine = [signal_bsm_files[i] for i in range(1, 10)]
    check_files(to_combine)

    print("Combining BSM signal files:")
    for f in to_combine:
        print(f"  {f}")

    combine_and_shuffle(
        to_combine,
        os.path.join(args.output_dir, "delphes_signal_bsm_shuffled_14TeV.h5")
    )

elif args.process_code == "signal_all":
    to_combine = [signal_sm_file] + [signal_bsm_files[i] for i in range(1, 10)]
    check_files(to_combine)

    print("Combining all signal files (SM + BSM):")
    for f in to_combine:
        print(f"  {f}")

    combine_and_shuffle(
        to_combine,
        os.path.join(args.output_dir, "delphes_s_shuffled_14TeV.h5")
    )

elif args.process_code == "background":
    to_combine = [background_file]
    check_files(to_combine)

    print("Combining background file:")
    for f in to_combine:
        print(f"  {f}")

    combine_and_shuffle(
        to_combine,
        os.path.join(args.output_dir, "delphes_b0_shuffled_14TeV.h5"),
        k_factors=[1]
    )

else:
    print("Invalid process_code. Choose from: signal_sm, signal_bsm, signal_all, background")