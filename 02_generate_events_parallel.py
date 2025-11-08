#!/usr/bin/env python
# coding: utf-8
import os
import logging
import numpy as np
from madminer.core import MadMiner
import argparse

#os.environ["TMPDIR"] = "/vols/cms/us322/tmp"

# MadMiner output
logging.basicConfig(
    format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
    datefmt="%H:%M",
    level=logging.DEBUG,
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)
        
import yaml
working_dir = "./" # CHANGE THIS LINE
with open(f"{working_dir}/workflow.yaml", "r") as file:
    workflow = yaml.safe_load(file)
    
parser = argparse.ArgumentParser()
parser.add_argument("-sm",action="store_true",help="Generate events only at the SM benchmark")
parser.add_argument("-supp",action="store_true",help="Generate events at a non-SM benchmark")
parser.add_argument("-supp_id",help="Index of non_SM benchmark to generate events")
parser.add_argument("-b",action="store_true",help="Generate background events (no reweighting needed)")
parser.add_argument("-run_card",help="Path to custom run card file (optional)")
parser.add_argument("-job_id",help="Unique job ID for parallel runs (optional)")

args = parser.parse_args()

mg_dir = os.environ.get('MADGRAPH_DIR', workflow["madgraph"]["dir"])

# Create unique output directories for parallel jobs
if args.job_id:
    job_suffix = f"_{args.job_id}"
else:
    job_suffix = ""

"""
GENERATE EVENTS
"""
# Always set n_runs = 1
n_runs = 1

energy = int(workflow["madgraph"]["energy"])

miner = MadMiner()
miner.load(workflow["morphing_setup"])

# Use custom run card if provided, otherwise use default 14TeV run cards
if args.run_card:
    if args.sm or args.supp:
        run_cards_signal = [args.run_card]
    if args.b:
        run_cards_background = [args.run_card]
else:
    # Default to 14TeV run cards
    run_cards_signal = [f"{working_dir}/cards/run_cards/run_card_signal_14TeV.dat"]
    run_cards_background = [f"{working_dir}/cards/run_cards/run_card_background_14TeV.dat"]

print(f"\nRunning at energy", workflow["madgraph"]["energy"], "TeV")
print(f"Number of runs: {n_runs}")

if args.sm:
    miner.run_multiple(
        sample_benchmarks=["sm"],
        mg_directory=mg_dir,
        mg_process_directory="{mg_process_output_dir}/signal_sm{job_suffix}".format(mg_process_output_dir = workflow["madgraph"]["output_dir"], job_suffix = job_suffix),
        proc_card_file=f"{working_dir}/cards/proc_card_signal.dat",
        param_card_template_file=f"{working_dir}/cards/restrict_LO.dat",
        madspin_card_file=f"{working_dir}/cards/madspin_card.dat",
        run_card_files=run_cards_signal,
        pythia8_card_file=f"{working_dir}/cards/pythia8_card.dat", 
        log_directory=f"{working_dir}/logs/signal_sm{job_suffix}",
        #python_executable="python3",
        order="LO",
        #systematics=["signal_norm"]
    )

if args.supp:
    miner.run_multiple(
        sample_benchmarks=[f"morphing_basis_vector_{args.supp_id}"],
        mg_directory=mg_dir,
        mg_process_directory="{mg_process_output_dir}/signal_supp{job_suffix}/morphing_basis_vector_{supp_id}".format(mg_process_output_dir = workflow["madgraph"]["output_dir"], supp_id = args.supp_id, job_suffix = job_suffix),
        proc_card_file=f"{working_dir}/cards/proc_card_signal.dat",
        param_card_template_file=f"{working_dir}/cards/restrict_LO.dat",
        madspin_card_file=f"{working_dir}/cards/madspin_card.dat",
        run_card_files=run_cards_signal,
        pythia8_card_file=f"{working_dir}/cards/pythia8_card.dat", 
        log_directory=f"{working_dir}/logs/signal_supp{job_suffix}/morphing_basis_vector_{args.supp_id}",
        #python_executable="python3",
        order="LO",
        #systematics=["signal_norm"]
    )

if args.b:
    for i in range(1):
        miner.run_multiple(
            is_background=True,
            sample_benchmarks=["sm"],
            mg_directory=mg_dir,
            mg_process_directory="{mg_process_output_dir}_2/background{job_suffix}".format(mg_process_output_dir = workflow["madgraph"]["output_dir"], job_suffix = job_suffix),
            proc_card_file="{working_dir}/cards/proc_card_background.dat".format(working_dir = working_dir),
            param_card_template_file=f"{working_dir}/cards/restrict_LO.dat",
            pythia8_card_file=f"{working_dir}/cards/pythia8_card.dat", 
            run_card_files=run_cards_background,
            log_directory=f"{working_dir}/logs_2/background{job_suffix}",
        )