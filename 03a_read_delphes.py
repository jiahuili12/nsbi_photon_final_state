#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import matplotlib
import math
import os

from madminer.delphes import DelphesReader
import argparse

logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.DEBUG
)

for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.DEBUG)

        
import yaml
with open("workflow.yaml", "r") as file:
    workflow = yaml.safe_load(file)
    
    
parser = argparse.ArgumentParser()
parser.add_argument("-p","--process_code",help="Choose from signal_sm, signal_supp, or background")
parser.add_argument("-b","--batch_index",help="batch_index")
parser.add_argument("-supp_id","--supp_id",help="Index of non_SM benchmark that events were generated at")
parser.add_argument("-dr","--delphes_run",action="store_true",help="Whether Delphes has been run on the events or not")
parser.add_argument("-start","--start",help="MadGraph run start index")
parser.add_argument("-stop","--stop",help="Madgraph run stop index")
args = parser.parse_args()
    
mg_dir = workflow["madgraph"]["dir"]
delphes = DelphesReader(workflow["morphing_setup"])
  
if "background" in args.process_code:
    is_background = True
elif "signal" in args.process_code:
    is_background = False
    
if args.process_code != "signal_supp":
    path_to_events_dir = "{input_dir_prefix}/{process_code}/batch_{batch_index}/".format(input_dir_prefix=workflow["delphes"]["input_dir_prefix"], process_code = args.process_code, batch_index=args.batch_index)
    sampled_from_benchmark = "sm"
else:
    path_to_events_dir = "{input_dir_prefix}/{process_code}/mb_vector_{supp_id}/batch_{batch_index}/".format(input_dir_prefix=workflow["delphes"]["input_dir_prefix"], process_code = args.process_code, batch_index=args.batch_index, supp_id = args.supp_id)
    sampled_from_benchmark = f"morphing_basis_vector_{args.supp_id}"
    
    
for run_id in range(int(args.start), int(args.stop)+1):   
    
    # background events have not gone through MadSpin
    if "background" in args.process_code:
        loc_dir = f"{path_to_events_dir}/run_{str(run_id).zfill(2)}"
    else:
        loc_dir = f"{path_to_events_dir}/run_{str(run_id).zfill(2)}_decayed_1"
    
    print(loc_dir)

    if args.delphes_run: 
        delphes.add_sample(lhe_filename=f"{loc_dir}/unweighted_events.lhe.gz",
                            hepmc_filename=f"{loc_dir}/tag_1_pythia8_events.hepmc.gz",
                               delphes_filename=f"{loc_dir}/tag_1_pythia8_events_delphes.root",
                               weights="lhe",
                            sampled_from_benchmark=sampled_from_benchmark,
                            is_background=is_background,
                            k_factor= 1.0,)
    else: 
        delphes.add_sample(lhe_filename=f"{loc_dir}/unweighted_events.lhe.gz",
                            hepmc_filename=f"{loc_dir}/tag_1_pythia8_events.hepmc.gz",
                               weights="lhe",
                            sampled_from_benchmark=sampled_from_benchmark,
                            is_background=is_background,
                            k_factor= 1.0,)

                                                                       
# Now we run Delphes on these samples (you can also do this externally and then add the keyword `delphes_filename` when calling `DelphesReader.add_sample()`):


if args.delphes_run: 
    print("Delphes has already been run.")
else:
    delphes.run_delphes(
        delphes_directory=mg_dir + "/HEPTools/Delphes-3.5.0/", # For latest madgraph version.
        delphes_card="cards/delphes_card_HLLHC.tcl",
        log_file=f"delpheslogs/delphes_{args.process_code}_batch{args.batch_index}{'_mb' + str(args.supp_id) if args.supp_id else ''}.log",
    )

"""
CUSTOM FUNCTIONS TO ISOLATE THE BJETS (HH → 4b)
"""

def get_four_bjets(j):
    # Return the four leading b-tagged jets
    bjets = [jet for jet in j if jet.b_tag == 1]
    if len(bjets) >= 4:
        return True, bjets[:4]
    return False, None

def count_bjets(l,a,j,met):
    return sum(1 for jet in j if jet.b_tag == 1)

def get_b_pt(l,a,j,met,index):
    found, bjets = get_four_bjets(j)
    if found:
        return bjets[index].pt
    return np.nan

def get_b_eta(l,a,j,met,index):
    found, bjets = get_four_bjets(j)
    if found:
        return bjets[index].eta
    return np.nan

def get_bb_deltaR(l,a,j,met,i1,i2):
    found, bjets = get_four_bjets(j)
    if found:
        return bjets[i1].deltaR(bjets[i2])
    return np.nan

def get_mbb(l,a,j,met,i1,i2):
    found, bjets = get_four_bjets(j)
    if found:
        return (bjets[i1] + bjets[i2]).m
    return np.nan

def get_ptbb(l,a,j,met,i1,i2):
    found, bjets = get_four_bjets(j)
    if found:
        return (bjets[i1] + bjets[i2]).pt
    return np.nan

def get_mtot_4b(l,a,j,met):
    found, bjets = get_four_bjets(j)
    if found:
        return sum(bjets, start=bjets[0].__class__(0)).m  # vector sum
    return np.nan


"""
MAIN ANALYSIS
"""

def add_observables(delphes):

    # Number of b-jets
    delphes.add_observable_from_function("num_bjets", count_bjets, required=True)

    # b-jet pT and eta
    for i in range(4):
        delphes.add_observable_from_function(f"b{i}_pt", lambda l,a,j,met,ii=i: get_b_pt(l,a,j,met,ii), required=True)
        delphes.add_observable_from_function(f"b{i}_eta", lambda l,a,j,met,ii=i: get_b_eta(l,a,j,met,ii), required=True)

    # ΔR between Higgs candidate jets
    delphes.add_observable_from_function("bb1_deltaR", lambda l,a,j,met: get_bb_deltaR(l,a,j,met,0,1), required=True)
    delphes.add_observable_from_function("bb2_deltaR", lambda l,a,j,met: get_bb_deltaR(l,a,j,met,2,3), required=True)

    # Invariant masses for Higgs candidates
    delphes.add_observable_from_function("m_bb1", lambda l,a,j,met: get_mbb(l,a,j,met,0,1), required=True)
    delphes.add_observable_from_function("m_bb2", lambda l,a,j,met: get_mbb(l,a,j,met,2,3), required=True)

    # pT of Higgs candidates
    delphes.add_observable_from_function("pt_bb1", lambda l,a,j,met: get_ptbb(l,a,j,met,0,1), required=True)
    delphes.add_observable_from_function("pt_bb2", lambda l,a,j,met: get_ptbb(l,a,j,met,2,3), required=True)

    # Total invariant mass of all 4 b-jets
    delphes.add_observable_from_function("m_tot_4b", get_mtot_4b, required=True)

def add_cuts_and_efficiencies(delphes, region=None):

    # Require ≥4 b-jets
    delphes.add_cut('num_bjets>=4')

    # pT cuts (from trigger plateau)
    delphes.add_cut('b0_pt>35')
    delphes.add_cut('b1_pt>35')
    delphes.add_cut('b2_pt>35')
    delphes.add_cut('b3_pt>30')

    # η cuts
    delphes.add_cut('abs(b0_eta)<2.5')
    delphes.add_cut('abs(b1_eta)<2.5')
    delphes.add_cut('abs(b2_eta)<2.5')
    delphes.add_cut('abs(b3_eta)<2.5')

    # ΔR separation between b’s in each Higgs candidate
    delphes.add_cut('bb1_deltaR>0.4')
    delphes.add_cut('bb2_deltaR>0.4')
    
    # Higgs mass windows
    delphes.add_cut('abs(m_bb1-125)<25')
    delphes.add_cut('abs(m_bb2-125)<25')

add_observables(delphes)
add_cuts_and_efficiencies(delphes)
    
# 4. Run analysis
delphes.analyse_delphes_samples()

# 5. Save results into new .h5 file

if args.process_code != "signal_supp":
    delphes.save("{delphes_output_data}_{process_code}_batch_{batch_index}.h5".format(delphes_output_data=workflow["delphes"]["output_file"], process_code=args.process_code, batch_index=args.batch_index))
else: 
    delphes.save("{delphes_output_data}_{process_code}_{supp_id}_batch_{batch_index}.h5".format(delphes_output_data=workflow["delphes"]["output_file"], process_code=args.process_code, batch_index=args.batch_index, supp_id = args.supp_id))
