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
parser.add_argument("-p", "--process_code", help="Choose from signal_sm, signal_supp, or background")
parser.add_argument("-b", "--batch_index", help="batch_index")
parser.add_argument("-supp_id", "--supp_id", help="Index of non_SM benchmark that events were generated at")
parser.add_argument("-dr", "--delphes_run", action="store_true", help="Whether Delphes has been run on the events or not")
parser.add_argument("-start", "--start", help="MadGraph run start index")
parser.add_argument("-stop", "--stop", help="Madgraph run stop index")
args = parser.parse_args()

mg_dir = workflow["madgraph"]["dir"]
delphes = DelphesReader(workflow["morphing_setup"])

if "background" in args.process_code:
    is_background = True
elif "signal" in args.process_code:
    is_background = False

if args.process_code != "signal_supp":
    path_to_events_dir = "{input_dir_prefix}/{process_code}/batch_{batch_index}/".format(
        input_dir_prefix=workflow["delphes"]["input_dir_prefix"],
        process_code=args.process_code,
        batch_index=args.batch_index
    )
    sampled_from_benchmark = "sm"
else:
    path_to_events_dir = "{input_dir_prefix}/{process_code}/mb_vector_{supp_id}/batch_{batch_index}/".format(
        input_dir_prefix=workflow["delphes"]["input_dir_prefix"],
        process_code=args.process_code,
        batch_index=args.batch_index,
        supp_id=args.supp_id
    )
    sampled_from_benchmark = f"morphing_basis_vector_{args.supp_id}"

for run_id in range(int(args.start), int(args.stop) + 1):

    # background events have not gone through MadSpin
    if "background" in args.process_code:
        loc_dir = f"{path_to_events_dir}/run_{str(run_id).zfill(2)}"
    else:
        loc_dir = f"{path_to_events_dir}/run_{str(run_id).zfill(2)}_decayed_1"

    print(loc_dir)

    if args.delphes_run:
        delphes.add_sample(
            lhe_filename=f"{loc_dir}/unweighted_events.lhe.gz",
            hepmc_filename=f"{loc_dir}/tag_1_pythia8_events.hepmc.gz",
            delphes_filename=f"{loc_dir}/tag_1_pythia8_events_delphes.root",
            weights="lhe",
            sampled_from_benchmark=sampled_from_benchmark,
            is_background=is_background,
            k_factor=1.0,
        )
    else:
        delphes.add_sample(
            lhe_filename=f"{loc_dir}/unweighted_events.lhe.gz",
            hepmc_filename=f"{loc_dir}/tag_1_pythia8_events.hepmc.gz",
            weights="lhe",
            sampled_from_benchmark=sampled_from_benchmark,
            is_background=is_background,
            k_factor=1.0,
        )

# Now we run Delphes on these samples (you can also do this externally and then add the keyword
# `delphes_filename` when calling `DelphesReader.add_sample()`):

if args.delphes_run:
    print("Delphes has already been run.")
else:
    delphes.run_delphes(
        delphes_directory=mg_dir + "/HEPTools/Delphes-3.5.0/",  # For latest madgraph version.
        delphes_card="cards/delphes_card_HLLHC.tcl",
        log_file=f"delpheslogs/delphes_{args.process_code}_batch{args.batch_index}{'_mb' + str(args.supp_id) if args.supp_id else ''}.log",
    )

"""
CUSTOM FUNCTIONS TO ISOLATE THE BJETS (HH → 4b)
"""


def get_four_bjets(j):
    # Return the four leading b-tagged jets (highest-pT four b-tagged jets)
    bjets = sorted([jet for jet in j if jet.b_tag == 1], key=lambda x: x.pt, reverse=True)
    if len(bjets) >= 4:
        return True, bjets[:4]
    return False, None


def count_bjets(l, a, j, met):
    return sum(1 for jet in j if jet.b_tag == 1)


def get_b_pt(l, a, j, met, index):
    found, bjets = get_four_bjets(j)
    if found:
        return bjets[index].pt
    return np.nan


def get_b_eta(l, a, j, met, index):
    found, bjets = get_four_bjets(j)
    if found:
        return bjets[index].eta
    return np.nan


def get_b_phi(l, a, j, met, index):
    found, bjets = get_four_bjets(j)
    if found:
        return bjets[index].phi
    return np.nan


def get_b_m(l, a, j, met, index):
    found, bjets = get_four_bjets(j)
    if found:
        return bjets[index].m
    return np.nan


# =========================================================
# OLD PAIRING IDEA (kept, but no longer used)
# =========================================================

# def _dhh_to_point_symmetric(m_h1, m_h2, m1=125.0, m2=120.0):
#     # Distance to the point (m1, m2), but symmetric under swapping the two Higgs candidates
#     d12 = math.sqrt((m_h1 - m1)**2 + (m_h2 - m2)**2)
#     d21 = math.sqrt((m_h1 - m2)**2 + (m_h2 - m1)**2)
#     return min(d12, d21)


# def best_pairing_indices(j, dhh_tie_threshold=30.0):
#     found, bjets = get_four_bjets(j)
#     if not found:
#         return False, None, None, None, None
#
#     pairings = [((0, 1), (2, 3)),
#                 ((0, 2), (1, 3)),
#                 ((0, 3), (1, 2))]
#
#     results = []
#
#     for (a, b), (c, d) in pairings:
#         Hx = bjets[a] + bjets[b]
#         Hy = bjets[c] + bjets[d]
#
#         if Hx.pt >= Hy.pt:
#             H1, H2 = Hx, Hy
#             idx = (a, b, c, d)
#         else:
#             H1, H2 = Hy, Hx
#             idx = (c, d, a, b)
#
#         mH1, mH2 = H1.m, H2.m
#         pTH1, pTH2 = H1.pt, H2.pt
#         dhh = _dhh_to_point_symmetric(mH1, mH2, m1=125.0, m2=120.0)
#         results.append((dhh, pTH1, pTH2, idx, mH1, mH2))
#
#     results.sort(key=lambda x: x[0])
#     best = results[0]
#     second = results[1]
#
#     if abs(best[0] - second[0]) < dhh_tie_threshold:
#         chosen = best if best[1] >= second[1] else second
#     else:
#         chosen = best
#
#     dhh_best, pTH1_best, pTH2_best, idx_best, mH1_best, mH2_best = chosen
#     return True, idx_best, (pTH1_best, pTH2_best), (mH1_best, mH2_best), dhh_best


# =========================================================
# NEW PAIRING: min-DeltaR baseline for non-resonant 4b
# =========================================================

def best_pairing_indices(j):
    """
    Min-DeltaR pairing baseline:
    - take the four highest-pT b-tagged jets
    - form the 3 possible pairings
    - define H1 as the candidate with the higher scalar sum of jet pT
    - choose the pairing whose H1 pair has the smallest DeltaR
    - if tied, use the subleading pair DeltaR as tie-breaker
    Return:
      (found,
       (iH1a, iH1b, iH2a, iH2b),
       (ptH1, ptH2),
       (mH1, mH2),
       (drH1, drH2),
       (etaH1, etaH2),
       m4j)
    """
    found, bjets = get_four_bjets(j)
    if not found:
        return False, None, None, None, None, None, None

    pairings = [
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    ]

    results = []

    for (a, b), (c, d) in pairings:
        Hx = bjets[a] + bjets[b]
        Hy = bjets[c] + bjets[d]

        # Use scalar sum of jet pT to define leading/subleading Higgs candidate
        sumpt_x = bjets[a].pt + bjets[b].pt
        sumpt_y = bjets[c].pt + bjets[d].pt

        dr_x = bjets[a].deltaR(bjets[b])
        dr_y = bjets[c].deltaR(bjets[d])

        if sumpt_x >= sumpt_y:
            idx = (a, b, c, d)
            H1, H2 = Hx, Hy
            dr1, dr2 = dr_x, dr_y
        else:
            idx = (c, d, a, b)
            H1, H2 = Hy, Hx
            dr1, dr2 = dr_y, dr_x

        pt1, pt2 = H1.pt, H2.pt
        m1, m2 = H1.m, H2.m
        eta1, eta2 = H1.eta, H2.eta
        m4j = (bjets[0] + bjets[1] + bjets[2] + bjets[3]).m

        # min-DeltaR baseline: smallest DeltaR of the leading Higgs candidate pair
        results.append((dr1, dr2, idx, (pt1, pt2), (m1, m2), (eta1, eta2), m4j))

    results.sort(key=lambda x: (x[0], x[1]))
    best = results[0]

    dr1_best, dr2_best, idx_best, pts_best, masses_best, etas_best, m4j_best = best

    return True, idx_best, pts_best, masses_best, (dr1_best, dr2_best), etas_best, m4j_best


def get_bb_deltaR_best(l, a, j, met, which=1):
    found, _, _, _, drs, _, _ = best_pairing_indices(j)
    if not found:
        return np.nan
    if which == 1:
        return drs[0]
    return drs[1]


def get_mbb_best(l, a, j, met, which=1):
    found, _, _, masses, _, _, _ = best_pairing_indices(j)
    if not found:
        return np.nan
    if which == 1:
        return masses[0]
    return masses[1]


def get_ptbb_best(l, a, j, met, which=1):
    found, _, pts, _, _, _, _ = best_pairing_indices(j)
    if not found:
        return np.nan
    if which == 1:
        return pts[0]
    return pts[1]


def get_deltaeta_hh_best(l, a, j, met):
    found, _, _, _, _, etas, _ = best_pairing_indices(j)
    if not found:
        return np.nan
    return abs(etas[0] - etas[1])


def get_xhh_best(l, a, j, met):
    """
    Xhh = sqrt(((mh1-120)/(0.1*mh1))^2 + ((mh2-110)/(0.1*mh2))^2)
    The 120 GeV / 110 GeV reference masses account for detector-level energy loss.
    """
    found, _, _, masses, _, _, _ = best_pairing_indices(j)
    if not found:
        return np.nan

    mh1, mh2 = masses

    if mh1 <= 0 or mh2 <= 0:
        return np.nan

    return math.sqrt(((mh1 - 120.0) / (0.1 * mh1))**2 + ((mh2 - 110.0) / (0.1 * mh2))**2)


def get_mtot_4b(l, a, j, met):
    found, bjets = get_four_bjets(j)
    if found:
        return (bjets[0] + bjets[1] + bjets[2] + bjets[3]).m
    return np.nan


def get_pTH1_best(l, a, j, met):
    found, _, pts, _, _, _, _ = best_pairing_indices(j)
    if not found:
        return np.nan
    return pts[0]


def get_pTH2_best(l, a, j, met):
    found, _, pts, _, _, _, _ = best_pairing_indices(j)
    if not found:
        return np.nan
    return pts[1]


"""
MAIN ANALYSIS
"""


def add_observables(delphes):

    # Number of b-jets
    delphes.add_observable_from_function("num_bjets", count_bjets, required=True)

    # b-jet pT and eta (four highest-pT b-tagged jets)
    for i in range(4):
        delphes.add_observable_from_function(
            f"b{i}_pt",
            lambda l, a, j, met, ii=i: get_b_pt(l, a, j, met, ii),
            required=True
        )
        delphes.add_observable_from_function(
            f"b{i}_eta",
            lambda l, a, j, met, ii=i: get_b_eta(l, a, j, met, ii),
            required=True
        )
        delphes.add_observable_from_function(
            f"b{i}_phi",
            lambda l, a, j, met, ii=i: get_b_phi(l, a, j, met, ii),
            required=True
        )
        delphes.add_observable_from_function(
            f"b{i}_m",
            lambda l, a, j, met, ii=i: get_b_m(l, a, j, met, ii),
            required=False
        )

    # DeltaR between Higgs candidate jets (min-DeltaR pairing)
    delphes.add_observable_from_function(
        "bb1_deltaR",
        lambda l, a, j, met: get_bb_deltaR_best(l, a, j, met, which=1),
        required=True
    )
    delphes.add_observable_from_function(
        "bb2_deltaR",
        lambda l, a, j, met: get_bb_deltaR_best(l, a, j, met, which=2),
        required=True
    )

    # Invariant masses for Higgs candidates
    delphes.add_observable_from_function(
        "m_bb1",
        lambda l, a, j, met: get_mbb_best(l, a, j, met, which=1),
        required=True
    )
    delphes.add_observable_from_function(
        "m_bb2",
        lambda l, a, j, met: get_mbb_best(l, a, j, met, which=2),
        required=True
    )

    # pT of Higgs candidates
    delphes.add_observable_from_function(
        "pt_bb1",
        lambda l, a, j, met: get_ptbb_best(l, a, j, met, which=1),
        required=False
    )
    delphes.add_observable_from_function(
        "pt_bb2",
        lambda l, a, j, met: get_ptbb_best(l, a, j, met, which=2),
        required=False
    )

    # Extra observables for non-resonant baseline selection
    delphes.add_observable_from_function("deltaeta_hh", get_deltaeta_hh_best, required=True)
    delphes.add_observable_from_function("xhh", get_xhh_best, required=True)

    # Optional debugging / plotting observables
    # delphes.add_observable_from_function("dhh_asym", get_dhh_asym_best, required=False)
    delphes.add_observable_from_function("m_tot_4b", get_mtot_4b, required=False)
    # delphes.add_observable_from_function("pt_H1", get_pTH1_best, required=False)
    # delphes.add_observable_from_function("pt_H2", get_pTH2_best, required=False)


def add_cuts_and_efficiencies(delphes, region=None):

    # =========================================================
    # OLD baseline cuts (kept as comments)
    # =========================================================

    # Require ≥4 b-jets
    # delphes.add_cut('num_bjets>=4')

    # pT cuts (from trigger plateau)
    # delphes.add_cut('b0_pt>25')
    # delphes.add_cut('b1_pt>25')
    # delphes.add_cut('b2_pt>25')
    # delphes.add_cut('b3_pt>20')

    # eta cuts
    # delphes.add_cut('abs(b0_eta)<2.5')
    # delphes.add_cut('abs(b1_eta)<2.5')
    # delphes.add_cut('abs(b2_eta)<2.5')
    # delphes.add_cut('abs(b3_eta)<2.5')

    # DeltaR separation between b’s in each Higgs candidate
    # delphes.add_cut('bb1_deltaR>0.4')
    # delphes.add_cut('bb2_deltaR>0.4')

    # Higgs mass windows
    # delphes.add_cut('abs(m_bb1-125)<25')
    # delphes.add_cut('abs(m_bb2-125)<25')

    # =========================================================
    # NEW non-resonant 4b baseline cuts
    # =========================================================

    # Four-tag cut:
    # at least four b-tagged jets, and the four selected b-jets satisfy pT > 40 GeV and |eta| < 2.5
    delphes.add_cut('num_bjets>=4')
    delphes.add_cut('b0_pt>40')
    delphes.add_cut('b1_pt>40')
    delphes.add_cut('b2_pt>40')
    delphes.add_cut('b3_pt>40')

    delphes.add_cut('abs(b0_eta)<2.5')
    delphes.add_cut('abs(b1_eta)<2.5')
    delphes.add_cut('abs(b2_eta)<2.5')
    delphes.add_cut('abs(b3_eta)<2.5')

    # Non-resonant analysis cut
    delphes.add_cut('deltaeta_hh<1.5')

    # Signal-region style cut accounting for detector-level energy loss 
    #  delphes.add_cut('xhh<1.6')

    # =========================================================
    # Not added here on purpose
    # =========================================================

    # No top veto for now:
    # the paper includes it to suppress tt background,
    # but your current generated background is only QCD pp -> b b~ b b~.

    # No resonant-analysis pT(h1), pT(h2), or DeltaR+min-Dhh cuts:
    # here we follow the non-resonant min-DeltaR baseline instead.


add_observables(delphes)
add_cuts_and_efficiencies(delphes)

# 4. Run analysis
delphes.analyse_delphes_samples()

# 5. Save results into new .h5 file
if args.process_code != "signal_supp":
    delphes.save(
        "{delphes_output_data}_{process_code}_batch_{batch_index}.h5".format(
            delphes_output_data=workflow["delphes"]["output_file"],
            process_code=args.process_code,
            batch_index=args.batch_index
        )
    )
else:
    delphes.save(
        "{delphes_output_data}_{process_code}_{supp_id}_batch_{batch_index}.h5".format(
            delphes_output_data=workflow["delphes"]["output_file"],
            process_code=args.process_code,
            batch_index=args.batch_index,
            supp_id=args.supp_id
        )
    )





