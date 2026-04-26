#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse
import pickle
import yaml
import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from helpers.network_training import NeuralNet, BNN, train_network
from helpers.utils import np_to_torch


parser = argparse.ArgumentParser()

parser.add_argument("-p", "--parameter_code", required=True)
parser.add_argument("-dtype", "--dtype", default="delphes_s")
parser.add_argument("-rid", "--run_id", required=True)
parser.add_argument("-n", "--network", default="dense")
parser.add_argument("-c1", action="store_true")
parser.add_argument("-c2", action="store_true")
parser.add_argument("-c3", action="store_true")
parser.add_argument("-s", "--seed", type=int, default=0)
parser.add_argument("--max_train", type=int, default=None)

parser.add_argument(
    "--feature_mode",
    choices=["m4b", "pt_h", "dr_bb", "hh_angles"],
    required=True,
)

args = parser.parse_args()


torch.set_num_threads(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, flush=True)


with open("workflow.yaml", "r") as file:
    workflow = yaml.safe_load(file)

run_id = args.run_id
seed = int(args.seed)

observable_name_map = {
    17: "bb1_deltaR",
    18: "bb2_deltaR",
    21: "pt_bb1",
    24: "m_tot_4b",
    25: "pt_H1",
    26: "pt_H2",
    27: "deltaeta_hh",
    30: "dphi_hh",
}

feature_sets = {
    "m4b": [24],
    "pt_h": [25, 26],
    "dr_bb": [27],  #deltaeta_hh
    "hh_angles": [30], #deltaphihh
} 

selected_features = feature_sets[args.feature_mode]

model_dir = f"models/{args.feature_mode}"
config_dir = f"run_configs/{args.feature_mode}"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(config_dir, exist_ok=True)

run_configs = {}
run_configs["input_precode"] = args.dtype
run_configs["parameter_code"] = args.parameter_code
run_configs["network_id"] = run_id
run_configs["seed"] = seed
run_configs["feature_mode"] = args.feature_mode
run_configs["features"] = selected_features
run_configs["feature_names"] = [observable_name_map[i] for i in selected_features]

run_configs["network.type"] = args.network
run_configs["network.layers"] = [32, 32]

run_configs["hyperparam.batch_size"] = 1024
run_configs["hyperparam.lr"] = 0.001
run_configs["hyperparam.n_epochs"] = 150
run_configs["hyperparam.patience_ES"] = 20
run_configs["hyperparam.patience_lr"] = 5

with open(f"{config_dir}/{run_id}.yml", "w") as outfile:
    yaml.dump(run_configs, outfile, default_flow_style=False)

print("Feature mode:", args.feature_mode)
print("Selected feature indices:", selected_features)
print("Selected feature names:", run_configs["feature_names"])


samples_dir = workflow["sampling"]["output_dir"]
identity_code = args.dtype
parameter_code = args.parameter_code
features = selected_features

signal_dir = f"{samples_dir}/plain_real/{identity_code}/{parameter_code}"
background_dir = f"{samples_dir}/plain_real/delphes_b0/{parameter_code}"

x_sm_path = f"{signal_dir}/x_sm.npy"
x_alt_path = f"{signal_dir}/x_alt_{parameter_code}.npy"
theta_alt_path = f"{signal_dir}/theta_alt_{parameter_code}.npy"
x_bkg_path = f"{background_dir}/x_bkg.npy"

for file_path in [x_sm_path, x_alt_path, theta_alt_path]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing required file: {file_path}")

if args.c2 or args.c3:
    if not os.path.exists(x_bkg_path):
        raise FileNotFoundError(f"Missing background file: {x_bkg_path}")


samples_SM = np.load(x_sm_path)[:, features]
samples_alt = np.load(x_alt_path)[:, features]
theta_alt = np.load(theta_alt_path)
theta_alt_sm = np.load(theta_alt_path)

samples_bkg = None
if os.path.exists(x_bkg_path):
    samples_bkg = np.load(x_bkg_path)[:, features]

if theta_alt.ndim == 1:
    theta_alt = theta_alt.reshape(-1, 1)

if theta_alt_sm.ndim == 1:
    theta_alt_sm = theta_alt_sm.reshape(-1, 1)


n_candidates = [
    samples_SM.shape[0],
    samples_alt.shape[0],
    theta_alt.shape[0],
    theta_alt_sm.shape[0],
]

if samples_bkg is not None:
    n_candidates.append(samples_bkg.shape[0])

N_train = min(n_candidates)

if args.max_train is not None:
    N_train = min(N_train, int(args.max_train))

samples_SM = samples_SM[:N_train]
samples_alt = samples_alt[:N_train]
theta_alt = theta_alt[:N_train]
theta_alt_sm = theta_alt_sm[:N_train]

if samples_bkg is not None:
    samples_bkg = samples_bkg[:N_train]

print("Using N_train =", N_train)
print("samples_SM shape  :", samples_SM.shape)
print("samples_alt shape :", samples_alt.shape)
print("theta_alt shape   :", theta_alt.shape)

if samples_bkg is not None:
    print("samples_bkg shape :", samples_bkg.shape)


samples_alt, theta_alt, theta_alt_sm = shuffle(samples_alt, theta_alt, theta_alt_sm, random_state=seed)
samples_SM = shuffle(samples_SM, random_state=seed)

if samples_bkg is not None:
    samples_bkg = shuffle(samples_bkg, random_state=seed)


print("Preprocessing data...")

if samples_bkg is not None:
    all_data = np.vstack((samples_SM, samples_bkg))
else:
    all_data = np.vstack((samples_SM, samples_alt))

scaler = StandardScaler()
scaler.fit(all_data)

samples_SM = scaler.transform(samples_SM)
samples_alt = scaler.transform(samples_alt)

if samples_bkg is not None:
    samples_bkg = scaler.transform(samples_bkg)

with open(f"{model_dir}/scaler_{run_id}.pkl", "wb") as ofile:
    pickle.dump(scaler, ofile)

print("Preprocessing done.")


def train_classifier(train_set_0, train_set_1, loc_id):
    x_train = np.vstack([train_set_0, train_set_1])
    all_labels = np.vstack([
        np.zeros((train_set_0.shape[0], 1)),
        np.ones((train_set_1.shape[0], 1)),
    ])

    X_train, X_val, Y_train, Y_val = train_test_split(
        x_train,
        all_labels,
        test_size=0.2,
        random_state=seed,
    )

    print(f"{loc_id} -> X_train: {X_train.shape}")
    print(f"{loc_id} -> X_val  : {X_val.shape}")

    X_train = np_to_torch(X_train, device)
    X_val = np_to_torch(X_val, device)
    Y_train = np_to_torch(Y_train, device)
    Y_val = np_to_torch(Y_val, device)

    kl_weight = run_configs["hyperparam.batch_size"] / X_train.shape[0]

    if args.network == "bnn":
        dense_net = BNN(
            n_inputs=train_set_0.shape[1],
            layers=run_configs["network.layers"],
            prior_sigma=0.1,
        )
        optimizer = torch.optim.AdamW(
            dense_net.parameters(),
            lr=run_configs["hyperparam.lr"],
            weight_decay=0,
        )
        train_bnn = True

    elif args.network == "dense":
        dense_net = NeuralNet(
            n_inputs=train_set_0.shape[1],
            layers=run_configs["network.layers"],
        )
        optimizer = torch.optim.AdamW(
            dense_net.parameters(),
            lr=run_configs["hyperparam.lr"],
            weight_decay=kl_weight,
        )
        train_bnn = False

    else:
        raise ValueError(f"Unknown network type: {args.network}")

    epochs, losses, losses_val = train_network(
        X_train,
        Y_train,
        X_val,
        Y_val,
        dense_net,
        optimizer,
        run_configs["hyperparam.n_epochs"],
        run_configs["hyperparam.batch_size"],
        device,
        seed=seed,
        train_bnn=train_bnn,
        kl_weight=kl_weight,
        network_id=f"{model_dir}/{run_id}_{loc_id}",
        use_early_stop=True,
        min_delta=0,
        patience_ES=run_configs["hyperparam.patience_ES"],
        patience_lr=run_configs["hyperparam.patience_lr"],
        loss_type="BCE",
    )

    history = {
        "epochs": epochs,
        "losses": losses,
        "losses_val": losses_val,
    }

    with open(f"{model_dir}/{run_id}_{loc_id}_history.pkl", "wb") as ofile:
        pickle.dump(history, ofile)


print("Using", args.network, "networks.")


if args.c1:
    print("Training classifier 1...")
    denom_c1 = np.c_[samples_SM, theta_alt_sm / 10.0]
    numer_c1 = np.c_[samples_alt, theta_alt / 10.0]
    train_classifier(denom_c1, numer_c1, "Ssm_Salt")
    print("Done with classifier 1.")

if args.c2:
    if samples_bkg is None:
        raise ValueError("Classifier 2 requested, but background samples were not loaded.")
    print("Training classifier 2...")
    denom_c2 = np.c_[samples_bkg, theta_alt_sm / 10.0]
    numer_c2 = np.c_[samples_alt, theta_alt / 10.0]
    train_classifier(denom_c2, numer_c2, "B_Salt")
    print("Done with classifier 2.")

if args.c3:
    if samples_bkg is None:
        raise ValueError("Classifier 3 requested, but background samples were not loaded.")
    print("Training classifier 3...")
    train_classifier(samples_SM, samples_bkg, "Ssm_B")
    print("Done with classifier 3.")