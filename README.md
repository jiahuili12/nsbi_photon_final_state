# Neural Simulation-based Inference for DiHiggs: Complete Workflow
This is the repository containing the code worked on during my UROP placement at Imperial College London in the CMS group. The majority of the work is adapted from the code used in paper "Constraining the Higgs Potential with Neural Simulation-based Inference for Di-Higgs Production" (https://arxiv.org/abs/2405.15847)

### General version notes
- Python 3.8 (PyTorch, NumPy etc.)
- MadMiner installation (custom [branch](https://github.com/umarsdq/madminer) which allows MadSpin cards)
- MadGraph [3.5.1](https://herwig.hepforge.org/downloads?f=mirror) (Pythia8, Delphes 3.5.1, [SMEFT@NLO model](https://feynrules.irmp.ucl.ac.be/wiki/SMEFTatNLO), LHAPDF). 

Full instructions to install and configure the Madgraph 3.5.1 installation can be found [here](https://github.com/umarsdq/DiHiggsSBI_4b/blob/main/MG_install.md). For Imperial users, you may use the existing installation directory:

`/vols/cms/us322/mg5amcnlo`

## Setup working directory

For cluster users, it is recommended to first create a temporary directory for all pip cache 

```
export TMPDIR=/vols/cms/us322/tmp
```

To setup the workflow and create the conda environment,

```
conda create -n madminer_env python=3.8
conda activate madminer_env
git clone https://github.com/umarsdq/DiHiggsSBI_4b.git
cd DiHiggsSBI_4b
git clone https://github.com/umarsdq/madminer.git
cd madminer
pip install -e .
cd ..
pip install -r requirements.txt
python -m ipykernel install --user --name python38_madminer --display-name "Python3.8 (madminer)"
```


## Analysis flow

The `workflow.yaml` file is used for I/O management, so such folders don't have to be defined in every script. You must change the contents to your working directory. It is also recommended to generate events within a temporary large disk space.

### Event generation
Most of the instructions below are provided by the [authors](https://github.com/rmastand/nsbi_for_dihiggs)

1. `01_setup_morphing_basis.ipynb`: specify the SMEFT operators (in the SMEFT@NLO basis) and define a set of "benchmark points". For every event generated in MadGraph, weights corresponding to each benchmark will be computed. 

2. `02_generate_events.py`: generate the events with MadGraph. MadGraph commands, MadSpin cards, and Pythia cards can be found in the `cards` folder. MadGraph run cards are in `cards/run_cards`.

   There are run cards for both the 14 TeV and 100 TeV collider setups. Signal runs cards have no cuts on the decay products, since MadGraph is only used for the $gg \rightarrow hh$ decays, and MadSpin in used for the higgs decays. Background run cards have stricter angular and mass window cuts corresponding to those specified in the main paper. 

   To generate signal events and background events, add the `-sm` and `-b` flags respectively. For BSM event generation, add the `-supp` flag followed by the BSM id (1-9). For example, to generate events at the non-SM benchmark 2, run `python 02_generate_events.py -supp -supp_id 2`. **You can specify the desired number of Madgraph runs in the `workflow.yaml`.**

   Note: The number of events are currently set to 25k. To replicate the author's results, the number of runs below were followed:

   |                  | HL-LHC (14 TeV) | Future-Collider (100 TeV) |
   |------------------|-----------------|---------------------------|
   | **Signal**       | 20              | 30                        |
   | **BSM**          | 10              | 15                        |
   | **Background**   | 160             | 152                       |

   To maximise efficiency, many single run jobs were sent to the queue. The job began with copying the MadGraph installation and the run_card, and replacing the seed with a random seed, such that all jobs were independent. 
   
   It is also recommended that you generate more events than necessary, as some events simply did not complete. To check that all events were successfully generated, use the `02a_check_events.py` script, which checks all `Event` folders and locates any missing files. Simply replace the folder of any missing files with a spare run folder and rename accordingly.

3. `03a_read_delphes.py`: Run Delphes on the previously generated files and make selection cuts on the events. 

   This script assumes that you have a specific directory setup, namely that the outputs of step 2 are in `</path_from_workflow_yaml_delphes_input_dir_prefix/process_id/batch_<i>/`. `process_id` is an argument to the script (`signal_sm`, `signal_supp` for non-SM benchmarks, or `background_0`), and the batch is indexed by an integer. That directory can contain any number of Madgraph output directories `run_j`. 

   Note that this directory setup must be manually created. The method is especially useful when events are generated in parallel on a cluster setup with separate scratch and long-term storage directories.   To maximise the available condor jobs during the time of the UROP, 2 runs were copied into each batch using `02b_copy_events_parallel.sh`.

   As an example, you could run Delphes and apply kinematic cuts on events from 20 MadGraph runs that have been generated at the non-SM benchmark 2 by running `python 03a_read_delphes.py -p signal_supp -supp_id 2 -b 0 -start 1 -stop 2`.

   Next, compile the events over all batches and signal benchmarks with `python 03b_compile.py -p signal` and `python 03b_compile.py -p background`. 
   
   Finally, count the number of events in the final .h5 files using `python 03c_count_events.py ../events_4b/03_post_delphes_data_4b/`.

4. `04a_make_samples.ipynb`: generate samples of signal events at arbitrary benchmark points, using MadMiner. These samples will be used for network training and testing. You can generate multiple datasets (identified by `parameter_code`) depending on which SMEFT Wilson coefficients you want to vary.

   Finally, `04b_visualize_features.ipynb` is helpful to quickly visualize how kinematic features change as a function of Wilson coefficients. 

### Likelihood rato evaluation
5. `05_train_network.py`: Train the neural networks (classifiers). Specify the dataset that you want to run over be changing `sampling.output_dir` in `workflow.yaml` and by providing the correct `parameter_code` for the argument. Both simple dense nets and Bayesian nets are implemented. Network architecture and hyperparameters are hard-coded in the script, but they are all saved out into a config `yaml` with a particular run id (`rid`, specified in the arguments). 

   As described in the paper, there are 3 classifiers that need to be trained: (1) likelihood ratio of BSM signal to SM signal, (2) likelihood ratio of BSM signal to background, (3) likelihood ratio of background to SM signal. 

   As an example, you could train classifier 1 on a set of data that only varies over the first Wilson coefficient, using 5 kinematic features, with `python 05_train_network.py -p c0 -rid test_run -f 5 -c1`. For all Wilson coefficients (using -p c0, c1, c0c1 etc.), the network must be trained for all 3 classifiers (using -c1, -c2 and -c3) to construct the log-likelihood ratio value.

6. `06a_evaluate_test_statistic.ipynb` and `06b_evaluate_coverage.ipynb`: calculate likelihood ratios on previously generated test sets (or multiple likelihood ratios over different test set instatiations). It is also possible to ensemble over several networks from different seeds. 

This must be repeated for all relevant Wilson coefficients for later plotting. The `06_run_evaluate.sh` script will evaluate the test statistic and coverage for the chosen coefficients.

7. `07_nice_plots.ipynb`: Plot log-likelihood ratios and coverage tests for varying Wilson coefficients.
