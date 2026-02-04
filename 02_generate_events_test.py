#!/usr/bin/env python
# coding: utf-8
import os
import logging
import numpy as np
import time
import threading
import psutil
from datetime import datetime

from madminer.core import MadMiner
import argparse
import glob
import subprocess

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

args = parser.parse_args()

mg_dir = workflow["madgraph"]["dir"]

def fix_and_rerun_pythia(py8_parallel_dir):
    split_dirs = sorted(glob.glob(os.path.join(py8_parallel_dir, "split_*")))
    for sd in split_dirs:
        lhe_gz = os.path.join(sd, "events.lhe.gz")
        lhe = os.path.join(sd, "events.lhe")
        card = os.path.join(sd, "PY8Card.dat")
        runner = os.path.join(sd, "run_PY8.sh")
        hepmc = os.path.join(sd, "events.hepmc")
        log = os.path.join(sd, "PY8_log.txt")

        if not os.path.exists(lhe_gz):
            continue

        if (not os.path.exists(lhe)) or (os.path.getsize(lhe) == 0):
            subprocess.run(["bash","-lc", f"gunzip -c {lhe_gz} > {lhe}"], check=True)

        subprocess.run(["bash","-lc", f"sed -i 's|^Beams:LHEF=.*$|Beams:LHEF=events.lhe|' {card}"], check=True)
        subprocess.run(["bash","-lc", f"rm -f {hepmc} {log}"], check=False)
        subprocess.run(["bash","-lc", f"cd {sd} && bash {runner}"], check=True)

        if (not os.path.exists(hepmc)) or (os.path.getsize(hepmc) == 0):
            raise RuntimeError(f"Pythia rerun failed: {hepmc} is empty in {sd}")


class CPUMonitor:
    """Monitor CPU usage during event generation"""

    
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.monitoring = False
        self.monitor_thread = None
        self.cpu_usage = []
        self.memory_usage = []
        self.timestamps = []
        
    def start_monitoring(self):
        """Start CPU monitoring in a separate thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("CPU monitoring started...")
        
    def stop_monitoring(self):
        """Stop CPU monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("CPU monitoring stopped.")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Get memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_gb = memory.used / (1024**3)
                
                # Get timestamp
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # Store data
                self.cpu_usage.append(cpu_percent)
                self.memory_usage.append(memory_percent)
                self.timestamps.append(timestamp)
                
                # Log to file if specified
                if self.log_file:
                    with open(self.log_file, 'a') as f:
                        f.write(f"{timestamp} - CPU: {cpu_percent:.1f}% | Memory: {memory_percent:.1f}% ({memory_gb:.1f} GB)\n")
                
                # Print to console every 30 seconds
                if len(self.cpu_usage) % 30 == 0:
                    print(f"[{timestamp}] CPU: {cpu_percent:.1f}% | Memory: {memory_percent:.1f}% ({memory_gb:.1f} GB)")
                    
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(1)
    
    def get_summary(self):
        """Get monitoring summary"""
        if not self.cpu_usage:
            return "No monitoring data available"
            
        avg_cpu = np.mean(self.cpu_usage)
        max_cpu = np.max(self.cpu_usage)
        avg_memory = np.mean(self.memory_usage)
        max_memory = np.max(self.memory_usage)
        
        return {
            'avg_cpu': avg_cpu,
            'max_cpu': max_cpu,
            'avg_memory': avg_memory,
            'max_memory': max_memory,
            'duration': len(self.cpu_usage),
            'samples': len(self.cpu_usage)
        }
    
    def print_summary(self):
        """Print monitoring summary"""
        summary = self.get_summary()
        if isinstance(summary, str):
            print(summary)
            return
            
        print("\n" + "="*50)
        print("CPU MONITORING SUMMARY")
        print("="*50)
        print(f"Monitoring duration: {summary['duration']} seconds")
        print(f"Average CPU usage: {summary['avg_cpu']:.1f}%")
        print(f"Maximum CPU usage: {summary['max_cpu']:.1f}%")
        print(f"Average memory usage: {summary['avg_memory']:.1f}%")
        print(f"Maximum memory usage: {summary['max_memory']:.1f}%")
        print(f"Total CPU cores: {psutil.cpu_count()}")
        print(f"CPU cores used: {psutil.cpu_count(logical=False)} (physical)")
        print("="*50)

"""
GENERATE EVENTS
"""

# Set n_runs based on energy and process type according to the table
energy = int(workflow["madgraph"]["energy"])

# Attempting to replicate results from the paper.
if energy == 14:  # HL-LHC
    if args.sm:
        n_runs = 20  # Signal runs
    elif args.supp:
        n_runs = 10  # BSM runs
    elif args.b:
        n_runs = 160  # Background runs
    else:
        n_runs = 1  # Default fallback
elif energy == 100:  # Future-Collider
    if args.sm:
        n_runs = 30  # Signal runs
    elif args.supp:
        n_runs = 15  # BSM runs
    elif args.b:
        n_runs = 152  # Background runs
    else:
        n_runs = 1  # Default fallback
else:
    n_runs = 1

if workflow["madgraph"]["test_run"]:
    n_runs = 1

miner = MadMiner()
miner.load(workflow["morphing_setup"])

if energy == 14:
    run_cards_signal = [f"{working_dir}/cards/run_cards/run_card_signal_14TeV_test.dat" for i in range(n_runs)]
    run_cards_background = [f"{working_dir}/cards/run_cards/run_card_background_14TeV_test.dat" for i in range(n_runs)]
elif energy == 100:
    run_cards_signal = [f"{working_dir}/cards/run_cards/run_card_signal_100TeV.dat" for i in range(n_runs)]
    run_cards_background = [f"{working_dir}/cards/run_cards/run_card_background_100TeV.dat" for i in range(n_runs)]
else:
    print("No run card for desired energy.")
    exit()

print(f"\nRunning at energy", workflow["madgraph"]["energy"], "TeV")
print(f"Number of runs: {n_runs}")

# Initialize CPU monitor
monitor_log_file = f"{working_dir}/logs/cpu_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
os.makedirs(os.path.dirname(monitor_log_file), exist_ok=True)
cpu_monitor = CPUMonitor(log_file=monitor_log_file)

print(f"CPU monitoring log: {monitor_log_file}")
print()

# Start CPU monitoring
cpu_monitor.start_monitoring()

if args.sm:
    miner.run_multiple(
        sample_benchmarks=["sm"],
        mg_directory=mg_dir,
        mg_process_directory="{mg_process_output_dir}/signal_sm".format(mg_process_output_dir = workflow["madgraph"]["output_dir"]),
        proc_card_file=f"{working_dir}/cards/proc_card_signal.dat",
        param_card_template_file=f"{working_dir}/cards/restrict_LO.dat",
        madspin_card_file=f"{working_dir}/cards/madspin_card.dat",
        run_card_files=run_cards_signal,
        pythia8_card_file=f"{working_dir}/cards/pythia8_card.dat", 
        log_directory=f"{working_dir}/logs/signal_sm",
        #python_executable="python3",
        order="LO",
        #systematics=["signal_norm"]
    )

    # 自动修复：把所有 run_*_decayed_*/PY8_parallelization 都处理一遍
    events_root = os.path.join(workflow["madgraph"]["output_dir"], "signal_sm", "Events")
    py8_dirs = sorted(glob.glob(os.path.join(events_root, "run_*", "PY8_parallelization"))) + \
              sorted(glob.glob(os.path.join(events_root, "run_*_decayed_*", "PY8_parallelization")))
    py8_dirs = sorted(set(py8_dirs))
    for d in py8_dirs:
        fix_and_rerun_pythia(d)

if args.supp:
    
    miner.run_multiple(
        sample_benchmarks=[f"morphing_basis_vector_{args.supp_id}"],
        mg_directory=mg_dir,
        mg_process_directory="{mg_process_output_dir}/signal_supp/morphing_basis_vector_{supp_id}".format(mg_process_output_dir = workflow["madgraph"]["output_dir"], supp_id = args.supp_id),
        proc_card_file=f"{working_dir}/cards/proc_card_signal.dat",
        param_card_template_file=f"{working_dir}/cards/restrict_LO.dat",
        madspin_card_file=f"{working_dir}/cards/madspin_card.dat",
        run_card_files=run_cards_signal,
        pythia8_card_file=f"{working_dir}/cards/pythia8_card.dat", 
        log_directory=f"{working_dir}/logs/signal_supp",
        #python_executable="python3",
        order="LO",
        #systematics=["signal_norm"]
    )

    events_root = os.path.join(workflow["madgraph"]["output_dir"], "signal_supp", f"morphing_basis_vector_{args.supp_id}", "Events")
    py8_dirs = sorted(glob.glob(os.path.join(events_root, "run_*", "PY8_parallelization"))) + \
              sorted(glob.glob(os.path.join(events_root, "run_*_decayed_*", "PY8_parallelization")))
    py8_dirs = sorted(set(py8_dirs))
    for d in py8_dirs:
        fix_and_rerun_pythia(d)



if args.b:
    for i in range(1):
        miner.run_multiple(
            is_background=True,
            sample_benchmarks=["sm"],
            mg_directory=mg_dir,
            mg_process_directory="{mg_process_output_dir}_2/background".format(mg_process_output_dir = workflow["madgraph"]["output_dir"], idd = i),
            proc_card_file="{working_dir}/cards/proc_card_background.dat".format(idd = i, working_dir = working_dir),
            param_card_template_file=f"{working_dir}/cards/restrict_LO.dat",
            pythia8_card_file=f"{working_dir}/cards/pythia8_card.dat", 
            run_card_files=run_cards_background,
            log_directory=f"{working_dir}/logs_2/background",
        )

        events_root = os.path.join(f"{workflow['madgraph']['output_dir']}_2", "background", "Events")
        py8_dirs = sorted(glob.glob(os.path.join(events_root, "run_*", "PY8_parallelization"))) + \
                  sorted(glob.glob(os.path.join(events_root, "run_*_decayed_*", "PY8_parallelization")))
        py8_dirs = sorted(set(py8_dirs))
        for d in py8_dirs:
            fix_and_rerun_pythia(d)

# Stop CPU monitoring and print summary
cpu_monitor.stop_monitoring()
cpu_monitor.print_summary()