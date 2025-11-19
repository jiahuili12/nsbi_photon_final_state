#!/usr/bin/env python3
"""
Script to check that all necessary files are present in the event generation directory
based on what the 03a_read_delphes.py script expects to find.

Modified to match Jiahui's current directory structure:
- Base dir: (default) current directory "."
- MG processes: 02_mg_processes, 02_mg_processes_2
- Signal process: signal_sm, signal_sm_*, signal_supp_*
- Runs: Events/run_*
- Required files (current stage): unweighted_events.lhe
"""

import os
import yaml
import argparse
import glob

def load_workflow_config():
    """Load the workflow configuration from workflow.yaml"""
    with open("workflow.yaml", "r") as file:
        return yaml.safe_load(file)

def check_directory_structure(base_dir, workflow):
    """Check the overall directory structure"""
    print("=" * 60)
    print("CHECKING DIRECTORY STRUCTURE")
    print("=" * 60)

    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"❌ Base directory {base_dir} does not exist!")
        return False

    print(f"✅ Base directory {base_dir} exists")

    # Use Jiahui's naming: 02_mg_processes and 02_mg_processes_2
    mg_processes_dir = os.path.join(base_dir, "02_mg_processes")
    mg_processes_2_dir = os.path.join(base_dir, "02_mg_processes_2")

    if not os.path.exists(mg_processes_dir):
        print(f"❌ 02_mg_processes directory not found: {mg_processes_dir}")
        return False
    else:
        print(f"✅ 02_mg_processes directory found: {mg_processes_dir}")

    # Background directory may not exist yet – warn but don't fail
    if not os.path.exists(mg_processes_2_dir):
        print(f"⚠️  02_mg_processes_2 directory not found (no backgrounds yet, this is OK for now): {mg_processes_2_dir}")
    else:
        print(f"✅ 02_mg_processes_2 directory found: {mg_processes_2_dir}")

    return True

def check_signal_directories(mg_processes_dir):
    """Check signal directories (signal_sm, signal_sm_* and signal_supp_*)"""
    print("\n" + "=" * 60)
    print("CHECKING SIGNAL DIRECTORIES")
    print("=" * 60)

    signal_dirs = []

    # 1) Exact 'signal_sm'
    signal_sm_exact = os.path.join(mg_processes_dir, "signal_sm")
    if os.path.exists(signal_sm_exact):
        signal_dirs.append(signal_sm_exact)
        print("Found signal_sm directory")

    # 2) signal_sm_* (if later you clone/copy with suffixes)
    signal_sm_dirs = glob.glob(os.path.join(mg_processes_dir, "signal_sm_*"))
    signal_dirs.extend(signal_sm_dirs)
    print(f"Found {len(signal_sm_dirs)} signal_sm_* directories")

    # 3) signal_supp_* (for BSM benchmarks)
    signal_supp_dirs = glob.glob(os.path.join(mg_processes_dir, "signal_supp_*"))
    print(f"Found {len(signal_supp_dirs)} signal_supp directories")

    # For signal_supp directories, we need to check the morphing_basis_vector subdirectories
    signal_supp_process_dirs = []
    for signal_supp_dir in signal_supp_dirs:
        morphing_dirs = glob.glob(os.path.join(signal_supp_dir, "morphing_basis_vector_*"))
        signal_supp_process_dirs.extend(morphing_dirs)
        print(f"  {os.path.basename(signal_supp_dir)}: Found {len(morphing_dirs)} morphing_basis_vector directories")

    signal_dirs.extend(signal_supp_process_dirs)

    if not signal_dirs:
        print("❌ No signal directories found!")
        return False

    print(f"✅ Total signal process directories found: {len(signal_dirs)}")
    return signal_dirs

def check_background_directories(mg_processes_2_dir):
    """Check background directories"""
    print("\n" + "=" * 60)
    print("CHECKING BACKGROUND DIRECTORIES")
    print("=" * 60)

    if not os.path.exists(mg_processes_2_dir):
        print(f"⚠️  Background base directory does not exist yet: {mg_processes_2_dir}")
        return []

    background_dirs = glob.glob(os.path.join(mg_processes_2_dir, "background_*"))

    if not background_dirs:
        print("⚠️  No background directories found (this is OK if you haven't generated backgrounds yet)")
        return []

    print(f"✅ Found {len(background_dirs)} background directories")
    return background_dirs

def check_required_files_in_directory(process_dir, process_type):
    """Check for required files in a specific process directory"""
    missing_files = []

    # Check for Events directory
    events_dir = os.path.join(process_dir, "Events")
    if not os.path.exists(events_dir):
        missing_files.append("Events/")
        return missing_files

    # Check for run directories
    # In Jiahui's setup we have Events/run_01, run_02, ...
    run_dirs = glob.glob(os.path.join(events_dir, "run_*"))

    if not run_dirs:
        missing_files.append("Events/run_*")
        return missing_files

    # Check required files in each run directory
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)

        # At this stage (after 02_generate_events_test.py) we at least require unweighted_events.lhe
        required_files = [
            "unweighted_events.lhe",
        ]

        # Optional files (may appear later in the chain)
        optional_files = [
            "unweighted_events.lhe.gz",
            "tag_1_pythia8_events.hepmc.gz",
            "tag_1_pythia8_events_delphes.root",
        ]

        for file_name in required_files:
            file_path = os.path.join(run_dir, file_name)
            if not os.path.exists(file_path):
                missing_files.append(f"{run_name}/{file_name}")

        for file_name in optional_files:
            file_path = os.path.join(run_dir, file_name)
            if os.path.exists(file_path):
                print(f"  ✅ Optional file found: {run_name}/{file_name}")

    return missing_files

def check_all_processes(signal_dirs, background_dirs):
    """Check all process directories for required files"""
    print("\n" + "=" * 60)
    print("CHECKING REQUIRED FILES IN ALL PROCESSES")
    print("=" * 60)

    total_checked = 0
    total_with_issues = 0

    # Check signal directories
    print("\nChecking signal directories...")
    for signal_dir in signal_dirs:
        process_name = os.path.basename(signal_dir)

        if "morphing_basis_vector" in signal_dir:
            parent_dir = os.path.basename(os.path.dirname(signal_dir))
            morphing_name = os.path.basename(signal_dir)
            process_name = f"{parent_dir}/{morphing_name}"

        missing_files = check_required_files_in_directory(signal_dir, "signal")

        if missing_files:
            print(f"❌ {process_name}: Missing files:")
            for missing_file in missing_files:
                print(f"    - {missing_file}")
            total_with_issues += 1
        else:
            print(f"✅ {process_name}: All required files present")

        total_checked += 1

    # Check background directories (if any)
    print("\nChecking background directories...")
    for background_dir in background_dirs:
        process_name = os.path.basename(background_dir)
        missing_files = check_required_files_in_directory(background_dir, "background")

        if missing_files:
            print(f"❌ {process_name}: Missing files:")
            for missing_file in missing_files:
                print(f"    - {missing_file}")
            total_with_issues += 1
        else:
            print(f"✅ {process_name}: All required files present")

        total_checked += 1

    print(f"\n📊 SUMMARY:")
    print(f"   Total processes checked: {total_checked}")
    print(f"   Processes with issues: {total_with_issues}")
    print(f"   Processes OK: {total_checked - total_with_issues}")

    # 返回 True/False 用于最终总结
    return total_with_issues == 0

def check_expected_structure_for_delphes(workflow):
    """Check the structure that the delphes script expects"""
    print("\n" + "=" * 60)
    print("CHECKING EXPECTED STRUCTURE FOR DELPHES SCRIPT")
    print("=" * 60)

    input_dir_prefix = workflow["delphes"]["input_dir_prefix"]

    if not os.path.exists(input_dir_prefix):
        print(f"❌ Expected input directory does not exist: {input_dir_prefix}")
        return False

    print(f"✅ Input directory exists: {input_dir_prefix}")

    # Check for expected process directories
    expected_processes = ["signal_sm", "signal_supp", "background"]

    for process in expected_processes:
        process_dir = os.path.join(input_dir_prefix, process)
        if os.path.exists(process_dir):
            print(f"✅ Found {process} directory: {process_dir}")

            if process == "signal_supp":
                mb_vector_dirs = glob.glob(os.path.join(process_dir, "mb_vector_*"))
                if mb_vector_dirs:
                    print(f"  Found {len(mb_vector_dirs)} mb_vector_* subdirectories")
                    for mb_dir in mb_vector_dirs[:5]:
                        print(f"    - {os.path.basename(mb_dir)}")
                    if len(mb_vector_dirs) > 5:
                        print(f"    ... and {len(mb_vector_dirs) - 5} more")
                else:
                    print(f"  ⚠️  No mb_vector_* subdirectories found in {process}")
        else:
            print(f"⚠️  {process} directory not found: {process_dir}")

    return True

def main():
    parser = argparse.ArgumentParser(description="Check event generation files")
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory to check (default: current directory '.')",
    )
    parser.add_argument(
        "--workflow",
        default="workflow.yaml",
        help="Workflow configuration file (default: workflow.yaml)",
    )
    args = parser.parse_args()

    # Load workflow configuration
    try:
        workflow = load_workflow_config()
        print("✅ Workflow configuration loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load workflow configuration: {e}")
        return 1

    # Check directory structure
    if not check_directory_structure(args.base_dir, workflow):
        print("\n❌ Directory structure check failed!")
        return 1

    # Check signal directories in 02_mg_processes
    signal_dirs = check_signal_directories(os.path.join(args.base_dir, "02_mg_processes"))
    if not signal_dirs:
        print("\n❌ No signal directories found!")
        return 1

    # Check background directories in 02_mg_processes_2 (may be empty at this stage)
    background_dirs = check_background_directories(os.path.join(args.base_dir, "02_mg_processes_2"))

    # Check all processes for required files
    all_files_ok = check_all_processes(signal_dirs, background_dirs)

    # Check expected structure for delphes script
    delphes_structure_ok = check_expected_structure_for_delphes(workflow)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    if all_files_ok and delphes_structure_ok:
        print("🎉 All checks passed! The event generation directory is ready for processing.")
        return 0
    else:
        print("⚠️  Some issues were found. Please check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())
