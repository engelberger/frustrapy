# @title Frustratometer in Python
import sys

# import functions as frustrapy_functions
import subprocess
import os
import logging
import time
from typing import Dict
from tabulate import tabulate
from collections import defaultdict
import psutil
import datetime
import frustrapy
from tqdm import tqdm


class Profiler:
    def __init__(self):
        self.timing_stats = defaultdict(
            lambda: {
                "duration": 0.0,
                "start_memory": 0,
                "peak_memory": 0,
                "end_memory": 0,
                "start_time": None,
                "calls": 0,
                "parent": None,
            }
        )
        self.section_stack = []
        self.process = psutil.Process()

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

    def start_section(self, name: str):
        """Start timing a section with memory tracking"""
        current_memory = self.get_memory_usage()
        self.timing_stats[name]["start_memory"] = current_memory
        self.timing_stats[name]["peak_memory"] = current_memory
        self.timing_stats[name]["start_time"] = time.time()
        self.timing_stats[name]["calls"] += 1
        if self.section_stack:
            self.timing_stats[name]["parent"] = self.section_stack[-1]
        self.section_stack.append(name)

    def end_section(self, name: str):
        """End timing a section and update statistics"""
        if name in self.section_stack:
            duration = time.time() - self.timing_stats[name]["start_time"]
            current_memory = self.get_memory_usage()

            self.timing_stats[name]["duration"] += duration
            self.timing_stats[name]["end_memory"] = current_memory
            self.timing_stats[name]["peak_memory"] = max(
                self.timing_stats[name]["peak_memory"], current_memory
            )

            self.section_stack.remove(name)
            return duration
        return 0

    def print_report(self):
        """Print a comprehensive profiling report"""
        root_sections = [
            name
            for name, stats in self.timing_stats.items()
            if stats["parent"] is None and name != "Total"
        ]
        total_time = sum(self.timing_stats[name]["duration"] for name in root_sections)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        timing_data = []
        for section, stats in self.timing_stats.items():
            if section == "Total":
                continue

            if stats["parent"]:
                parent_time = self.timing_stats[stats["parent"]]["duration"]
                percentage = (
                    (stats["duration"] / parent_time * 100) if parent_time > 0 else 0
                )
            else:
                percentage = (
                    (stats["duration"] / total_time * 100) if total_time > 0 else 0
                )

            memory_change = stats["end_memory"] - stats["start_memory"]

            section_name = section
            if stats["parent"]:
                section_name = "  └─ " + section

            timing_data.append(
                [
                    section_name,
                    f"{stats['duration']:.2f}s",
                    f"{percentage:.1f}%",
                    stats["calls"],
                    (
                        f"{stats['duration']/stats['calls']:.2f}s"
                        if stats["calls"] > 0
                        else "N/A"
                    ),
                    f"{stats['start_memory']:.1f}",
                    f"{stats['peak_memory']:.1f}",
                    f"{stats['end_memory']:.1f}",
                    f"{memory_change:+.1f}",
                ]
            )

        timing_data.sort(key=lambda x: float(x[1][:-1]), reverse=True)

        print("\n" + "=" * 100)
        print(f"Profiling Report - {current_time}")
        print("=" * 100)

        print("\nSystem Information:")
        print(f"CPU Count: {psutil.cpu_count()}")
        print(
            f"Total System Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB"
        )
        print(
            f"Available System Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB"
        )

        print("\nExecution Statistics:")
        print(
            tabulate(
                timing_data,
                headers=[
                    "Section",
                    "Duration",
                    "% of Parent",
                    "Calls",
                    "Avg Time/Call",
                    "Start Mem (MB)",
                    "Peak Mem (MB)",
                    "End Mem (MB)",
                    "Mem Δ (MB)",
                ],
                tablefmt="grid",
                floatfmt=".2f",
            )
        )

        print("\nSummary:")
        print(f"Total Execution Time: {total_time:.2f}s")
        print(
            f"Peak Memory Usage: {max(stats['peak_memory'] for stats in self.timing_stats.values()):.1f} MB"
        )

        print("\nObservations:")
        for section, stats in self.timing_stats.items():
            if section == "Total":
                continue
            if not stats["parent"] and stats["duration"] > total_time * 0.3:
                print(
                    f"- {section} took {stats['duration']/total_time*100:.1f}% of total execution time"
                )
            if (stats["peak_memory"] - stats["start_memory"]) > 500:
                print(
                    f"- {section} had significant memory usage (peak: {stats['peak_memory']:.1f} MB)"
                )

        print("=" * 100)


# Create profiler instance
profiler = Profiler()


# Start overall timing
profiler.start_section("Total Execution")

# Configuration
profiler.start_section("Configuration")
mode = "singleresidue"
pdbs_dir = "/workspaces/frustrapy/example_pdbs"
results_dir = "/workspaces/frustrapy/Results_example"
example = True
overwrite = False
debug = "INFO"

# Disable all logging by default
logging.getLogger().handlers = []  # Remove any existing handlers
logging.getLogger().setLevel(
    logging.CRITICAL
)  # Set root logger to CRITICAL (highest level)

# Only configure logging if debug level is specified
if debug.upper() in ["DEBUG", "INFO"]:
    if debug.upper() == "DEBUG":
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO

    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set frustrapy logger level
    logging.getLogger("frustrapy").setLevel(logging_level)
else:
    # Disable all loggers when debug is "NONE"
    logging.getLogger("frustrapy").setLevel(logging.CRITICAL)
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)


if example:
    pdbs_dir = "/workspaces/frustrapy"
    results_dir = "/workspaces/frustrapy/Results_example"
    subprocess.run(["rm", "-rf", "/workspaces/frustrapy/Results_example/*"])

if overwrite:
    if example:
        subprocess.run(
            ["rm", "-rf", "/workspaces/frustrapy/Results_example/*"]
        )
    else:
        results_dir = os.path.abspath(results_dir)
        subprocess.run(["rm", "-rf", f"{results_dir}/*"])
profiler.end_section("Configuration")


# Configurational Frustration Analysis
profiler.start_section("Configurational Frustration Analysis")
pdb_file = "af2_masking_vanilla_94a41_best_model_2_ptm_r3_seed_000_mask_false_id_X.pdb"
pdb_config, plots_config, _density_config, _single_res_config = frustrapy.calculate_frustration(
    pdb_file=os.path.join(pdbs_dir, pdb_file),
    mode="configurational",
    results_dir=results_dir,
    debug=debug.upper(),
    chain="A",
)
profiler.end_section("Configurational Frustration Analysis")

# Define residues to analyze
residues_to_analyze = {"A": [144, 146]}
# Directory frustration analysis
profiler.start_section("Directory Frustration Analysis")
plots_dir_dict = frustrapy.dir_frustration(
    pdbs_dir=pdbs_dir,
    mode=mode,
    results_dir=results_dir,
    debug=debug.upper(),
    chain="A",
    residues=residues_to_analyze,
)
profiler.end_section("Directory Frustration Analysis")
# Single PDB analysis (Single Residue mode)
profiler.start_section("Single PDB Analysis")
pdb_file = "af2_masking_vanilla_94a41_best_model_2_ptm_r3_seed_000_mask_false_id_X.pdb"

# Calculate total mutations to process
total_mutations = (
    sum(len(residues) for residues in residues_to_analyze.values()) * 20
)  # 20 amino acids per residue

# Remove the progress bar from here since it's handled in the mutations module
pdb, plots, _density, single_res = frustrapy.calculate_frustration(
    pdb_file=os.path.join(pdbs_dir, pdb_file),
    mode=mode,
    results_dir=results_dir,
    debug=debug.upper(),
    chain="A",
    residues=residues_to_analyze,
)
profiler.end_section("Single PDB Analysis")
# Results analysis and display
profiler.start_section("Results Analysis")
# Read the human-readable single-residue frustration tables instead of deserializing
# pickle files discovered by walking the filesystem. pickle.load on untrusted files is
# an arbitrary-code-execution risk (P0-1); the .pdb_singleresidue text tables carry the
# same per-residue FrstIndex in a safe format.
import pandas as pd

results_found = 0
for root, dirs, files in os.walk(results_dir):
    for file in files:
        if file.endswith(".pdb_singleresidue"):
            table_path = os.path.join(root, file)
            table = pd.read_csv(table_path, sep=r"\s+")
            print(f"\nSingle-residue frustration from: {file}")
            results_found += 1
            for res_num in [144, 146]:
                row = table[table["Res"] == res_num]
                if not row.empty:
                    r = row.iloc[0]
                    print(
                        f"Position {res_num} (Native: {r['AA']}): "
                        f"FrstIndex = {float(r['FrstIndex']):.3f}"
                    )
            print("-" * 50)
if results_found == 0:
    print("No single-residue frustration tables found.")
profiler.end_section("Results Analysis")
# End overall timing and print report
profiler.end_section("Total Execution")
profiler.print_report()
