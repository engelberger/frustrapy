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
pdbs_dir = "/content"
results_dir = "/home/ceramirez/github/frustrapy/Results_example"
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
    pdbs_dir = "/home/ceramirez/github/frustrapy"
    results_dir = "/home/ceramirez/github/frustrapy/Results_example"
    subprocess.run(["rm", "-rf", "/home/ceramirez/github/frustrapy/Results_example/*"])

if overwrite:
    if example:
        subprocess.run(
            ["rm", "-rf", "/home/ceramirez/github/frustrapy/Results_example/*"]
        )
    else:
        results_dir = os.path.abspath(results_dir)
        subprocess.run(["rm", "-rf", f"{results_dir}/*"])
profiler.end_section("Configuration")


# Configurational Frustration Analysis
profiler.start_section("Configurational Frustration Analysis")
pdb_file = "af2_masking_vanilla_94a41_best_model_2_ptm_r3_seed_000_mask_false_id_X.pdb"
pdb_config, plots_config, density_results, _ = frustrapy.calculate_frustration(
    pdb_file=os.path.join(pdbs_dir, pdb_file),
    mode="configurational",
    results_dir=results_dir,
    debug=debug.upper(),
    chain="A",
)

# You can now use density_results directly
if density_results:
    print("\nFrustration Density Results:")
    print(f"Number of residues analyzed: {len(density_results.densities)}")
    print(f"Number of contact points: {len(density_results.contact_coordinates)}")

    # Example: Print summary of first few residues
    print("\nFirst 5 residues density analysis:")
    for density in density_results.densities[:5]:
        print(f"\nResidue {density.residue_number} Chain {density.chain_id}:")
        print(f"  Total contacts: {density.total_density}")
        print(
            f"  Highly frustrated: {density.highly_frustrated} ({density.rel_highly_frustrated:.2%})"
        )
        print(
            f"  Neutrally frustrated: {density.neutrally_frustrated} ({density.rel_neutrally_frustrated:.2%})"
        )
        print(
            f"  Minimally frustrated: {density.minimally_frustrated} ({density.rel_minimally_frustrated:.2%})"
        )

    # Find residue with lowest configurational frustration density
    min_frustration_density = min(
        density_results.densities,
        key=lambda x: x.rel_highly_frustrated if x.total_density > 0 else float("inf"),
    )

    print("\nResidue with lowest configurational frustration density:")
    print(
        f"Residue {min_frustration_density.residue_number} Chain {min_frustration_density.chain_id}:"
    )
    print(f"  Total contacts: {min_frustration_density.total_density}")
    print(
        f"  Highly frustrated: {min_frustration_density.highly_frustrated} ({min_frustration_density.rel_highly_frustrated:.2%})"
    )
    print(
        f"  Neutrally frustrated: {min_frustration_density.neutrally_frustrated} ({min_frustration_density.rel_neutrally_frustrated:.2%})"
    )
    print(
        f"  Minimally frustrated: {min_frustration_density.minimally_frustrated} ({min_frustration_density.rel_minimally_frustrated:.2%})"
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

# Update unpacking to handle 4 return values including single_residue_data
pdb, plots, density_results, single_residue_data = frustrapy.calculate_frustration(
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
try:
    if single_residue_data and "A" in single_residue_data:
        for res_num in [144, 146]:
            if res_num in single_residue_data["A"]:
                res_data = single_residue_data["A"][res_num]
                mutations = res_data.mutations
                # Find most and least frustrated mutations
                most_frustrated = min(mutations.items(), key=lambda x: x[1])
                least_frustrated = max(mutations.items(), key=lambda x: x[1])

                print(f"\nPosition {res_num} (Native: {res_data.residue_name})")
                print(
                    f"Most frustrated mutation: {res_data.residue_name} → {most_frustrated[0]} "
                    f"(Frustration Index: {most_frustrated[1]:.3f})"
                )
                print(
                    f"Least frustrated mutation: {res_data.residue_name} → {least_frustrated[0]} "
                    f"(Frustration Index: {least_frustrated[1]:.3f})"
                )

                # Sort and display mutations
                sorted_mutations = sorted(mutations.items(), key=lambda x: x[1])
                print(
                    "\nAll mutations sorted by frustration (top 5 most and least frustrated):"
                )
                print("Most frustrated:")
                for mut, score in sorted_mutations[:5]:
                    print(f"  {res_data.residue_name} → {mut}: {score:.3f}")
                print("Least frustrated:")
                for mut, score in sorted_mutations[-5:]:
                    print(f"  {res_data.residue_name} → {mut}: {score:.3f}")
                print("-" * 50)
    else:
        print("\nNo single residue data available in the results")
except Exception as e:
    print(f"Error analyzing results: {str(e)}")
profiler.end_section("Results Analysis")

# End overall timing and print report
profiler.end_section("Total Execution")
profiler.print_report()
