"""Core benchmark functionality for measuring FrustraPy performance."""

import os
import pandas as pd
import numpy as np
import frustrapy
import time
from typing import Dict, List, Union, Tuple
import platform
import multiprocessing
import re
import json

def run_benchmark(
    pdb_file: str, 
    chain: str, 
    residue: int, 
    cpu_list: list, 
    results_dir: str,
    repeats: int = 3,
    cautious: bool = True
) -> pd.DataFrame:
    """
    Run mutation frustration analysis with varying CPU counts and collect performance metrics.
    
    Args:
        pdb_file: Path to the PDB file
        chain: Chain identifier
        residue: Residue number to mutate
        cpu_list: List of CPU counts to test (e.g., [1, 2, 4, 8])
        results_dir: Directory for storing benchmark outputs
        repeats: Number of repetitions for each CPU configuration (default=3)
        cautious: Boolean indicating whether to enable cautious mode
        
    Returns:
        DataFrame with aggregated benchmark results containing:
            - n_cpus: Number of CPUs used
            - time_s: Mean execution time in seconds
            - time_s_std: Standard deviation of execution time
            - time_per_aa: Mean time per amino acid mutation
            - time_per_aa_std: Standard deviation of time per amino acid
            - pdb_id: PDB identifier
            - speedup: Mean speedup factor relative to single-CPU
            - speedup_std: Standard deviation of speedup
            - efficiency: Mean parallel efficiency
            - efficiency_std: Standard deviation of efficiency
    """
    # Prepare output directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Get PDB identifier from filename
    pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]
    
    # Run benchmarks for each CPU count (with cautious mode)
    raw_path = os.path.join(results_dir, 'raw_benchmark_data.csv')
    df_old = None
    if cautious and os.path.exists(raw_path):
        df_old = pd.read_csv(raw_path)

    all_results = []
    if df_old is not None:
        all_results.append(df_old)

    # Gather CPU specs for reproducibility
    cpu_specs = {}
    try:
        cpu_specs['cpu_count_logical'] = multiprocessing.cpu_count()
    except:
        pass
    cpu_specs['machine'] = platform.machine()
    cpu_specs['processor'] = platform.processor()
    cpu_specs['platform'] = platform.platform()
    if os.path.exists('/proc/cpuinfo'):
        try:
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if 'model name' in line:
                        cpu_specs['model_name'] = line.split(':')[1].strip()
                        break
        except:
            pass
    try:
        import psutil
        freq = psutil.cpu_freq()
        cpu_specs['cpu_freq_mhz'] = getattr(freq, 'max', None) or getattr(freq, 'current', None)
    except:
        pass
    # Log CPU specs
    print('CPU specifications for this benchmark run:')
    for k, v in cpu_specs.items():
        print(f'  {k}: {v}')
    # Save CPU specs to JSON for full reproducibility
    spec_path = os.path.join(results_dir, 'cpu_specs.json')
    try:
        with open(spec_path, 'w') as f:
            json.dump(cpu_specs, f, indent=2)
        print(f'CPU specifications saved to {spec_path}')
    except Exception:
        print(f'Warning: failed to save CPU specs to {spec_path}')

    for ncpu in cpu_list:
        # Determine how many previous runs exist for this CPU count
        prev_runs = 0
        if df_old is not None:
            prev_data = df_old[df_old['n_cpus'] == ncpu]
            if not prev_data.empty:
                prev_runs = prev_data['run'].max()

        # Calculate how many new runs to perform
        runs_to_do = repeats if df_old is None else max(0, repeats - prev_runs)
        if runs_to_do == 0:
            continue

        cpu_results = []
        for i in range(runs_to_do):
            run_number = prev_runs + i + 1
            # Create per-run directory
            run_dir = os.path.join(results_dir, f"run_{ncpu}_cpus_{run_number}")
            os.makedirs(run_dir, exist_ok=True)

            # Run single-residue frustration analysis
            pdb, plots, density, single_data = frustrapy.calculate_frustration(
                pdb_file=pdb_file,
                mode="singleresidue",
                chain=chain,
                residues={chain: [residue]},
                results_dir=run_dir,
                n_cpus=ncpu,
                debug=False
            )

            # Extract metrics
            metrics = getattr(pdb, 'MutationAnalysis', {}).copy()
            metrics['n_cpus'] = ncpu
            metrics['pdb_id'] = pdb_id
            metrics['run'] = run_number
            cpu_results.append(metrics)

        # Create DataFrame for this CPU configuration
        df_cpu = pd.DataFrame(cpu_results)
        all_results.append(df_cpu)

    # Combine all results
    df_raw = pd.concat(all_results, ignore_index=True)

    # Attach CPU specs to raw results for full reproducibility
    for spec, val in cpu_specs.items():
        df_raw[spec] = val

    # Save detailed raw results
    df_raw.to_csv(raw_path, index=False)
    
    # Compute aggregated statistics (mean and std)
    stats = []
    for cpu in cpu_list:
        cpu_data = df_raw[df_raw['n_cpus'] == cpu]
        
        # Calculate mean and std for each metric
        stats_dict = {'n_cpus': cpu, 'pdb_id': pdb_id}
        
        # Include CPU specs for reproducibility
        stats_dict.update(cpu_specs)
        
        if 'time_s' in cpu_data.columns:
            stats_dict['time_s'] = cpu_data['time_s'].mean()
            stats_dict['time_s_std'] = cpu_data['time_s'].std()
            
        if 'time_per_aa' in cpu_data.columns:
            stats_dict['time_per_aa'] = cpu_data['time_per_aa'].mean()
            stats_dict['time_per_aa_std'] = cpu_data['time_per_aa'].std()
        
        # Add raw data paths
        stats_dict['data_source'] = raw_path
        stats_dict['n_repeats'] = repeats
        
        stats.append(stats_dict)
    
    df_stats = pd.DataFrame(stats)
    
    # Compute speedup and efficiency if possible
    if 'time_s' in df_stats.columns and len(df_stats) > 0 and 1 in df_stats['n_cpus'].values:
        # Get baseline time (single CPU)
        t1_mean = df_stats.loc[df_stats['n_cpus'] == 1, 'time_s'].iloc[0]
        t1_std = df_stats.loc[df_stats['n_cpus'] == 1, 'time_s_std'].iloc[0]
        
        # Calculate speedups and efficiencies
        for idx, row in df_stats.iterrows():
            cpu_count = row['n_cpus']
            mean_time = row['time_s']
            std_time = row['time_s_std']
            
            # Calculate speedup with error propagation
            speedup = t1_mean / mean_time
            
            # Error propagation for division: (σ_result/result)² = (σ_t1/t1)² + (σ_time/time)²
            rel_error_squared = (t1_std/t1_mean)**2 + (std_time/mean_time)**2
            speedup_std = speedup * np.sqrt(rel_error_squared)
            
            # Calculate efficiency
            efficiency = speedup / cpu_count
            efficiency_std = speedup_std / cpu_count
            
            # Update DataFrame
            df_stats.loc[idx, 'speedup'] = speedup
            df_stats.loc[idx, 'speedup_std'] = speedup_std
            df_stats.loc[idx, 'efficiency'] = efficiency
            df_stats.loc[idx, 'efficiency_std'] = efficiency_std
    
    return df_stats


def get_raw_benchmark_data(
    results_dir: str
) -> pd.DataFrame:
    """
    Load raw benchmark data from a previous run, including all individual repetitions.
    
    Args:
        results_dir: Directory containing benchmark results
        
    Returns:
        DataFrame with raw benchmark data for all runs
    """
    raw_path = os.path.join(results_dir, 'raw_benchmark_data.csv')
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw benchmark data not found at {raw_path}")
    
    return pd.read_csv(raw_path) 