#!/usr/bin/env python
"""
Command-line tool for benchmarking FrustraPy mutation analysis performance.

Usage:
    python run_benchmark.py --pdb_file 1m6k.pdb --chain A --residue 75 --cpus 1 2 4 8 --repeats 3
"""

import os
import argparse
import pandas as pd
import frustrapy.benchmark as benchmark
import sys
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Parse arguments and run benchmark with visualizations."""
    parser = argparse.ArgumentParser(
        description="Benchmark mutational analysis performance in FrustraPy."
    )
    parser.add_argument('--pdb_file', required=True, help='Path to the PDB file')
    parser.add_argument('--chain', required=True, help='Chain identifier')
    parser.add_argument('--residue', type=int, required=True, help='Residue number to mutate')
    parser.add_argument('--results_dir', default='benchmark_results', help='Directory for benchmark results')
    parser.add_argument('--cpus', nargs='+', type=int, default=[1,2,4,8], help='List of CPU counts to test')
    parser.add_argument('--repeats', type=int, default=3, help='Number of times to repeat each benchmark for statistical significance')
    parser.add_argument('--output_csv', default='benchmark.csv', help='Output CSV filename (inside results_dir)')
    parser.add_argument('--plot_format', default='both', choices=['plotly', 'seaborn', 'both'], 
                      help='Output plot format (default: both)')
    parser.add_argument('--no-cautious', dest='cautious', action='store_false', 
                        help='Disable cautious mode; rerun benchmarks even if data already recorded')
    args = parser.parse_args()

    # Create directory structure
    results_dir = args.results_dir
    data_dir = os.path.join(results_dir, 'data')
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Report cautious mode status
    print(f"Cautious mode {'enabled: skipping already recorded runs' if args.cautious else 'disabled: rerunning all benchmarks'}")
    
    # Run benchmark
    print(f"Running benchmark with {len(args.cpus)} CPU configurations: {args.cpus}")
    print(f"Each configuration will be repeated {args.repeats} times for statistical significance")
    print(f"Cautious mode {'enabled: skipping already recorded runs' if args.cautious else 'disabled: rerunning all benchmarks'}")
    
    df = benchmark.run_benchmark(
        pdb_file=args.pdb_file,
        chain=args.chain,
        residue=args.residue,
        cpu_list=args.cpus,
        results_dir=results_dir,
        repeats=args.repeats,
        cautious=args.cautious
    )
    
    # Save aggregated data
    csv_path = os.path.join(data_dir, args.output_csv)
    df.to_csv(csv_path, index=False)
    print(f"Aggregated benchmark results saved to {csv_path}")
    
    # Also save raw data
    raw_path = os.path.join(data_dir, 'raw_benchmark_data.csv')
    if os.path.exists(raw_path):
        print(f"Raw benchmark data saved to {raw_path}")
    
    # Print results table
    print("\nBenchmark Results Summary:")
    display_cols = ['n_cpus', 'time_s', 'time_s_std', 'speedup', 'speedup_std', 'efficiency', 'efficiency_std']
    display_cols = [col for col in display_cols if col in df.columns]
    print(df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    
    # Titles for publication plots
    titles = {
        'speedup': "Performance Scaling with Multi-core Processing",
        'efficiency': "Parallel Efficiency of Mutational Analysis",
        'time': "Computation Time Reduction with Parallel Processing"
    }
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Plotly plots
    if args.plot_format in ['plotly', 'both']:
        print("Creating Plotly plots...")
        # Speedup plot
        fig_su = benchmark.plot_speedup_linear(df, title=titles['speedup'])
        fig_su.write_image(os.path.join(plots_dir, 'speedup_linear.png'), scale=2)
        fig_su.write_html(os.path.join(plots_dir, 'speedup_linear.html'))
        
        # Efficiency plot
        fig_eff = benchmark.plot_efficiency(df, title=titles['efficiency'])
        fig_eff.write_image(os.path.join(plots_dir, 'efficiency.png'), scale=2)
        fig_eff.write_html(os.path.join(plots_dir, 'efficiency.html'))
        
        # Execution time plot
        fig_time = benchmark.plot_execution_time(df, title=titles['time'])
        fig_time.write_image(os.path.join(plots_dir, 'execution_time.png'), scale=2)
        fig_time.write_html(os.path.join(plots_dir, 'execution_time.html'))
    
    # Seaborn plots
    if args.plot_format in ['seaborn', 'both']:
        print("Creating Seaborn plots...")
        benchmark.plot_seaborn_speedup(
            df, 
            save_path=os.path.join(plots_dir, 'speedup_seaborn.png'),
            title=titles['speedup']
        )
        benchmark.plot_seaborn_efficiency(
            df, 
            save_path=os.path.join(plots_dir, 'efficiency_seaborn.png'),
            title=titles['efficiency']
        )
        benchmark.plot_seaborn_execution_time(
            df, 
            save_path=os.path.join(plots_dir, 'execution_time_seaborn.png'),
            title=titles['time']
        )
    
    print(f"\nAll plots saved to {plots_dir}/")
    print("Summary of output files:")
    print(f"  - Aggregated results: {csv_path}")
    print(f"  - Raw data: {raw_path}")
    if args.plot_format in ['plotly', 'both']:
        print("  - Plotly plots: speedup_linear.png/html, efficiency.png/html, execution_time.png/html")
    if args.plot_format in ['seaborn', 'both']:
        print("  - Seaborn plots: speedup_seaborn.png, efficiency_seaborn.png, execution_time_seaborn.png")
    print("\nUse these plots to demonstrate parallel scaling in your manuscript!")


if __name__ == '__main__':
    main() 