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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text
from rich.style import Style
from rich.columns import Columns
from rich import box
from rich.tree import Tree
from rich.markup import escape

# Initialize Rich console
console = Console()

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Parse arguments and run benchmark with visualizations."""
    parser = argparse.ArgumentParser(
        description="Benchmark mutational analysis performance in FrustraPy."
    )
    parser.add_argument('--pdb_file', required=True, help='Path to the PDB file')
    parser.add_argument('--chain', required=True, help='Chain identifier')
    parser.add_argument('--residue', dest='residues', nargs='+', type=int, required=True, help='Residue number(s) to mutate')
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
    
    # Display a beautiful header
    console.print(Panel.fit(
        "[bold cyan]FrustraPy Benchmark Tool[/bold cyan]\n"
        "[dim]Analyze parallel scaling of mutational analysis[/dim]",
        box=box.ROUNDED,
        border_style="blue",
        padding=(1, 2),
        title="💻 Multi-Core Performance",
        subtitle="🧬 Protein Frustration Analysis"
    ))
    
    # Report cautious mode status with styled text
    mode_style = "green" if args.cautious else "yellow"
    mode_text = "[green]enabled: skipping already recorded runs" if args.cautious else "[yellow]disabled: rerunning all benchmarks"
    console.print(f"[bold blue]Cautious mode:[/bold blue] {mode_text}")
    
    # Build configuration panel
    config_table = Table(box=box.SIMPLE, show_header=False)
    config_table.add_column("Parameter", style="bold cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("PDB File", os.path.basename(args.pdb_file))
    config_table.add_row("Chain", args.chain)
    config_table.add_row("Residues", ", ".join(map(str, args.residues)))
    config_table.add_row("CPU Configurations", ", ".join(map(str, args.cpus)))
    config_table.add_row("Repeats per Config", str(args.repeats))
    config_table.add_row("Results Directory", args.results_dir)
    config_table.add_row("Plot Format", args.plot_format)
    
    console.print(Panel(config_table, title="[bold]Benchmark Configuration[/bold]", border_style="cyan", padding=(1, 1)))
    
    # Display benchmark start message
    console.print("\n[bold]Starting benchmark...[/bold]")
    
    # Run the benchmark
    console.print("[bold green]Running benchmark calculations...[/bold green]")
    df = benchmark.run_benchmark(
        pdb_file=args.pdb_file,
        chain=args.chain,
        residues=args.residues,
        cpu_list=args.cpus,
        results_dir=results_dir,
        repeats=args.repeats,
        cautious=args.cautious
    )
    console.print("[bold green]Benchmark calculations completed![/bold green]")
    
    # Save aggregated data
    csv_path = os.path.join(data_dir, args.output_csv)
    df.to_csv(csv_path, index=False)
    
    # Also save raw data
    raw_path = os.path.join(data_dir, 'raw_benchmark_data.csv')
    
    # Create a file tree to show saved files
    file_tree = Tree("📁 [bold]Benchmark Results[/bold]")
    data_branch = file_tree.add("📁 [bold cyan]Data Files[/bold cyan]")
    data_branch.add(f"📊 [green]{os.path.basename(csv_path)}[/green] (Aggregated)")
    if os.path.exists(raw_path):
        data_branch.add(f"📊 [green]{os.path.basename(raw_path)}[/green] (Raw data)")
    
    # Print results table with Rich
    console.print("\n[bold cyan]Benchmark Results Summary:[/bold cyan]")
    results_table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
    
    # Define the columns to display
    display_cols = ['residue', 'n_cpus', 'time_s', 'time_s_std', 'speedup', 'speedup_std', 'efficiency', 'efficiency_std']
    display_cols = [col for col in display_cols if col in df.columns]
    
    # Add column titles with better names
    col_titles = {
        'residue': "Residue",
        'n_cpus': "CPU Cores",
        'time_s': "Time (s)",
        'time_s_std': "± Std Dev",
        'speedup': "Speedup",
        'speedup_std': "± Std Dev",
        'efficiency': "Efficiency",
        'efficiency_std': "± Std Dev"
    }
    
    # Add columns to table with proper formatting
    for col in display_cols:
        fmt = ".3f" if col.endswith('_std') or col in ['time_s', 'speedup', 'efficiency'] else ""
        results_table.add_column(col_titles.get(col, col), justify="center" if col == 'residue' else "right", no_wrap=True)
    
    # Add rows to the table
    for _, row in df.iterrows():
        values = []
        for col in display_cols:
            val = row[col]
            if isinstance(val, (int, float)):
                if col.endswith('_std') or col in ['time_s', 'speedup', 'efficiency']:
                    val = f"{val:.3f}"
                elif col == 'n_cpus':
                    val = str(int(val))
                else:
                    val = str(val)
            else:
                val = str(val)
            values.append(val)
        results_table.add_row(*values)
    
    console.print(results_table)
    
    # Titles for publication plots
    titles = {
        'speedup': "Performance Scaling with Multi-core Processing",
        'efficiency': "Parallel Efficiency of Mutational Analysis",
        'time': "Computation Time Reduction with Parallel Processing"
    }
    
    # Generate plots
    console.print("\n[bold cyan]Generating visualization plots...[/bold cyan]")

    # Check if baseline data (n_cpus=1) is available for speedup/efficiency plots
    baseline_exists = 1 in df['n_cpus'].values
    if not baseline_exists:
        console.print(Panel(
            "[yellow]Baseline data (n_cpus=1) not found.\n"
            "Speedup and efficiency plots require baseline data and will be skipped.\n"
            "Please include '--cpus 1' in your command to generate these plots.[/yellow]",
            border_style="yellow",
            title="⚠️ Warning",
            padding=(1, 1)
        ))
    
    # Create a progress display for plot generation
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold green]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        # Calculate total number of plots to generate
        plot_count = 1  # Execution time plot is always generated
        if baseline_exists:
            plot_count += 2  # Add speedup and efficiency if baseline exists
        
        if args.plot_format == 'both':
            plot_count *= 2  # Double for both Plotly and Seaborn
            
        plot_task = progress.add_task("[cyan]Creating visualizations...", total=plot_count)
        
        # Plotly plots
        plots_branch = file_tree.add("📁 [bold cyan]Visualization Plots[/bold cyan]")
        
        if args.plot_format in ['plotly', 'both']:
            plotly_branch = plots_branch.add("📊 [bold]Plotly Interactive Plots[/bold]")
            if baseline_exists:
                # Speedup plot
                fig_su = benchmark.plot_speedup_linear(df, title=titles['speedup'])
                fig_su.write_image(os.path.join(plots_dir, 'speedup_linear.png'), scale=2)
                fig_su.write_html(os.path.join(plots_dir, 'speedup_linear.html'))
                plotly_branch.add("📈 [green]speedup_linear.png/html[/green]")
                progress.update(plot_task, advance=1)
                
                # Efficiency plot
                fig_eff = benchmark.plot_efficiency(df, title=titles['efficiency'])
                fig_eff.write_image(os.path.join(plots_dir, 'efficiency.png'), scale=2)
                fig_eff.write_html(os.path.join(plots_dir, 'efficiency.html'))
                plotly_branch.add("📉 [green]efficiency.png/html[/green]")
                progress.update(plot_task, advance=1)
            
            # Execution time plot (always generated)
            fig_time = benchmark.plot_execution_time(df, title=titles['time'])
            fig_time.write_image(os.path.join(plots_dir, 'execution_time.png'), scale=2)
            fig_time.write_html(os.path.join(plots_dir, 'execution_time.html'))
            plotly_branch.add("⏱️ [green]execution_time.png/html[/green]")
            progress.update(plot_task, advance=1)
        
        # Seaborn plots
        if args.plot_format in ['seaborn', 'both']:
            seaborn_branch = plots_branch.add("📊 [bold]Seaborn Publication Plots[/bold]")
            if baseline_exists:
                benchmark.plot_seaborn_speedup(
                    df, 
                    save_path=os.path.join(plots_dir, 'speedup_seaborn.png'),
                    title=titles['speedup']
                )
                seaborn_branch.add("📈 [green]speedup_seaborn.png[/green]")
                progress.update(plot_task, advance=1)
                
                benchmark.plot_seaborn_efficiency(
                    df, 
                    save_path=os.path.join(plots_dir, 'efficiency_seaborn.png'),
                    title=titles['efficiency']
                )
                seaborn_branch.add("📉 [green]efficiency_seaborn.png[/green]")
                progress.update(plot_task, advance=1)
                
            # Execution time plot (always generated)
            benchmark.plot_seaborn_execution_time(
                df, 
                save_path=os.path.join(plots_dir, 'execution_time_seaborn.png'),
                title=titles['time']
            )
            seaborn_branch.add("⏱️ [green]execution_time_seaborn.png[/green]")
            progress.update(plot_task, advance=1)
    
    # Print file summary tree
    console.print(file_tree)
    
    # Final success message
    console.print(Panel.fit(
        "[bold green]Benchmark completed successfully![/bold green]\n\n"
        "[cyan]The plots are ready to be used in your manuscript or presentation.[/cyan]\n"
        "[dim]Run with different CPU configurations to explore parallel scaling behavior.[/dim]",
        border_style="green",
        title="✅ Complete",
        padding=(1, 2)
    ))


if __name__ == '__main__':
    main() 