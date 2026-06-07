#!/usr/bin/env python
"""
Command-line tool for benchmarking FrustraPy mutation analysis performance.

Usage:
    python run_benchmark.py --input_pdbs 1m6k.pdb 1nfi.pdb --chain A --residue 75 --cpus 1 2 4 8 --repeats 3
    python run_benchmark.py --input_pdbs ./pdb_directory/ --chain A --residue 75 --cpus 1 2 4 8 --repeats 3
"""

import os
import glob
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


def safe_write_image(fig, path, **kwargs):
    """Write a static PNG, degrading gracefully if the optional `kaleido`
    engine is not installed.

    Static Plotly export needs the `kaleido` package (an optional `viz` extra).
    Its absence must not abort the whole benchmark run — the CSVs, interactive
    HTML plots, and seaborn PNGs are still produced. Warn once per missing file.
    """
    try:
        fig.write_image(path, **kwargs)
    except Exception as exc:  # kaleido missing, or any export backend failure
        console.print(
            f"[yellow]Skipping PNG export of {os.path.basename(path)}: {exc}.\n"
            f"Install the optional 'kaleido' package for static images "
            f"(pip install kaleido).[/yellow]"
        )

def process_input_path(path):
    """Process an input path to find PDB files.
    
    Args:
        path: A path to either a PDB file or a directory containing PDB files
        
    Returns:
        List of absolute paths to PDB files
    """
    path = os.path.abspath(path)
    
    if os.path.isfile(path):
        # Check if it's a PDB file
        if path.lower().endswith(('.pdb', '.ent')):
            return [path]
        else:
            console.print(f"[yellow]Warning: {path} is not a PDB file (must end with .pdb or .ent). Skipping.[/yellow]")
            return []
    elif os.path.isdir(path):
        # Find all PDB files in the directory
        pdb_files = glob.glob(os.path.join(path, "*.pdb"))
        pdb_files.extend(glob.glob(os.path.join(path, "*.ent")))
        if not pdb_files:
            console.print(f"[yellow]Warning: No PDB files found in directory: {path}[/yellow]")
        return pdb_files
    else:
        console.print(f"[yellow]Warning: Path does not exist: {path}[/yellow]")
        return []

def main():
    """Parse arguments and run benchmark with visualizations."""
    parser = argparse.ArgumentParser(
        description="Benchmark mutational analysis performance in FrustraPy."
    )
    parser.add_argument('--input_pdbs', nargs='+', required=True, 
                        help='Path(s) to PDB file(s) or directories containing PDB files')
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

    # Process input paths to get list of PDB files
    all_pdb_files = []
    for input_path in args.input_pdbs:
        pdb_files = process_input_path(input_path)
        all_pdb_files.extend(pdb_files)
    
    if not all_pdb_files:
        console.print("[bold red]Error: No valid PDB files found in the provided input paths.[/bold red]")
        sys.exit(1)
    
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
    
    # Format PDB file list for display
    pdb_files_display = ", ".join([os.path.basename(f) for f in all_pdb_files[:5]])
    if len(all_pdb_files) > 5:
        pdb_files_display += f" and {len(all_pdb_files) - 5} more"
    
    config_table.add_row("PDB Files", pdb_files_display)
    config_table.add_row("Number of PDB Files", str(len(all_pdb_files)))
    config_table.add_row("Chain", args.chain)
    config_table.add_row("Residues", ", ".join(map(str, args.residues)))
    config_table.add_row("CPU Configurations", ", ".join(map(str, args.cpus)))
    config_table.add_row("Repeats per Config", str(args.repeats))
    config_table.add_row("Results Directory", args.results_dir)
    config_table.add_row("Plot Format", args.plot_format)
    
    console.print(Panel(config_table, title="[bold]Benchmark Configuration[/bold]", border_style="cyan", padding=(1, 1)))
    
    # Display benchmark start message
    console.print("\n[bold]Starting benchmark...[/bold]")
    
    # Create a progress bar for PDB file processing
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold green]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        pdb_task = progress.add_task("[cyan]Processing PDB files...", total=len(all_pdb_files))
        
        # Run benchmarks for each PDB file and collect results
        all_results = []
        for pdb_file in all_pdb_files:
            pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]
            progress.update(pdb_task, description=f"[cyan]Processing {pdb_id}...")
            
            # Create subdirectory for this PDB's results
            pdb_results_dir = os.path.join(results_dir, pdb_id)
            os.makedirs(pdb_results_dir, exist_ok=True)
            
            # Run the benchmark for this PDB
            console.print(f"[bold green]Running benchmark calculations for {pdb_id}...[/bold green]")
            df = benchmark.run_benchmark(
                pdb_file=pdb_file,
                chain=args.chain,
                residues=args.residues,
                cpu_list=args.cpus,
                results_dir=pdb_results_dir,
                repeats=args.repeats,
                cautious=args.cautious
            )
            all_results.append(df)
            progress.update(pdb_task, advance=1)
    
    # Combine all results into one DataFrame
    console.print("[bold green]Combining results from all PDB files...[/bold green]")
    df_combined = pd.concat(all_results, ignore_index=True)
    
    # Save aggregated data
    csv_path = os.path.join(data_dir, args.output_csv)
    df_combined.to_csv(csv_path, index=False)
    
    # Create a file tree to show saved files
    file_tree = Tree("📁 [bold]Benchmark Results[/bold]")
    data_branch = file_tree.add("📁 [bold cyan]Data Files[/bold cyan]")
    data_branch.add(f"📊 [green]{os.path.basename(csv_path)}[/green] (Aggregated)")
    
    # Print results table with Rich
    console.print("\n[bold cyan]Benchmark Results Summary:[/bold cyan]")
    results_table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
    
    # Define the columns to display
    display_cols = ['pdb_id', 'residue', 'n_cpus', 'time_s', 'time_s_std', 'speedup', 'speedup_std', 'efficiency', 'efficiency_std']
    display_cols = [col for col in display_cols if col in df_combined.columns]
    
    # Add column titles with better names
    col_titles = {
        'pdb_id': "PDB ID",
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
        results_table.add_column(col_titles.get(col, col), justify="center" if col in ['pdb_id', 'residue'] else "right", no_wrap=True)
    
    # Add rows to the table
    for _, row in df_combined.iterrows():
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
    baseline_exists = 1 in df_combined['n_cpus'].values
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
        # Get unique PDB IDs
        pdb_ids = df_combined['pdb_id'].unique()
        
        # Calculate total number of plots to generate
        # Base plots: combined plots (1 set)
        base_plot_count = 1  # Execution time plot is always generated
        if baseline_exists:
            base_plot_count += 2  # Add speedup and efficiency if baseline exists
        
        # Total plots: combined plots + individual PDB plots * number of PDBs
        plot_count = base_plot_count * (len(pdb_ids) + 1)  # +1 for combined plots
        
        if args.plot_format == 'both':
            plot_count *= 2  # Double for both Plotly and Seaborn
            
        plot_task = progress.add_task("[cyan]Creating visualizations...", total=plot_count)
        
        # Plotly plots branch in the file tree
        plots_branch = file_tree.add("📁 [bold cyan]Visualization Plots[/bold cyan]")
        
        # Generate combined plots first
        if args.plot_format in ['plotly', 'both']:
            plotly_branch = plots_branch.add("📊 [bold]Plotly Interactive Plots[/bold]")
            combined_plotly = plotly_branch.add("📊 [bold]Combined PDB Plots[/bold]")
            
            if baseline_exists:
                # Combined speedup plot
                fig_su = benchmark.plot_speedup_linear(df_combined, title=f"{titles['speedup']} - All PDBs")
                safe_write_image(fig_su, os.path.join(plots_dir, 'combined_speedup_linear.png'), scale=2)
                fig_su.write_html(os.path.join(plots_dir, 'combined_speedup_linear.html'))
                combined_plotly.add("📈 [green]combined_speedup_linear.png/html[/green]")
                progress.update(plot_task, advance=1)
                
                # Combined efficiency plot
                fig_eff = benchmark.plot_efficiency(df_combined, title=f"{titles['efficiency']} - All PDBs")
                safe_write_image(fig_eff, os.path.join(plots_dir, 'combined_efficiency.png'), scale=2)
                fig_eff.write_html(os.path.join(plots_dir, 'combined_efficiency.html'))
                combined_plotly.add("📉 [green]combined_efficiency.png/html[/green]")
                progress.update(plot_task, advance=1)
            
            # Combined execution time plot (always generated)
            fig_time = benchmark.plot_execution_time(df_combined, title=f"{titles['time']} - All PDBs")
            safe_write_image(fig_time, os.path.join(plots_dir, 'combined_execution_time.png'), scale=2)
            fig_time.write_html(os.path.join(plots_dir, 'combined_execution_time.html'))
            combined_plotly.add("⏱️ [green]combined_execution_time.png/html[/green]")
            progress.update(plot_task, advance=1)
            
            # Generate individual PDB plots
            for pdb_id in pdb_ids:
                # Filter data for current PDB
                df_pdb = df_combined[df_combined['pdb_id'] == pdb_id]
                
                # Create PDB-specific branch
                pdb_plotly = plotly_branch.add(f"📊 [bold]{pdb_id} Plots[/bold]")
                
                if baseline_exists and 1 in df_pdb['n_cpus'].values:
                    # Individual speedup plot
                    fig_su = benchmark.plot_speedup_linear(df_pdb, title=f"{titles['speedup']} - {pdb_id}")
                    safe_write_image(fig_su, os.path.join(plots_dir, f'{pdb_id}_speedup_linear.png'), scale=2)
                    fig_su.write_html(os.path.join(plots_dir, f'{pdb_id}_speedup_linear.html'))
                    pdb_plotly.add(f"📈 [green]{pdb_id}_speedup_linear.png/html[/green]")
                    progress.update(plot_task, advance=1)
                    
                    # Individual efficiency plot
                    fig_eff = benchmark.plot_efficiency(df_pdb, title=f"{titles['efficiency']} - {pdb_id}")
                    safe_write_image(fig_eff, os.path.join(plots_dir, f'{pdb_id}_efficiency.png'), scale=2)
                    fig_eff.write_html(os.path.join(plots_dir, f'{pdb_id}_efficiency.html'))
                    pdb_plotly.add(f"📉 [green]{pdb_id}_efficiency.png/html[/green]")
                    progress.update(plot_task, advance=1)
                elif baseline_exists:
                    # Skip speedup/efficiency plots if no baseline for this PDB
                    progress.update(plot_task, advance=2)
                
                # Individual execution time plot
                fig_time = benchmark.plot_execution_time(df_pdb, title=f"{titles['time']} - {pdb_id}")
                safe_write_image(fig_time, os.path.join(plots_dir, f'{pdb_id}_execution_time.png'), scale=2)
                fig_time.write_html(os.path.join(plots_dir, f'{pdb_id}_execution_time.html'))
                pdb_plotly.add(f"⏱️ [green]{pdb_id}_execution_time.png/html[/green]")
                progress.update(plot_task, advance=1)
        
        # Seaborn plots
        if args.plot_format in ['seaborn', 'both']:
            seaborn_branch = plots_branch.add("📊 [bold]Seaborn Publication Plots[/bold]")
            combined_seaborn = seaborn_branch.add("📊 [bold]Combined PDB Plots[/bold]")
            
            if baseline_exists:
                # Combined speedup plot
                benchmark.plot_seaborn_speedup(
                    df_combined, 
                    save_path=os.path.join(plots_dir, 'combined_speedup_seaborn.png'),
                    title=f"{titles['speedup']} - All PDBs"
                )
                combined_seaborn.add("📈 [green]combined_speedup_seaborn.png[/green]")
                progress.update(plot_task, advance=1)
                
                # Combined efficiency plot
                benchmark.plot_seaborn_efficiency(
                    df_combined, 
                    save_path=os.path.join(plots_dir, 'combined_efficiency_seaborn.png'),
                    title=f"{titles['efficiency']} - All PDBs"
                )
                combined_seaborn.add("📉 [green]combined_efficiency_seaborn.png[/green]")
                progress.update(plot_task, advance=1)
            
            # Combined execution time plot
            benchmark.plot_seaborn_execution_time(
                df_combined, 
                save_path=os.path.join(plots_dir, 'combined_execution_time_seaborn.png'),
                title=f"{titles['time']} - All PDBs"
            )
            combined_seaborn.add("⏱️ [green]combined_execution_time_seaborn.png[/green]")
            progress.update(plot_task, advance=1)
            
            # Generate individual PDB plots
            for pdb_id in pdb_ids:
                # Filter data for current PDB
                df_pdb = df_combined[df_combined['pdb_id'] == pdb_id]
                
                # Create PDB-specific branch
                pdb_seaborn = seaborn_branch.add(f"📊 [bold]{pdb_id} Plots[/bold]")
                
                if baseline_exists and 1 in df_pdb['n_cpus'].values:
                    # Individual speedup plot
                    benchmark.plot_seaborn_speedup(
                        df_pdb, 
                        save_path=os.path.join(plots_dir, f'{pdb_id}_speedup_seaborn.png'),
                        title=f"{titles['speedup']} - {pdb_id}"
                    )
                    pdb_seaborn.add(f"📈 [green]{pdb_id}_speedup_seaborn.png[/green]")
                    progress.update(plot_task, advance=1)
                    
                    # Individual efficiency plot
                    benchmark.plot_seaborn_efficiency(
                        df_pdb, 
                        save_path=os.path.join(plots_dir, f'{pdb_id}_efficiency_seaborn.png'),
                        title=f"{titles['efficiency']} - {pdb_id}"
                    )
                    pdb_seaborn.add(f"📉 [green]{pdb_id}_efficiency_seaborn.png[/green]")
                    progress.update(plot_task, advance=1)
                elif baseline_exists:
                    # Skip speedup/efficiency plots if no baseline for this PDB
                    progress.update(plot_task, advance=2)
                
                # Individual execution time plot
                benchmark.plot_seaborn_execution_time(
                    df_pdb, 
                    save_path=os.path.join(plots_dir, f'{pdb_id}_execution_time_seaborn.png'),
                    title=f"{titles['time']} - {pdb_id}"
                )
                pdb_seaborn.add(f"⏱️ [green]{pdb_id}_execution_time_seaborn.png[/green]")
                progress.update(plot_task, advance=1)
    
    # Print file summary tree
    console.print(file_tree)
    
    # Final success message
    console.print(Panel.fit(
        f"[bold green]Benchmark completed successfully for {len(all_pdb_files)} PDB files![/bold green]\n\n"
        "[cyan]Individual plots for each PDB and combined plots are available in the results directory.[/cyan]\n"
        "[dim]The plots are ready to be used in your manuscript or presentation.[/dim]",
        border_style="green",
        title="✅ Complete",
        padding=(1, 2)
    ))


if __name__ == '__main__':
    main() 