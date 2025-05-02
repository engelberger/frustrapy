"""Visualization functions for benchmark results."""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# Constants for styling
PASTEL_COLORS = px.colors.qualitative.Pastel
PLOTLY_TITLE_FONT = dict(family="Arial", size=18, color="#444444")
PLOTLY_AXIS_FONT = dict(family="Arial", size=14, color="#444444")

# Configure matplotlib for consistent styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.titlesize'] = 18
sns.set_style("whitegrid")

# Helper function to aggregate benchmark data by PDB ID and CPU count
def _aggregate_benchmark_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate benchmark metrics by PDB ID and CPU count, averaging across residues."""
    # Dynamically select metrics to aggregate
    agg_funcs = {}
    for metric in ['time_s', 'speedup', 'efficiency']:
        if metric in df.columns:
            agg_funcs[metric] = ['mean', 'std']
    # Group and aggregate
    df_agg = df.groupby(['pdb_id', 'n_cpus']).agg(agg_funcs)
    # Flatten MultiIndex columns
    df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
    # Rename std columns to represent variation across residues
    rename_map = {f"{metric}_std": f"{metric}_residue_std" for metric in ['time_s', 'speedup', 'efficiency']}
    df_agg = df_agg.rename(columns={k: v for k, v in rename_map.items() if k in df_agg.columns})
    # Reset index to make grouping columns regular
    df_agg = df_agg.reset_index()
    return df_agg

# Helper to generate title
def _generate_plot_title(df: pd.DataFrame, base_title: str) -> str:
    """Generate plot title, indicating residue aggregation if applicable."""
    residues = sorted(df['residue'].unique())
    if len(residues) > 1:
        residue_str = ', '.join(map(str, residues))
        return f"{base_title} (Avg. across Residues: {residue_str})"
    elif len(residues) == 1:
         return f"{base_title} (Residue: {residues[0]})"
    else:
        return base_title

# -------------------------------------------------------------------------
# Plotly visualization functions
# -------------------------------------------------------------------------

def plot_speedup_linear(
    df: pd.DataFrame, 
    title: str = "Performance Scaling with Multi-core Processing"
) -> go.Figure:
    """
    Generate a publication-quality speedup plot with Plotly.
    Handles multiple residues by averaging and showing variation across residues.
    """
    if 'speedup' not in df.columns:
        print("Warning: 'speedup' column not found. Skipping speedup plot.")
        return go.Figure() # Return empty figure

    df_agg = _aggregate_benchmark_data(df)
    plot_title = _generate_plot_title(df, title)

    # Create figure
    fig = go.Figure()
    
    # Prepare colors
    pdb_ids = df_agg['pdb_id'].unique()
    colors = PASTEL_COLORS[:len(pdb_ids)]
    color_map = dict(zip(pdb_ids, colors))
    
    # Plot data for each PDB ID
    for pdb_id in pdb_ids:
        pdb_data = df_agg[df_agg['pdb_id'] == pdb_id].sort_values('n_cpus')
        
        # Add error bars using std dev across residues
        error_y = None
        if 'speedup_residue_std' in pdb_data.columns:
            error_y = dict(
                type='data',
                array=pdb_data['speedup_residue_std'].fillna(0), # Use 0 error if NaN (e.g., single residue)
                visible=True,
                thickness=1.5,
                width=3,
                color=color_map[pdb_id]
            )
        
        # Add line+markers trace using mean speedup
        fig.add_trace(go.Scatter(
            x=pdb_data['n_cpus'],
            y=pdb_data['speedup_mean'] if 'speedup_mean' in pdb_data.columns else pdb_data['speedup'], # Use mean if exists
            mode='lines+markers',
            name=pdb_id,
            line=dict(color=color_map[pdb_id], width=2),
            marker=dict(size=8, color=color_map[pdb_id]),
            error_y=error_y
        ))
    
    # Add ideal scaling line
    max_cpus = df_agg['n_cpus'].max()
    ideal_x = list(range(1, max_cpus + 1))
    fig.add_scatter(
        x=ideal_x,
        y=ideal_x,
        mode='lines', 
        name='Ideal Linear Scaling', 
        line=dict(dash='dash', color='rgba(0,0,0,0.7)', width=1.5)
    )
    
    # Enhance styling
    fig.update_layout(
        template='plotly_white',
        title={'text': plot_title, 'x': 0.5, 'xanchor': 'center'},
        title_font=PLOTLY_TITLE_FONT,
        font=PLOTLY_AXIS_FONT,
        xaxis_title='Number of CPU Cores',
        yaxis_title='Speedup Factor (×)',
        legend=dict(
            title='PDB ID',
            yanchor="top", y=0.98, xanchor="left", x=0.02, # Move legend to left
            bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        margin=dict(t=120, b=60, l=60, r=20)
    )
    fig.update_xaxes(dtick=1)
    # Ensure y-axis starts at 0 and accommodates max speedup
    max_speedup = (df_agg['speedup_mean'] if 'speedup_mean' in df_agg.columns else df_agg['speedup']).max()
    max_y = max(max_cpus, max_speedup if pd.notna(max_speedup) else 0) 
    fig.update_yaxes(range=[0, max_y * 1.1])
    
    return fig


def plot_efficiency(
    df: pd.DataFrame, 
    title: str = "Parallel Efficiency of Mutational Analysis"
) -> go.Figure:
    """
    Generate a publication-quality efficiency plot with Plotly.
    Handles multiple residues by averaging and showing variation across residues.
    """
    if 'efficiency' not in df.columns:
        print("Warning: 'efficiency' column not found. Skipping efficiency plot.")
        return go.Figure()

    df_agg = _aggregate_benchmark_data(df)
    plot_title = _generate_plot_title(df, title)

    # Create figure
    fig = go.Figure()
    
    # Prepare colors
    pdb_ids = df_agg['pdb_id'].unique()
    colors = PASTEL_COLORS[:len(pdb_ids)]
    color_map = dict(zip(pdb_ids, colors))
    
    # Plot data for each PDB ID
    for pdb_id in pdb_ids:
        pdb_data = df_agg[df_agg['pdb_id'] == pdb_id].sort_values('n_cpus')
        
        # Add error bars using std dev across residues
        error_y = None
        if 'efficiency_residue_std' in pdb_data.columns:
            error_y = dict(
                type='data',
                array=pdb_data['efficiency_residue_std'].fillna(0),
                visible=True,
                thickness=1.5,
                width=3,
                color=color_map[pdb_id]
            )
        
        # Add line+markers trace using mean efficiency
        fig.add_trace(go.Scatter(
            x=pdb_data['n_cpus'],
            y=pdb_data['efficiency_mean'] if 'efficiency_mean' in pdb_data.columns else pdb_data['efficiency'],
            mode='lines+markers',
            name=pdb_id,
            line=dict(color=color_map[pdb_id], width=2),
            marker=dict(size=8, color=color_map[pdb_id]),
            error_y=error_y
        ))
    
    # Add ideal efficiency line
    fig.add_hline(
        y=1, 
        line=dict(dash='dash', color='rgba(0,0,0,0.7)', width=1.5), 
        name="Ideal Efficiency (100%)" # Updated label
    )
    
    # Enhance styling
    fig.update_layout(
        template='plotly_white',
        title={'text': plot_title, 'x': 0.5, 'xanchor': 'center'},
        title_font=PLOTLY_TITLE_FONT,
        font=PLOTLY_AXIS_FONT,
        xaxis_title='Number of CPU Cores',
        yaxis_title='Parallel Efficiency',
        yaxis_tickformat='.0%', # Format as percentage
        legend=dict(
            title='PDB ID',
            yanchor="top", y=0.98, xanchor="right", x=0.98,
            bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        margin=dict(t=120, b=60, l=60, r=20)
    )
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(range=[0, 1.1]) # Keep range 0 to 110%
    
    return fig


def plot_execution_time(
    df: pd.DataFrame, 
    title: str = "Computation Time Reduction with Parallel Processing"
) -> go.Figure:
    """
    Generate a publication-quality execution time plot with Plotly.
    Handles multiple residues by averaging and showing variation across residues.
    """
    if 'time_s' not in df.columns:
         print("Warning: 'time_s' column not found. Skipping execution time plot.")
         return go.Figure()

    df_agg = _aggregate_benchmark_data(df)
    plot_title = _generate_plot_title(df, title)

    # Create figure
    fig = go.Figure()
    
    # Prepare colors
    pdb_ids = df_agg['pdb_id'].unique()
    colors = PASTEL_COLORS[:len(pdb_ids)]
    color_map = dict(zip(pdb_ids, colors))
    
    # Plot data for each PDB ID
    for pdb_id in pdb_ids:
        pdb_data = df_agg[df_agg['pdb_id'] == pdb_id].sort_values('n_cpus')
        
        # Add error bars using std dev across residues
        error_y = None
        if 'time_s_residue_std' in pdb_data.columns:
            error_y = dict(
                type='data',
                array=pdb_data['time_s_residue_std'].fillna(0),
                visible=True,
                thickness=1.5,
                width=3,
                color=color_map[pdb_id]
            )
        
        # Add line+markers trace using mean time
        fig.add_trace(go.Scatter(
            x=pdb_data['n_cpus'],
            y=pdb_data['time_s_mean'] if 'time_s_mean' in pdb_data.columns else pdb_data['time_s'],
            mode='lines+markers',
            name=pdb_id,
            line=dict(color=color_map[pdb_id], width=2),
            marker=dict(size=8, color=color_map[pdb_id]),
            error_y=error_y
        ))
    
    # Enhance styling
    fig.update_layout(
        template='plotly_white',
        title={'text': plot_title, 'x': 0.5, 'xanchor': 'center'},
        title_font=PLOTLY_TITLE_FONT,
        font=PLOTLY_AXIS_FONT,
        xaxis_title='Number of CPU Cores',
        yaxis_title='Execution Time (s)',
        legend=dict(
            title='PDB ID',
            yanchor="top", y=0.98, xanchor="right", x=0.98,
            bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        margin=dict(t=120, b=60, l=60, r=20)
    )
    fig.update_xaxes(dtick=1)
    # Set linear scale starting from 0 for both axes
    max_cpus = df_agg['n_cpus'].max()
    max_time = (df_agg['time_s_mean'] if 'time_s_mean' in df_agg.columns else df_agg['time_s']).max()
    fig.update_xaxes(range=[0, max_cpus * 1.1])
    fig.update_yaxes(type='linear', range=[0, max_time * 1.1]) # Change type to linear and set range
    
    return fig


# -------------------------------------------------------------------------
# Seaborn visualization functions
# -------------------------------------------------------------------------

def plot_seaborn_speedup(
    df: pd.DataFrame, 
    save_path: str = None, 
    title: str = "Performance Scaling with Multi-core Processing",
    dpi: int = 300
) -> plt.Figure:
    """
    Generate a publication-quality Seaborn speedup plot with error bars.
    Handles multiple residues by averaging and showing variation across residues.
    """
    if 'speedup' not in df.columns:
        print("Warning: 'speedup' column not found. Skipping speedup plot.")
        return plt.figure() # Return empty figure

    df_agg = _aggregate_benchmark_data(df)
    plot_title = _generate_plot_title(df, title)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create palette matching number of unique pdb_ids
    num_pdbs = df_agg['pdb_id'].nunique()
    palette = sns.color_palette("pastel", n_colors=num_pdbs)
    
    # Use errorbar plot showing variation across residues
    y_col = 'speedup_mean' if 'speedup_mean' in df_agg.columns else 'speedup'
    yerr_col = 'speedup_residue_std' if 'speedup_residue_std' in df_agg.columns else None

    # Group by PDB ID for plotting
    for i, (pdb_id, group_data) in enumerate(df_agg.groupby('pdb_id')):
        yerr_data = group_data[yerr_col].fillna(0) if yerr_col else None
        ax.errorbar(
            x=group_data['n_cpus'], 
            y=group_data[y_col],
            yerr=yerr_data,
            marker='o',
            markersize=8,
            linewidth=2,
            elinewidth=1.5,
            capsize=4,
            label=pdb_id,
            color=palette[i]
        )

    # Add ideal scaling line
    max_cpus = df_agg['n_cpus'].max()
    x_ideal = np.arange(1, max_cpus + 1)
    plt.plot(
        x_ideal, 
        x_ideal, 
        '--', 
        color='black', 
        alpha=0.7, 
        linewidth=1.5, 
        label='Ideal Linear Scaling'
    )
    
    # Styling
    plt.title(plot_title, fontsize=18, pad=20, color="#444444")
    plt.xlabel('Number of CPU Cores', fontsize=14, labelpad=10)
    plt.ylabel('Speedup Factor (×)', fontsize=14, labelpad=10)
    plt.xticks(df_agg['n_cpus'].unique())
    max_y = max(max_cpus, df_agg[y_col].max() if not df_agg[y_col].empty else 0)
    plt.ylim(0, max_y * 1.1)
    
    # Improve legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles, 
        labels=labels, 
        title='PDB ID', 
        frameon=True, 
        loc='upper left', 
        fontsize=12
    )
    
    # Grid and background styling
    ax.grid(True, alpha=0.3)
    sns.despine(left=False, bottom=False)
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # leave space for title
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_seaborn_efficiency(
    df: pd.DataFrame, 
    save_path: str = None, 
    title: str = "Parallel Efficiency of Mutational Analysis",
    dpi: int = 300
) -> plt.Figure:
    """
    Generate a publication-quality Seaborn efficiency plot with error bars.
    Handles multiple residues by averaging and showing variation across residues.
    """
    if 'efficiency' not in df.columns:
        print("Warning: 'efficiency' column not found. Skipping efficiency plot.")
        return plt.figure()

    df_agg = _aggregate_benchmark_data(df)
    plot_title = _generate_plot_title(df, title)
        
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create palette matching number of unique pdb_ids
    num_pdbs = df_agg['pdb_id'].nunique()
    palette = sns.color_palette("pastel", n_colors=num_pdbs)
    
    # Use errorbar plot showing variation across residues
    y_col = 'efficiency_mean' if 'efficiency_mean' in df_agg.columns else 'efficiency'
    yerr_col = 'efficiency_residue_std' if 'efficiency_residue_std' in df_agg.columns else None

    # Group by PDB ID for plotting
    for i, (pdb_id, group_data) in enumerate(df_agg.groupby('pdb_id')):
        yerr_data = group_data[yerr_col].fillna(0) if yerr_col else None
        ax.errorbar(
            x=group_data['n_cpus'], 
            y=group_data[y_col],
            yerr=yerr_data,
            marker='o',
            markersize=8,
            linewidth=2,
            elinewidth=1.5,
            capsize=4,
            label=pdb_id,
            color=palette[i]
        )
    
    # Add ideal efficiency line
    plt.axhline(
        y=1, 
        linestyle='--', 
        color='black', 
        alpha=0.7, 
        linewidth=1.5, 
        label='Ideal Efficiency (100%)' # Updated label
    )
    
    # Styling
    plt.title(plot_title, fontsize=18, pad=20, color="#444444")
    plt.xlabel('Number of CPU Cores', fontsize=14, labelpad=10)
    plt.ylabel('Parallel Efficiency', fontsize=14, labelpad=10)
    plt.xticks(df_agg['n_cpus'].unique())
    plt.ylim(0, 1.1)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Improve legend 
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles, 
        labels=labels, 
        title='PDB ID', 
        frameon=True, 
        loc='upper right', 
        fontsize=12
    )
    
    # Grid and background styling
    ax.grid(True, alpha=0.3)
    sns.despine(left=False, bottom=False)
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # leave space for title
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_seaborn_execution_time(
    df: pd.DataFrame, 
    save_path: str = None, 
    title: str = "Computation Time Reduction with Parallel Processing",
    dpi: int = 300
) -> plt.Figure:
    """
    Generate a publication-quality Seaborn execution time plot with error bars.
    Handles multiple residues by averaging and showing variation across residues.
    """
    if 'time_s' not in df.columns:
        print("Warning: 'time_s' column not found. Skipping execution time plot.")
        return plt.figure()

    df_agg = _aggregate_benchmark_data(df)
    plot_title = _generate_plot_title(df, title)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create palette matching number of unique pdb_ids  
    num_pdbs = df_agg['pdb_id'].nunique()
    palette = sns.color_palette("pastel", n_colors=num_pdbs)
    
    # Use errorbar plot showing variation across residues
    y_col = 'time_s_mean' if 'time_s_mean' in df_agg.columns else 'time_s'
    yerr_col = 'time_s_residue_std' if 'time_s_residue_std' in df_agg.columns else None

    # Group by PDB ID for plotting
    for i, (pdb_id, group_data) in enumerate(df_agg.groupby('pdb_id')):
         yerr_data = group_data[yerr_col].fillna(0) if yerr_col else None
         ax.errorbar(
            x=group_data['n_cpus'], 
            y=group_data[y_col],
            yerr=yerr_data,
            marker='o',
            markersize=8,
            linewidth=2,
            elinewidth=1.5,
            capsize=4,
            label=pdb_id,
            color=palette[i]
        )

    # Styling
    plt.title(plot_title, fontsize=18, pad=20, color="#444444") 
    plt.xlabel('Number of CPU Cores', fontsize=14, labelpad=10)
    plt.ylabel('Execution Time (s)', fontsize=14, labelpad=10)
    
    # Compute robust axis limits (avoid NaN/Inf)
    # Combine mean and error if available
    time_vals = df_agg[y_col].copy()
    if yerr_col and yerr_col in df_agg.columns:
        time_vals = time_vals + df_agg[yerr_col].fillna(0)
    # Drop invalid values
    valid_times = time_vals.dropna()[np.isfinite(time_vals.dropna())]
    if not valid_times.empty:
        max_cpus = df_agg['n_cpus'].max()
        max_time = valid_times.max()
        # Set x-axis ticks including zero
        x_ticks = [0] + sorted(df_agg['n_cpus'].unique())
        plt.xticks(x_ticks)
        plt.xlim(0, max_cpus * 1.1)
        plt.ylim(0, max_time * 1.1)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))  # integer ticks
    else:
        print("Warning: Cannot set axis limits for execution time (invalid data)")
    
    # Improve legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles, 
        labels=labels, 
        title='PDB ID', 
        frameon=True, 
        loc='upper right', 
        fontsize=12
    )
    
    # Grid and background styling  
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5) # Grid for linear scale
    sns.despine(left=False, bottom=False)
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # leave space for title
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig 