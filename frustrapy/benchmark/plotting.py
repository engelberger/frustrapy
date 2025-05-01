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

# -------------------------------------------------------------------------
# Plotly visualization functions
# -------------------------------------------------------------------------

def plot_speedup_linear(
    df: pd.DataFrame, 
    title: str = "Performance Scaling with Multi-core Processing"
) -> go.Figure:
    """
    Generate a publication-quality speedup plot with Plotly.
    
    Args:
        df: DataFrame with benchmark results, including speedup_std if available
        title: Plot title for publication
    
    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Prepare colors
    pdb_ids = df['pdb_id'].unique()
    colors = PASTEL_COLORS[:len(pdb_ids)]
    color_map = dict(zip(pdb_ids, colors))
    
    # Plot data for each PDB ID
    for pdb_id in pdb_ids:
        pdb_data = df[df['pdb_id'] == pdb_id].sort_values('n_cpus')
        
        # Add error bars if standard deviation is available
        error_y = None
        if 'speedup_std' in pdb_data.columns:
            error_y = dict(
                type='data',
                array=pdb_data['speedup_std'],
                visible=True,
                thickness=1.5,
                width=3,
                color=color_map[pdb_id]
            )
        
        # Add line+markers trace
        fig.add_trace(go.Scatter(
            x=pdb_data['n_cpus'],
            y=pdb_data['speedup'],
            mode='lines+markers',
            name=pdb_id,
            line=dict(color=color_map[pdb_id], width=2),
            marker=dict(size=8, color=color_map[pdb_id]),
            error_y=error_y
        ))
    
    # Add ideal scaling line
    max_cpus = df['n_cpus'].max()
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
        title=title,
        title_font=PLOTLY_TITLE_FONT,
        font=PLOTLY_AXIS_FONT,
        xaxis_title='Number of CPU Cores',
        yaxis_title='Speedup Factor (×)',
        legend=dict(
            yanchor="top", y=0.98, xanchor="right", x=0.98,
            bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        margin=dict(t=80, b=60, l=60, r=20)
    )
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(range=[0, max_cpus * 1.1])
    
    return fig


def plot_efficiency(
    df: pd.DataFrame, 
    title: str = "Parallel Efficiency of Mutational Analysis"
) -> go.Figure:
    """
    Generate a publication-quality efficiency plot with Plotly.
    
    Args:
        df: DataFrame with benchmark results, including efficiency_std if available
        title: Plot title for publication
    
    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Prepare colors
    pdb_ids = df['pdb_id'].unique()
    colors = PASTEL_COLORS[:len(pdb_ids)]
    color_map = dict(zip(pdb_ids, colors))
    
    # Plot data for each PDB ID
    for pdb_id in pdb_ids:
        pdb_data = df[df['pdb_id'] == pdb_id].sort_values('n_cpus')
        
        # Add error bars if standard deviation is available
        error_y = None
        if 'efficiency_std' in pdb_data.columns:
            error_y = dict(
                type='data',
                array=pdb_data['efficiency_std'],
                visible=True,
                thickness=1.5,
                width=3,
                color=color_map[pdb_id]
            )
        
        # Add line+markers trace
        fig.add_trace(go.Scatter(
            x=pdb_data['n_cpus'],
            y=pdb_data['efficiency'],
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
        name="Ideal Efficiency"
    )
    
    # Enhance styling
    fig.update_layout(
        template='plotly_white',
        title=title,
        title_font=PLOTLY_TITLE_FONT,
        font=PLOTLY_AXIS_FONT,
        xaxis_title='Number of CPU Cores',
        yaxis_title='Parallel Efficiency',
        legend=dict(
            yanchor="top", y=0.98, xanchor="right", x=0.98,
            bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        margin=dict(t=80, b=60, l=60, r=20)
    )
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(range=[0, 1.1], tickformat='.2f')
    
    return fig


def plot_execution_time(
    df: pd.DataFrame, 
    title: str = "Computation Time Reduction with Parallel Processing"
) -> go.Figure:
    """
    Generate a publication-quality execution time plot with Plotly.
    
    Args:
        df: DataFrame with benchmark results, including time_s_std if available
        title: Plot title for publication
    
    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Prepare colors
    pdb_ids = df['pdb_id'].unique()
    colors = PASTEL_COLORS[:len(pdb_ids)]
    color_map = dict(zip(pdb_ids, colors))
    
    # Plot data for each PDB ID
    for pdb_id in pdb_ids:
        pdb_data = df[df['pdb_id'] == pdb_id].sort_values('n_cpus')
        
        # Add error bars if standard deviation is available
        error_y = None
        if 'time_s_std' in pdb_data.columns:
            error_y = dict(
                type='data',
                array=pdb_data['time_s_std'],
                visible=True,
                thickness=1.5,
                width=3,
                color=color_map[pdb_id]
            )
        
        # Add line+markers trace
        fig.add_trace(go.Scatter(
            x=pdb_data['n_cpus'],
            y=pdb_data['time_s'],
            mode='lines+markers',
            name=pdb_id,
            line=dict(color=color_map[pdb_id], width=2),
            marker=dict(size=8, color=color_map[pdb_id]),
            error_y=error_y
        ))
    
    # Enhance styling
    fig.update_layout(
        template='plotly_white',
        title=title,
        title_font=PLOTLY_TITLE_FONT,
        font=PLOTLY_AXIS_FONT,
        xaxis_title='Number of CPU Cores',
        yaxis_title='Execution Time (s)',
        legend=dict(
            yanchor="top", y=0.98, xanchor="right", x=0.98,
            bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        margin=dict(t=80, b=60, l=60, r=20)
    )
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(type='log')
    
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
    
    Args:
        df: DataFrame with benchmark results, including speedup_std if available
        save_path: Path to save the figure (if None, figure is not saved)
        title: Plot title for publication
        dpi: DPI for saved figure
    
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create palette matching number of unique pdb_ids
    num_pdbs = df['pdb_id'].nunique()
    palette = sns.color_palette("pastel", n_colors=num_pdbs)
    
    # If standard deviation is available, use errorbar plot
    if 'speedup_std' in df.columns:
        # Group by PDB ID for plotting
        for i, (pdb_id, group_data) in enumerate(df.groupby('pdb_id')):
            ax.errorbar(
                x=group_data['n_cpus'], 
                y=group_data['speedup'],
                yerr=group_data['speedup_std'],
                marker='o',
                markersize=8,
                linewidth=2,
                elinewidth=1.5,
                capsize=4,
                label=pdb_id,
                color=palette[i]
            )
    else:
        # Otherwise fall back to regular lineplot
        sns.lineplot(
            data=df, 
            x='n_cpus', 
            y='speedup', 
            hue='pdb_id', 
            marker='o', 
            markersize=8, 
            linewidth=2, 
            palette=palette,
            ax=ax
        )
    
    # Add ideal scaling line
    max_cpus = df['n_cpus'].max()
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
    plt.title(title, fontsize=18, pad=20, color="#444444")
    plt.xlabel('Number of CPU Cores', fontsize=14, labelpad=10)
    plt.ylabel('Speedup Factor (×)', fontsize=14, labelpad=10)
    plt.xticks(df['n_cpus'].unique())
    plt.ylim(0, max_cpus * 1.1)
    
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
    plt.tight_layout()
    
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
    
    Args:
        df: DataFrame with benchmark results, including efficiency_std if available
        save_path: Path to save the figure (if None, figure is not saved)
        title: Plot title for publication
        dpi: DPI for saved figure
    
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create palette matching number of unique pdb_ids
    num_pdbs = df['pdb_id'].nunique()
    palette = sns.color_palette("pastel", n_colors=num_pdbs)
    
    # If standard deviation is available, use errorbar plot
    if 'efficiency_std' in df.columns:
        # Group by PDB ID for plotting
        for i, (pdb_id, group_data) in enumerate(df.groupby('pdb_id')):
            ax.errorbar(
                x=group_data['n_cpus'], 
                y=group_data['efficiency'],
                yerr=group_data['efficiency_std'],
                marker='o',
                markersize=8,
                linewidth=2,
                elinewidth=1.5,
                capsize=4,
                label=pdb_id,
                color=palette[i]
            )
    else:
        # Otherwise fall back to regular lineplot
        sns.lineplot(
            data=df, 
            x='n_cpus', 
            y='efficiency', 
            hue='pdb_id', 
            marker='o', 
            markersize=8, 
            linewidth=2, 
            palette=palette,
            ax=ax
        )
    
    # Add ideal efficiency line
    plt.axhline(
        y=1, 
        linestyle='--', 
        color='black', 
        alpha=0.7, 
        linewidth=1.5, 
        label='Ideal Efficiency'
    )
    
    # Styling
    plt.title(title, fontsize=18, pad=20, color="#444444")
    plt.xlabel('Number of CPU Cores', fontsize=14, labelpad=10)
    plt.ylabel('Parallel Efficiency', fontsize=14, labelpad=10)
    plt.xticks(df['n_cpus'].unique())
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
    plt.tight_layout()
    
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
    
    Args:
        df: DataFrame with benchmark results, including time_s_std if available
        save_path: Path to save the figure (if None, figure is not saved)
        title: Plot title for publication
        dpi: DPI for saved figure
    
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create palette matching number of unique pdb_ids  
    num_pdbs = df['pdb_id'].nunique()
    palette = sns.color_palette("pastel", n_colors=num_pdbs)
    
    # If standard deviation is available, use errorbar plot
    if 'time_s_std' in df.columns:
        # Group by PDB ID for plotting
        for i, (pdb_id, group_data) in enumerate(df.groupby('pdb_id')):
            ax.errorbar(
                x=group_data['n_cpus'], 
                y=group_data['time_s'],
                yerr=group_data['time_s_std'],
                marker='o',
                markersize=8,
                linewidth=2,
                elinewidth=1.5,
                capsize=4,
                label=pdb_id,
                color=palette[i]
            )
    else:
        # Otherwise fall back to regular lineplot
        sns.lineplot(
            data=df, 
            x='n_cpus', 
            y='time_s', 
            hue='pdb_id', 
            marker='o', 
            markersize=8, 
            linewidth=2, 
            palette=palette,
            ax=ax
        )
    
    # Styling
    plt.title(title, fontsize=18, pad=20, color="#444444") 
    plt.xlabel('Number of CPU Cores', fontsize=14, labelpad=10)
    plt.ylabel('Execution Time (s)', fontsize=14, labelpad=10)
    plt.xticks(df['n_cpus'].unique())
    
    # Use log scale for y-axis
    plt.yscale('log')
    
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
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig 