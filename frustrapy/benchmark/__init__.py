"""Benchmark module for measuring and visualizing FrustraPy performance."""

# Export main functions for easy imports
from .benchmark import run_benchmark, get_raw_benchmark_data
from .plotting import (
    plot_speedup_linear,
    plot_efficiency,
    plot_execution_time,
    plot_seaborn_speedup,
    plot_seaborn_efficiency,
    plot_seaborn_execution_time
)

__all__ = [
    'run_benchmark',
    'get_raw_benchmark_data',
    'plot_speedup_linear',
    'plot_efficiency',
    'plot_execution_time', 
    'plot_seaborn_speedup',
    'plot_seaborn_efficiency',
    'plot_seaborn_execution_time'
] 