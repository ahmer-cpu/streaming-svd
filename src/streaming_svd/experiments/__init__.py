"""Experiment runners comparing cold-start vs warm-start rSVD.

Five runners are available, each with a CLI entry point:

run_series    -- Control experiment: independent random matrices
                 python -m streaming_svd.experiments.run_series

run_synthetic -- Dual-regime: additive noise + rotating subspace
                 python -m streaming_svd.experiments.run_synthetic

run_weather   -- Single-variable real data experiment
                 python -m streaming_svd.experiments.run_weather

run_sweep     -- Parameter grid sweep across all three regimes
                 python -m streaming_svd.experiments.run_sweep

hurricane     -- Comprehensive 13-variable × 48-timestep pipeline
                 python -m streaming_svd.experiments.hurricane.collect
                 python -m streaming_svd.experiments.hurricane.analyze
                 python -m streaming_svd.experiments.hurricane.plot
"""

from .run_series import run_series_experiment
from .run_synthetic import run_experiment_additive, run_experiment_rotating
from .run_weather import run_weather_experiment
from .run_sweep import run_sweep
from .hurricane import (
    collect_hurricane_experiment,
    analyze_hurricane_results,
    plot_per_variable,
    plot_cross_variable,
)

__all__ = [
    'run_series_experiment',
    'run_experiment_additive',
    'run_experiment_rotating',
    'run_weather_experiment',
    'run_sweep',
    'collect_hurricane_experiment',
    'analyze_hurricane_results',
    'plot_per_variable',
    'plot_cross_variable',
]
