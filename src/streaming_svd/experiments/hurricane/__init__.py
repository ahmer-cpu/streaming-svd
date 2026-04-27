"""Hurricane Isabel dataset experiment pipeline.

Three-stage pipeline for comprehensive cold-start vs warm-start rSVD analysis
across all 13 atmospheric variables and 48 hourly timesteps.

Stages
------
1. **collect** — run SVD, save per-variable raw CSVs::

       python -m streaming_svd.experiments.hurricane.collect --help

2. **analyze** — aggregate raw CSVs into summary statistics::

       python -m streaming_svd.experiments.hurricane.analyze --help

3. **plot** — generate all figures from CSVs (no re-computation)::

       python -m streaming_svd.experiments.hurricane.plot --help

Quick start (smoke test — one variable, 5 timesteps)::

    python -m streaming_svd.experiments.hurricane.collect \\
        --data-dir data/raw --vars Uf --start 1 --end 5

    python -m streaming_svd.experiments.hurricane.analyze \\
        --raw-dir results/hurricane/raw \\
        --out results/hurricane/hurricane_summary.csv --print-table

    python -m streaming_svd.experiments.hurricane.plot \\
        --raw-dir results/hurricane/raw \\
        --summary results/hurricane/hurricane_summary.csv \\
        --fig-dir results/hurricane/figures --vars Uf
"""

def __getattr__(name):
    # Lazy imports to avoid circular-import warnings when sub-modules are run
    # directly as __main__ (e.g. python -m ...hurricane.collect).
    if name == "collect_hurricane_experiment":
        from streaming_svd.experiments.hurricane.collect import collect_hurricane_experiment
        return collect_hurricane_experiment
    if name == "analyze_hurricane_results":
        from streaming_svd.experiments.hurricane.analyze import analyze_hurricane_results
        return analyze_hurricane_results
    if name in ("plot_per_variable", "plot_cross_variable"):
        from streaming_svd.experiments.hurricane import plot as _plot_mod
        return getattr(_plot_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "collect_hurricane_experiment",
    "analyze_hurricane_results",
    "plot_per_variable",
    "plot_cross_variable",
]
