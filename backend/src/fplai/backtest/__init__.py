"""Walk-forward policy backtester (stage 4): season replay per MODEL_DESIGN_INPUTS §7.

Public surface: :func:`run_backtest` (real-data season replay), :func:`run` (CLI-ready
wrapper for ``pipeline.py``), :func:`replay` (the frame-injection core the tests drive),
plus the parameter/result models and the §7.3 reference constants. See
``fplai.backtest.harness`` for the full protocol/leakage documentation.
"""

from fplai.backtest.harness import (
    AVERAGE_MANAGER_POINTS,
    HINDSIGHT_OPTIMUM_POINTS_2019,
    TOP_10K_POINTS,
    BacktestParams,
    BacktestResult,
    GwLedger,
    PolicyLedger,
    ScoredGw,
    apply_autosubs,
    compute_xp_metrics,
    last5_xp_frame,
    replay,
    run,
    run_backtest,
    score_gw,
)

__all__ = [
    "AVERAGE_MANAGER_POINTS",
    "HINDSIGHT_OPTIMUM_POINTS_2019",
    "TOP_10K_POINTS",
    "BacktestParams",
    "BacktestResult",
    "GwLedger",
    "PolicyLedger",
    "ScoredGw",
    "apply_autosubs",
    "compute_xp_metrics",
    "last5_xp_frame",
    "replay",
    "run",
    "run_backtest",
    "score_gw",
]
