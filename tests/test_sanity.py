"""Sanity tests that catch the classic backtest bugs (look-ahead, sign flips)."""
import numpy as np
import pandas as pd

from alpha import Settings, evaluate, simulate
from alpha.data import load_synthetic

DATA = load_synthetic(n_stocks=80, n_days=600)


def test_random_alpha_has_no_edge():
    rng = np.random.default_rng(0)
    noise = pd.DataFrame(rng.normal(size=DATA["close"].shape), DATA["close"].index, DATA["close"].columns)
    assert abs(simulate(noise, DATA).metrics["sharpe"]) < 1.0


def test_planted_reversal_is_found():
    r = simulate(evaluate("rank(-ts_delta(close, 5))", DATA), DATA, Settings(delay=0))
    assert r.metrics["sharpe"] > 1.0


def test_future_data_is_not_used_with_delay1():
    # Tomorrow's return as a signal must look like a miracle with delay 0
    # when shifted back two days, and must be useless when not shifted.
    fut = DATA["returns"].shift(-1)
    assert simulate(fut, DATA, Settings(delay=0)).metrics["sharpe"] > 5
    assert abs(simulate(fut, DATA, Settings(delay=1)).metrics["sharpe"]) < 3


def test_weights_respect_truncation():
    r = simulate(evaluate("rank(close)", DATA), DATA, Settings(truncation=0.05))
    assert r.metrics["max_weight"] <= 0.05 + 1e-6


def test_ternary_and_multiline():
    a = evaluate("x = ts_mean(close, 5); x > close ? 1 : -1", DATA)
    assert set(np.unique(a.dropna().values)) <= {1.0, -1.0}
