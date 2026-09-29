"""WorldQuant BRAIN-style operators.

Every operator works on a wide DataFrame: index = dates, columns = tickers.
Cross-sectional ops act across a row (one day), time-series ops act down a
column (one stock). Names and semantics follow the BRAIN operator docs so an
expression written here can be pasted into BRAIN with little or no change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Frame = pd.DataFrame


def _d(d) -> int:
    d = int(d)
    if d < 1:
        raise ValueError("lookback must be >= 1")
    return d


# ---------------------------------------------------------------- arithmetic
def abs_(x): return x.abs()
def log(x): return np.log(x.where(x > 0))
def sign(x): return np.sign(x)
def sqrt(x): return np.sqrt(x.where(x >= 0))
def power(x, e): return x ** e
def signed_power(x, e): return np.sign(x) * x.abs() ** e
def max_(x, y): return np.maximum(x, y)
def min_(x, y): return np.minimum(x, y)
def inverse(x): return 1.0 / x.replace(0, np.nan)
def reverse(x): return -x


def if_else(cond, a, b):
    """BRAIN's `cond ? a : b` / if_else(cond, a, b)."""
    a = a if isinstance(a, pd.DataFrame) else pd.DataFrame(a, index=cond.index, columns=cond.columns)
    return a.where(cond.astype(bool), b)


# ---------------------------------------------------------- cross-sectional
def rank(x, rate=None):
    """Cross-sectional rank scaled to [0, 1]."""
    r = x.rank(axis=1, method="average")
    n = x.notna().sum(axis=1)
    return r.sub(1).div((n - 1).replace(0, np.nan), axis=0)


def zscore(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1).replace(0, np.nan), axis=0)


def scale(x, scale=1.0):
    """Rescale so the sum of absolute values per day equals `scale`."""
    return x.div(x.abs().sum(axis=1).replace(0, np.nan), axis=0) * scale


def normalize(x, useStd=False):
    out = x.sub(x.mean(axis=1), axis=0)
    return out.div(x.std(axis=1).replace(0, np.nan), axis=0) if useStd else out


def winsorize(x, std=4):
    mu, sd = x.mean(axis=1), x.std(axis=1)
    return x.clip(lower=mu - std * sd, upper=mu + std * sd, axis=0)


def truncate(x, maxPercent=0.01):
    """Cap each weight at maxPercent of the book (applied after scaling)."""
    w = scale(x)
    return w.clip(-maxPercent, maxPercent)


# ------------------------------------------------------------- time series
def ts_delay(x, d): return x.shift(_d(d))
def ts_delta(x, d): return x - x.shift(_d(d))
def ts_mean(x, d): return x.rolling(_d(d), min_periods=max(1, _d(d) // 2)).mean()
def ts_sum(x, d): return x.rolling(_d(d), min_periods=max(1, _d(d) // 2)).sum()
def ts_std_dev(x, d): return x.rolling(_d(d), min_periods=max(2, _d(d) // 2)).std()
def ts_min(x, d): return x.rolling(_d(d), min_periods=1).min()
def ts_max(x, d): return x.rolling(_d(d), min_periods=1).max()
def ts_product(x, d): return x.rolling(_d(d), min_periods=1).apply(np.prod, raw=True)


def ts_zscore(x, d):
    return (x - ts_mean(x, d)) / ts_std_dev(x, d).replace(0, np.nan)


def ts_rank(x, d, constant=0):
    """Rank of today's value within the last d days, scaled to [0, 1]."""
    d = _d(d)
    return x.rolling(d, min_periods=max(2, d // 2)).rank(pct=True) + constant


def ts_arg_max(x, d):
    """Days since the max within the window (0 = today)."""
    d = _d(d)
    return x.rolling(d, min_periods=1).apply(lambda a: len(a) - 1 - np.nanargmax(a), raw=True)


def ts_arg_min(x, d):
    d = _d(d)
    return x.rolling(d, min_periods=1).apply(lambda a: len(a) - 1 - np.nanargmin(a), raw=True)


def ts_corr(x, y, d):
    d = _d(d)
    return x.rolling(d, min_periods=max(3, d // 2)).corr(y)


def ts_covariance(x, y, d):
    d = _d(d)
    return x.rolling(d, min_periods=max(3, d // 2)).cov(y)


def ts_decay_linear(x, d, dense=False):
    """Linearly weighted moving average, newest day weighted d, oldest 1."""
    d = _d(d)
    w = np.arange(1, d + 1, dtype=float)
    vals = x.to_numpy(dtype=float)
    if not dense:
        vals = np.nan_to_num(vals)
    out = np.full_like(vals, np.nan)
    for t in range(d - 1, len(vals)):
        out[t] = w @ vals[t - d + 1 : t + 1] / w.sum()
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def ts_regression(y, x, d, lag=0, rettype=0):
    """Rolling OLS y ~ x. rettype 0 = residual, 1 = alpha, 2 = beta."""
    x = x.shift(lag) if lag else x
    beta = ts_covariance(y, x, d) / ts_std_dev(x, d).pow(2).replace(0, np.nan)
    alpha = ts_mean(y, d) - beta * ts_mean(x, d)
    if rettype == 2:
        return beta
    if rettype == 1:
        return alpha
    return y - (alpha + beta * x)


def ts_backfill(x, d):
    return x.ffill(limit=_d(d))


def days_from_last_change(x):
    changed = x.ne(x.shift())
    grp = changed.cumsum()
    return grp.apply(lambda c: c.groupby(c).cumcount())


def hump(x, hump=0.01):
    """Only move a position when the change exceeds `hump` * total book."""
    vals = x.to_numpy(dtype=float)
    out = np.empty_like(vals)
    out[0] = vals[0]
    for t in range(1, len(vals)):
        prev = out[t - 1]
        thr = hump * np.nansum(np.abs(vals[t]))
        move = np.abs(vals[t] - prev) > thr
        out[t] = np.where(move | np.isnan(prev), vals[t], prev)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


# ------------------------------------------------------------------ group
def _group_apply(x, group, fn):
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    g = group.reindex(index=x.index, columns=x.columns) if isinstance(group, pd.DataFrame) else group
    labels = pd.unique(pd.DataFrame(g).stack().dropna()) if isinstance(g, pd.DataFrame) else pd.unique(g.dropna())
    for lab in labels:
        mask = (g == lab)
        if isinstance(mask, pd.Series):
            mask = pd.DataFrame(np.broadcast_to(mask.values, x.shape), index=x.index, columns=x.columns)
        sub = x.where(mask)
        out = out.where(~mask, fn(sub))
    return out


def group_neutralize(x, group):
    return _group_apply(x, group, lambda s: s.sub(s.mean(axis=1), axis=0))


def group_rank(x, group):
    return _group_apply(x, group, rank)


def group_zscore(x, group):
    return _group_apply(x, group, zscore)


def group_mean(x, group):
    return _group_apply(x, group, lambda s: pd.DataFrame(
        np.broadcast_to(s.mean(axis=1).values[:, None], s.shape), index=s.index, columns=s.columns))


# BRAIN names that clash with Python builtins
OPERATORS = {
    name: obj for name, obj in globals().items()
    if callable(obj) and not name.startswith("_") and name not in {"Frame"}
}
OPERATORS.update({"abs": abs_, "max": max_, "min": min_})
for k in ("abs_", "max_", "min_"):
    OPERATORS.pop(k, None)
