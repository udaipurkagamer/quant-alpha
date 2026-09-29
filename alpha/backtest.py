"""BRAIN-style simulation and metrics.

Pipeline per day (same order as BRAIN settings):
    raw alpha -> pasteurize -> decay -> neutralize -> scale to book
    -> truncate -> positions -> PnL with delay

Metric definitions follow BRAIN's docs:
    Sharpe   = mean(daily pnl) / std(daily pnl) * sqrt(252)
    Returns  = annualised pnl / (booksize / 2)
    Turnover = mean daily dollars traded / booksize
    Fitness  = Sharpe * sqrt(|Returns| / max(Turnover, 0.125))
    Drawdown = max peak-to-trough of cum pnl / (booksize / 2)
    Margin   = total pnl / total dollars traded   (reported in bps)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import operators as op

YEAR = 252


@dataclass
class Settings:
    delay: int = 1
    decay: int = 0
    neutralization: str = "market"   # none | market | sector | industry
    truncation: float = 0.08
    booksize: float = 20e6
    pasteurize: bool = True
    start: str | None = None
    end: str | None = None


@dataclass
class Result:
    pnl: pd.Series
    positions: pd.DataFrame
    metrics: dict
    yearly: pd.DataFrame
    checks: dict = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        return (f"Sharpe {m['sharpe']:.2f} | Fitness {m['fitness']:.2f} | "
                f"Returns {m['returns']:.1%} | Turnover {m['turnover']:.1%} | "
                f"Drawdown {m['drawdown']:.1%} | Margin {m['margin_bps']:.1f}bps")


def _truncate(w: pd.DataFrame, cap: float) -> pd.DataFrame:
    for _ in range(10):  # clip then renormalise until stable
        w = w.clip(-cap, cap)
        w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0)
        if (w.abs() <= cap + 1e-9).all().all():
            break
    return w


def to_weights(alpha: pd.DataFrame, data: dict, s: Settings) -> pd.DataFrame:
    a = alpha.replace([np.inf, -np.inf], np.nan)
    if s.pasteurize:
        a = a.where(data["close"].notna())
    if s.decay > 1:
        a = op.ts_decay_linear(a, s.decay).where(a.notna() | a.shift(1).notna())
    if s.neutralization == "market":
        a = a.sub(a.mean(axis=1), axis=0)
    elif s.neutralization in ("sector", "industry"):
        a = op.group_neutralize(a, data[s.neutralization])
    w = a.div(a.abs().sum(axis=1).replace(0, np.nan), axis=0)
    if s.truncation and s.truncation < 1:
        w = _truncate(w, s.truncation)
    return w.fillna(0.0)


def simulate(alpha: pd.DataFrame, data: dict, s: Settings | None = None) -> Result:
    s = s or Settings()
    w = to_weights(alpha, data, s)
    # Row t of w uses data through close t. Delay 0 trades at close t and
    # earns day t+1's return (shift 1); delay 1 uses only data through t-1
    # for the book traded at close t, so it earns t+1 with a shift of 2.
    pos = w.shift(s.delay + 1).fillna(0.0) * s.booksize
    ret = data["returns"].reindex_like(pos).fillna(0.0)
    pnl = (pos * ret).sum(axis=1)
    traded = pos.diff().abs().sum(axis=1)

    live = pos.abs().sum(axis=1) > 0
    first = live.idxmax() if live.any() else pos.index[0]
    sl = slice(s.start or first, s.end)
    pnl, traded, pos = pnl.loc[sl], traded.loc[sl], pos.loc[sl]

    metrics = _metrics(pnl, traded, pos, s.booksize)
    yearly = pd.DataFrame({
        y: _metrics(pnl[pnl.index.year == y], traded[traded.index.year == y],
                    pos[pos.index.year == y], s.booksize)
        for y in sorted(set(pnl.index.year))
    }).T[["sharpe", "returns", "turnover", "fitness", "drawdown", "margin_bps"]]
    res = Result(pnl, pos, metrics, yearly)
    res.checks = brain_checks(res)
    return res


def _metrics(pnl, traded, pos, book) -> dict:
    half = book / 2
    sd = pnl.std()
    sharpe = pnl.mean() / sd * np.sqrt(YEAR) if sd > 0 else 0.0
    returns = pnl.mean() * YEAR / half
    turnover = (traded / book).mean()
    fitness = sharpe * np.sqrt(abs(returns) / max(turnover, 0.125))
    cum = pnl.cumsum()
    drawdown = (cum.cummax() - cum).max() / half
    margin = pnl.sum() / traded.sum() * 1e4 if traded.sum() > 0 else 0.0
    return dict(
        sharpe=sharpe, returns=returns, turnover=turnover, fitness=fitness,
        drawdown=drawdown, margin_bps=margin,
        long_count=(pos > 0).sum(axis=1).mean(), short_count=(pos < 0).sum(axis=1).mean(),
        max_weight=(pos.abs().div(pos.abs().sum(axis=1).replace(0, np.nan), axis=0)).max().max(),
    )


def brain_checks(r: Result) -> dict:
    """BRAIN's main in-sample submission tests (delay-1 thresholds)."""
    m = r.metrics
    return {
        "sharpe >= 1.25": m["sharpe"] >= 1.25,
        "fitness >= 1.0": m["fitness"] >= 1.0,
        "turnover 1%-70%": 0.01 <= m["turnover"] <= 0.70,
        "max weight <= 10%": (m["max_weight"] or 0) <= 0.10,
        "positive most years": (r.yearly["sharpe"] > 0).mean() >= 0.7,
    }


def ic(alpha: pd.DataFrame, data: dict, horizon: int = 1, delay: int = 1) -> pd.Series:
    """Daily rank IC between alpha and forward `horizon`-day return."""
    fwd = data["close"].shift(-(horizon + delay)) / data["close"].shift(-delay) - 1
    return alpha.rank(axis=1).corrwith(fwd.rank(axis=1), axis=1)


def self_correlation(pnls: dict[str, pd.Series]) -> pd.DataFrame:
    """PnL correlation matrix. BRAIN rejects submissions with > 0.7 vs your
    existing alphas, so use this to keep the library diverse."""
    return pd.DataFrame(pnls).corr()
