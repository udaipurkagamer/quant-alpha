"""Market data loaders. Everything returns a dict of wide DataFrames
(dates x tickers) with BRAIN field names: open, high, low, close, volume,
vwap, returns, adv20, cap, plus `sector` and `industry` group labels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# A liquid, sector-diverse US universe (~100 names) for the yfinance loader.
UNIVERSE = {
    "Tech": "AAPL MSFT NVDA GOOGL META AVGO ORCL CRM ADBE AMD INTC CSCO QCOM TXN IBM AMAT MU NOW INTU ADI",
    "Financials": "JPM BAC WFC GS MS C BLK SCHW AXP SPGI CB PGR MMC USB PNC",
    "Health": "UNH JNJ LLY PFE MRK ABBV TMO ABT DHR BMY AMGN GILD CVS MDT ISRG",
    "Consumer": "AMZN TSLA HD MCD NKE SBUX LOW TGT BKNG TJX PG KO PEP COST WMT PM MO CL",
    "Industrials": "CAT DE HON UNP UPS GE BA LMT RTX MMM ETN EMR FDX CSX NSC",
    "Energy": "XOM CVX COP EOG SLB PSX MPC OXY VLO KMI",
    "Utilities": "NEE DUK SO D AEP EXC SRE XEL",
    "Comms": "NFLX DIS CMCSA T VZ TMUS CHTR",
}


def _finish(close, open_, high, low, volume, sector, industry=None):
    d = {"close": close, "open": open_, "high": high, "low": low, "volume": volume}
    d["vwap"] = (high + low + close) / 3  # typical-price proxy; BRAIN has true vwap
    d["returns"] = close.pct_change(fill_method=None)
    d["adv20"] = (close * volume).rolling(20, min_periods=5).mean()
    d["cap"] = close * volume.rolling(60, min_periods=5).mean() * 50  # rough size proxy
    d["sector"] = sector
    d["industry"] = industry if industry is not None else sector
    return d


def load_yfinance(start="2015-01-01", end=None, cache="data/prices.pkl"):
    """Download daily OHLCV for UNIVERSE via yfinance (cached on disk)."""
    cache = Path(cache)
    if cache.exists():
        raw = pd.read_pickle(cache)
    else:
        import yfinance as yf
        tickers = " ".join(UNIVERSE.values()).split()
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError("yfinance returned no data (network blocked?). Use --data synthetic or csv.")
        cache.parent.mkdir(parents=True, exist_ok=True)
        raw.to_pickle(cache)
    sector = pd.Series({t: s for s, ts in UNIVERSE.items() for t in ts.split()})
    f = lambda k: raw[k].dropna(axis=1, how="all")
    close = f("Close")
    sector = sector.reindex(close.columns)
    cols = close.columns
    return _finish(close, f("Open")[cols], f("High")[cols], f("Low")[cols], f("Volume")[cols], sector)


def load_csv(path):
    """Long-format CSV with columns: date,ticker,open,high,low,close,volume[,sector]."""
    df = pd.read_csv(path, parse_dates=["date"])
    wide = lambda k: df.pivot(index="date", columns="ticker", values=k).sort_index()
    close = wide("close")
    sector = (df.groupby("ticker")["sector"].first().reindex(close.columns)
              if "sector" in df else pd.Series("ALL", index=close.columns))
    return _finish(close, wide("open"), wide("high"), wide("low"), wide("volume"), sector)


def load_synthetic(n_stocks=200, n_days=1500, n_sectors=8, seed=7):
    """Realistic-ish fake market for offline testing.

    Returns follow a market + sector factor model with fat tails, plus one
    small *planted* anomaly (5-day short-term reversal) so a correct
    backtester has something to find. Everything else should score ~0,
    which makes this a good check for look-ahead bugs.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    tickers = [f"S{i:03d}" for i in range(n_stocks)]
    sec_id = rng.integers(0, n_sectors, n_stocks)
    beta = rng.normal(1.0, 0.3, n_stocks)
    vol = rng.uniform(0.012, 0.03, n_stocks)

    mkt = rng.standard_t(4, n_days) * 0.008
    sec = rng.standard_t(4, (n_days, n_sectors)) * 0.006
    idio = rng.standard_t(4, (n_days, n_stocks)) * vol / np.sqrt(2)

    ret = np.zeros((n_days, n_stocks))
    for t in range(n_days):
        r = beta * mkt[t] + sec[t, sec_id] + idio[t]
        if t >= 6:
            past5 = ret[t - 5 : t].sum(0)
            r -= 0.012 * (past5 - past5.mean())  # planted reversal
        ret[t] = r
    close = 50 * np.exp(np.cumsum(np.log1p(ret), axis=0))
    base_vol = rng.lognormal(14, 1, n_stocks)
    volume = base_vol * np.exp(rng.normal(0, 0.3, (n_days, n_stocks)) + 8 * np.abs(idio))
    rng_hl = np.abs(rng.normal(0, 1, (n_days, n_stocks))) * vol
    open_ = close / (1 + ret) * (1 + rng.normal(0, 0.3, (n_days, n_stocks)) * vol)

    F = lambda a: pd.DataFrame(a, index=dates, columns=tickers)
    close_f = F(close)
    high = F(np.maximum(close, open_) * (1 + rng_hl))
    low = F(np.minimum(close, open_) * (1 - rng_hl))
    sector = pd.Series([f"SEC{s}" for s in sec_id], index=tickers)
    return _finish(close_f, F(open_), high, low, F(volume), sector)


def load(name="synthetic", **kw):
    if name == "synthetic":
        return load_synthetic(**kw)
    if name == "yfinance":
        return load_yfinance(**kw)
    return load_csv(name)
