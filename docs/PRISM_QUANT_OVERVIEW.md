# Prism Quant — Product Overview

**Prism Quant is a local research lab for WorldQuant BRAIN–style equity alphas.**
You write a trading idea as a BRAIN *Fast Expression*, backtest it on daily stock
data, and get the same numbers BRAIN reports — Sharpe, Fitness, Turnover, Returns,
Drawdown and Margin — together with BRAIN's submission checks and PnL
self-correlation. The same expression string can then be pasted straight into
BRAIN once it looks good.

---

## The problem it solves

BRAIN simulations are slow, limited in number, and opaque — you cannot see inside
them. Prism Quant reimplements the core of the BRAIN simulator locally, so that:

- ideas are screened in **seconds**, and across many settings at once;
- **every step** of the pipeline (decay, neutralization, truncation, delay) is
  visible, inspectable and unit-tested;
- you only spend real BRAIN simulations on ideas that have already survived
  local testing.

---

## How it works

An idea travels through four stages:

**1. Write the alpha as a Fast Expression**

```python
from alpha import evaluate, simulate, Settings
from alpha.data import load

data  = load("yfinance")                              # ~110 US large caps
alpha = evaluate("rank(-ts_delta(close, 5))", data)   # same syntax as BRAIN
res   = simulate(alpha, data, Settings(decay=4, neutralization="sector"))
print(res.summary())
```

**2. The simulator runs BRAIN's pipeline, in BRAIN's order**

```
raw alpha → pasteurize → decay → neutralize (market/sector/industry)
          → scale to book → truncate → positions → PnL with delay
```

**3. Metrics and submission checks come out**

```
Sharpe 3.89 | Fitness 3.13 | Returns 61.2% | Turnover 39% | Drawdown 8.4% | Margin 24.1bps
```

**4. Sweep the settings grid** to find the configuration where the idea actually
passes BRAIN's constraints, instead of guessing.

---

## What is inside

| Component | What it gives you |
|---|---|
| **30+ BRAIN operators** | `rank`, `zscore`, `ts_delta`, `ts_corr`, `ts_rank`, `ts_decay_linear`, `ts_regression`, `group_neutralize`, `group_rank`, `hump`, `winsorize`, and more (42 functions in total) |
| **Fast Expression parser** | Runs BRAIN expression strings directly, including ternaries (`cond ? a : b`), `&&` / `\|\|`, and multi-line definitions (`x = ...; ...`) |
| **BRAIN-style simulator** | Full configurable pipeline: delay, decay, neutralization, truncation, book size, pasteurize, date range |
| **BRAIN metrics** | Sharpe, Fitness, Turnover, Returns, Drawdown, Margin, plus a year-by-year breakdown and long/short counts |
| **Submission checks** | The five delay-1 in-sample tests: Sharpe ≥ 1.25, Fitness ≥ 1.0, Turnover within 1%–70%, max weight ≤ 10%, positive in most years |
| **Correlation control** | Rank IC against forward returns, and a PnL self-correlation matrix — BRAIN rejects alphas correlated above 0.7 with your existing ones |
| **Settings sweep** | Grid search over decay / neutralization / truncation for a single alpha |
| **Starter alpha library** | Nine documented price–volume alphas: short-term reversal, volatility-scaled reversal, 12-1 momentum, price–volume correlation, volume-shock reversal, low volatility, intraday reversal, VWAP deviation, trend quality |
| **Three data sources** | `yfinance` (US large caps, grouped by sector, cached to disk), your own CSV, or a synthetic market for bias testing |
| **Look-ahead protection** | A pytest suite proves the simulator cannot peek at future data under delay 1 |

---

## Metric definitions (identical to BRAIN)

| Metric | Formula |
|---|---|
| Sharpe | mean(daily PnL) / std(daily PnL) × √252 |
| Returns | annual PnL / (book / 2) |
| Turnover | daily $ traded / book |
| Fitness | Sharpe × √(\|Returns\| / max(Turnover, 0.125)) |
| Drawdown | max peak-to-trough of cumulative PnL / (book / 2) |
| Margin | total PnL / total $ traded, in bps |

**Delay 1** is the default, matching BRAIN: the book traded at today's close uses
only data up to yesterday's close.

---

## Why you can trust the numbers

A backtester is only useful if it finds real effects and ignores fake ones.
The built-in synthetic market contains exactly **one** planted anomaly — a 5-day
reversal. A correct pipeline must find that and score everything else near zero:

| Alpha | Sharpe | Fitness | Turnover | BRAIN checks |
|---|---:|---:|---:|---:|
| reversal_5d | **3.89** | **3.13** | 39% | 5/5 |
| reversal_vol_scaled | 2.90 | 1.29 | 94% | 4/5 |
| volume_shock_reversal | 2.74 | 1.28 | 80% | 4/5 |
| momentum_12_1 | −0.32 | −0.13 | 3% | 2/5 |
| low_volatility | −0.16 | −0.05 | 3% | 2/5 |

The reversal family lights up; momentum and low volatility — which were never
planted — stay at roughly zero. The high-turnover variants correctly fail BRAIN's
70% turnover cap, which is exactly what the settings sweep is for.

---

## Getting started

```bash
pip install -r requirements.txt

python run.py                                    # full alpha library, offline synthetic data
python run.py --data yfinance                    # real US large caps (needs internet)
python run.py --expr "rank(-ts_delta(close, 5))" --decay 4 --neutralization sector
python sweep.py "rank(-returns)" --decays 0 4 8  # grid of BRAIN settings
pytest -q                                        # including the look-ahead tests
```

Every run writes a metrics table to `reports/summary.csv` and a cumulative PnL
chart to `reports/pnl.png`.

Requirements: Python with pandas, numpy, matplotlib and yfinance. Everything runs
locally — no account, server or API key is needed for synthetic or CSV data.

---

## Known limitations (stated upfront)

- Free Yahoo data covers only **today's** large caps, so it carries survivorship
  bias; results will look better here than on BRAIN's TOP3000 universe.
- `vwap` is approximated by the typical price (H+L+C)/3, and `cap` is a rough proxy.
- No transaction costs beyond what Margin reports — BRAIN also excludes them
  from Sharpe.

## Roadmap

- Fundamental and analyst datasets (BRAIN `fundamental6`, `analyst4` style)
- Out-of-sample split and a simple overfitting score
- Portfolio construction across alphas (equal-weight, then ML) under a
  correlation budget

---

*Prism Quant — MIT licensed.*
