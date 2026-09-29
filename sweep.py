"""Grid-search BRAIN settings for one expression, like clicking through
decay / neutralization / truncation in the BRAIN simulator.

    python sweep.py "rank(-ts_delta(close, 5))"
    python sweep.py "rank(-returns)" --data yfinance --decays 0 2 4 8
"""
import argparse
import itertools

import pandas as pd

from alpha import Settings, evaluate, simulate
from alpha.data import load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("expr")
    p.add_argument("--data", default="synthetic")
    p.add_argument("--decays", type=int, nargs="+", default=[0, 2, 4, 8, 16])
    p.add_argument("--neutralizations", nargs="+", default=["market", "sector"])
    p.add_argument("--truncations", type=float, nargs="+", default=[0.05, 0.08])
    a = p.parse_args()

    data = load(a.data)
    alpha = evaluate(a.expr, data)
    rows = []
    for d, n, t in itertools.product(a.decays, a.neutralizations, a.truncations):
        r = simulate(alpha, data, Settings(decay=d, neutralization=n, truncation=t))
        rows.append(dict(decay=d, neutralization=n, truncation=t, **{
            k: r.metrics[k] for k in ("sharpe", "fitness", "returns", "turnover", "drawdown", "margin_bps")},
            passes=f"{sum(r.checks.values())}/{len(r.checks)}"))
    table = pd.DataFrame(rows).sort_values("fitness", ascending=False)
    pd.set_option("display.width", 140)
    print(a.expr, "\n")
    print(table.round(3).to_string(index=False))
    print("\nTip: pick the most stable row, not just the top one; BRAIN's out-of-sample punishes overfit settings.")


if __name__ == "__main__":
    main()
