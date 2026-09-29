"""Run the alpha library (or one expression) and print a BRAIN-style report.

    python run.py                                  # all library alphas, synthetic data
    python run.py --data yfinance                  # real US large caps (needs internet)
    python run.py --expr "rank(-ts_delta(close,5))" --decay 4 --neutralization sector
"""
import argparse
from pathlib import Path

import pandas as pd

from alpha import Settings, evaluate, simulate
from alpha.backtest import ic, self_correlation
from alpha.data import load
from alpha.library import ALPHAS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="synthetic", help="synthetic | yfinance | path/to.csv")
    p.add_argument("--expr", help="a single Fast Expression to test")
    p.add_argument("--decay", type=int)
    p.add_argument("--neutralization")
    p.add_argument("--truncation", type=float)
    p.add_argument("--delay", type=int, default=1)
    p.add_argument("--out", default="reports")
    a = p.parse_args()

    data = load(a.data)
    todo = {"custom": (a.expr, {}, "")} if a.expr else ALPHAS
    rows, pnls = [], {}
    for name, (expr, kw, _) in todo.items():
        kw = {**kw, "delay": a.delay}
        for k in ("decay", "neutralization", "truncation"):
            if getattr(a, k) is not None:
                kw[k] = getattr(a, k)
        alpha = evaluate(expr, data)
        r = simulate(alpha, data, Settings(**kw))
        pnls[name] = r.pnl
        m = r.metrics
        rows.append(dict(alpha=name, sharpe=m["sharpe"], fitness=m["fitness"], returns=m["returns"],
                         turnover=m["turnover"], drawdown=m["drawdown"], margin_bps=m["margin_bps"],
                         ic=ic(alpha, data).mean(), passes=f"{sum(r.checks.values())}/{len(r.checks)}"))
        if a.expr:
            print(expr, "\n", r.summary(), "\n\nYearly:\n", r.yearly.round(3), "\n\nChecks:")
            for k, v in r.checks.items():
                print(f"  {'PASS' if v else 'FAIL'}  {k}")

    table = pd.DataFrame(rows).set_index("alpha").sort_values("fitness", ascending=False)
    pd.set_option("display.width", 140)
    print("\n", table.round(3).to_string())
    if len(pnls) > 1:
        print("\nPnL self-correlation (BRAIN wants < 0.7 vs your other alphas):")
        print(self_correlation(pnls).round(2).to_string())

    out = Path(a.out); out.mkdir(exist_ok=True)
    table.to_csv(out / "summary.csv")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = (pd.DataFrame(pnls).cumsum() / 1e6).plot(figsize=(10, 5), title=f"Cumulative PnL ($M, $20M book) - {a.data}")
        ax.set_ylabel("$M"); plt.tight_layout(); plt.savefig(out / "pnl.png", dpi=120)
        print(f"\nSaved {out/'summary.csv'} and {out/'pnl.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
