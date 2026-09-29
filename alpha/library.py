"""Starter alpha library, written as BRAIN Fast Expressions.

Each entry: name -> (expression, suggested settings, idea). The expressions
use only price/volume fields so they run on any data source here and on
BRAIN's USA TOP3000 price-volume datasets.
"""

ALPHAS = {
    "reversal_5d": (
        "rank(-ts_delta(close, 5))",
        dict(decay=4, neutralization="sector"),
        "Short-term reversal: last week's losers bounce back.",
    ),
    "reversal_vol_scaled": (
        "rank(-returns / ts_std_dev(returns, 20))",
        dict(decay=3, neutralization="sector"),
        "Reversal on risk-adjusted 1-day move, so volatile names don't dominate.",
    ),
    "momentum_12_1": (
        "rank(ts_delay(close, 21) / ts_delay(close, 252) - 1)",
        dict(decay=10, neutralization="sector"),
        "Classic 12-1 month momentum, skipping the reversal month.",
    ),
    "price_volume_corr": (
        "-ts_corr(rank(close), rank(volume), 10)",
        dict(decay=5, neutralization="sector"),
        "Price up on rising volume tends to mean-revert (WorldQuant 101 style).",
    ),
    "volume_shock_reversal": (
        "vol_ratio = volume / ts_mean(volume, 20);"
        "rank(-returns) * rank(vol_ratio)",
        dict(decay=4, neutralization="sector"),
        "Reversal is stronger when the move came on unusual volume.",
    ),
    "low_volatility": (
        "rank(-ts_std_dev(returns, 60))",
        dict(decay=10, neutralization="sector"),
        "Low-volatility anomaly: calmer stocks earn better risk-adjusted returns.",
    ),
    "intraday_reversal": (
        "rank((open - close) / (high - low + 0.001))",
        dict(decay=3, neutralization="sector"),
        "Stocks that closed near the low of their range tend to recover.",
    ),
    "vwap_deviation": (
        "rank(vwap - close) * (ts_rank(volume, 20) > 0.5 ? 1 : 0.5)",
        dict(decay=3, neutralization="sector"),
        "Close below VWAP on heavy volume signals a temporary dislocation.",
    ),
    "trend_quality": (
        "rank(ts_mean(returns, 60) / ts_std_dev(returns, 60))",
        dict(decay=10, neutralization="sector"),
        "Rolling 3-month Sharpe of each stock: smooth trends persist.",
    ),
}
