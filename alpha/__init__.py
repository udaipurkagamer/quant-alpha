"""quant-alpha: BRAIN-style alpha research toolkit."""
from .expression import evaluate
from .backtest import Settings, simulate

__all__ = ["evaluate", "Settings", "simulate"]
