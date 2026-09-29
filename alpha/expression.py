"""Evaluate BRAIN "Fast Expression" strings against local data.

    evaluate("rank(-ts_delta(close, 5))", data)

Supports the BRAIN ternary `cond ? a : b`, `&&`, `||` and `;`-separated
multi-line expressions with assignments (the last line is the alpha).
"""
from __future__ import annotations

import re

import pandas as pd

from .operators import OPERATORS

def _ternary(s: str) -> str:
    """Rewrite the innermost-left `c ? a : b` as if_else(c, a, b), repeatedly."""
    while "?" in s:
        q = s.index("?")
        depth, i = 0, q - 1  # walk left to the start of the condition
        while i >= 0:
            ch = s[i]
            if ch == ")": depth += 1
            elif ch == "(":
                if depth == 0: break
                depth -= 1
            elif ch == "," and depth == 0: break
            i -= 1
        start = i + 1
        depth, j, colon = 0, q + 1, None  # walk right to ':' and the end of b
        while j < len(s):
            ch = s[j]
            if ch == "(": depth += 1
            elif ch == ")":
                if depth == 0: break
                depth -= 1
            elif ch == ":" and depth == 0 and colon is None: colon = j
            elif ch == "," and depth == 0 and colon is not None: break
            j += 1
        if colon is None:
            raise SyntaxError(f"'?' without ':' in {s!r}")
        c, a, b = s[start:q], s[q + 1:colon], s[colon + 1:j]
        s = f"{s[:start]}if_else({c.strip()}, {a.strip()}, {b.strip()}){s[j:]}"
    return s


def _translate(line: str) -> str:
    line = line.replace("&&", "&").replace("||", "|").replace("!=", "__NE__")
    line = line.replace("!", "~").replace("__NE__", "!=")
    return _ternary(line)


def evaluate(expr: str, data: dict) -> pd.DataFrame:
    env = {**OPERATORS, **data, "__builtins__": {}}
    lines = [l.strip() for l in re.split(r"[;\n]", expr) if l.strip() and not l.strip().startswith("#")]
    result = None
    for line in lines:
        if re.match(r"^[A-Za-z_]\w*\s*=(?!=)", line):
            name, rhs = line.split("=", 1)
            env[name.strip()] = eval(_translate(rhs), env)
        else:
            result = eval(_translate(line), env)
    if result is None:
        raise ValueError("expression has no final output line")
    if not isinstance(result, pd.DataFrame):
        raise TypeError("expression must evaluate to a dates x tickers frame")
    return result
