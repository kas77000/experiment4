"""Tier-agnostic rendering helpers.

Anything specific to one tier's output lives in that tier's folder; this module
only knows about zones, which all three tiers agree on.
"""

from __future__ import annotations
import pandas as pd

ZONE_ORDER = ["OUT_LOW", "IN_RANGE", "OUT_HIGH", "NO_BAND"]


def zone_summary(scored: pd.DataFrame, flag_col: str = "flagged") -> str:
    """Counts and % per zone across a scored frame."""
    counts = scored["zone"].value_counts()
    total = max(len(scored), 1)
    lines = ["  zone         count     pct"]
    for z in ZONE_ORDER:
        c = int(counts.get(z, 0))
        if c or z != "NO_BAND":
            lines.append(f"  {z:<11} {c:>6}  {100*c/total:5.1f}%")
    flagged = int(scored[flag_col].sum())
    lines.append(f"  {'-'*26}")
    lines.append(f"  FLAGGED     {flagged:>6}  {100*flagged/total:5.1f}%")
    if "review_required" in scored.columns:
        rr = int(scored["review_required"].sum())
        lines.append(f"  TO REVIEW   {rr:>6}  {100*rr/total:5.1f}%   (after materiality gate)")
    return "\n".join(lines)


def header(title: str, width: int = 72) -> str:
    return f"\n{'=' * width}\n{title}\n{'=' * width}"


def frame(df: pd.DataFrame, max_rows: int = 30) -> str:
    """Print a frame without pandas truncating the middle out of it."""
    if df is None or not len(df):
        return "  (empty)"
    with pd.option_context("display.max_rows", max_rows,
                           "display.max_columns", 50,
                           "display.width", 200):
        return df.to_string()
