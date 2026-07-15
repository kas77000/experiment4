"""Human-readable rendering of the threshold table and score distributions."""

from __future__ import annotations
import pandas as pd

from tca import schema, thresholds


def format_threshold_table(table: pd.DataFrame) -> str:
    """Pretty one-line-per-group view, bands in units of spread."""
    show = table.copy()
    for c in ["q_red_lo", "q_grey_lo", "q_median", "q_grey_hi", "q_red_hi", "mad_spreads"]:
        show[c] = show[c].round(3)
    cols = ["level", schema.ALGO, schema.MARKET, schema.ADV_BUCKET, "n", "trusted",
            "q_red_lo", "q_grey_lo", "q_median", "q_grey_hi", "q_red_hi"]
    return show[cols].to_string(index=False)


def zone_summary(scored: pd.DataFrame) -> str:
    """Counts and % per zone across a scored frame."""
    order = [thresholds.RED_LOW, thresholds.GREY_LOW, thresholds.GREEN,
             thresholds.GREY_HIGH, thresholds.RED_HIGH, "NO_BAND"]
    counts = scored["zone"].value_counts()
    total = len(scored)
    lines = ["  zone         count     pct"]
    for z in order:
        c = int(counts.get(z, 0))
        lines.append(f"  {z:<11} {c:>6}  {100*c/total:5.1f}%")
    flagged = int(scored["flagged"].sum())
    lines.append(f"  {'-'*26}")
    lines.append(f"  FLAGGED      {flagged:>6}  {100*flagged/total:5.1f}%")
    return "\n".join(lines)
