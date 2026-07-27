"""Score a tier's flags against known ground truth.

Only usable on the synthetic demo, where `synthetic_data` records which orders
were genuinely broken and why. On real data these functions no-op, because you
do not know the answer -- which is exactly why calibration (Tier 3's coverage
check) matters there instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tca import schema


def has_truth(df: pd.DataFrame) -> bool:
    return schema.TRUE_OUTLIER in df.columns


def detection_stats(scored: pd.DataFrame, flag_col: str = "flagged") -> dict:
    """Precision / recall / F1 of a tier's flags against injected true failures.

    precision -> of the orders you sent for review, how many were really broken
    recall    -> of the orders that were really broken, how many did you catch
    """
    if not has_truth(scored):
        return {}

    truth = scored[schema.TRUE_OUTLIER].astype(bool)
    pred = scored[flag_col].astype(bool)

    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "n": len(scored),
        "flagged": int(pred.sum()),
        "flag_rate_pct": 100.0 * pred.mean(),
        "true_failures": int(truth.sum()),
        "caught": tp,
        "false_alarms": fp,
        "missed": fn,
        "precision_pct": 100.0 * precision,
        "recall_pct": 100.0 * recall,
        "f1_pct": 100.0 * f1,
    }


def recall_by_cause(scored: pd.DataFrame, flag_col: str = "flagged") -> pd.DataFrame:
    """Which *kinds* of failure a tier catches, and which it sleeps through."""
    if not has_truth(scored) or schema.TRUE_CAUSE not in scored.columns:
        return pd.DataFrame()

    real = scored[scored[schema.TRUE_OUTLIER].astype(bool)]
    if not len(real):
        return pd.DataFrame()

    g = real.groupby(schema.TRUE_CAUSE)
    return pd.DataFrame({
        "n": g.size(),
        "caught": g[flag_col].sum().astype(int),
        "recall_pct": (100.0 * g[flag_col].mean()).round(1),
    }).sort_values("recall_pct")


def precision_at_budget(scored: pd.DataFrame, budget_pct: float,
                        rank_col: str = "rank_stat") -> dict:
    """Detection quality when every tier is held to the SAME review budget.

    Comparing tiers at their own natural flag rates is not a fair test -- a tier
    that flags 10% of the book will always out-recall one that flags 4%. Here
    each tier ranks the whole book by its own severity statistic, the top
    `budget_pct` are 'sent for review', and precision/recall are measured on
    that fixed-size queue. This is the comparison that says which METHOD ranks
    orders better, independent of where anyone set their threshold.
    """
    if not has_truth(scored) or rank_col not in scored.columns:
        return {}

    k = max(int(round(budget_pct / 100.0 * len(scored))), 1)
    stat = scored[rank_col].fillna(-np.inf)
    top = stat.nlargest(k).index

    truth = scored[schema.TRUE_OUTLIER].astype(bool)
    pred = pd.Series(False, index=scored.index)
    pred.loc[top] = True

    tp = int((truth & pred).sum())
    n_true = int(truth.sum())
    return {
        "budget_pct": budget_pct,
        "queue": k,
        "caught": tp,
        "precision_pct": 100.0 * tp / k,
        "recall_pct": 100.0 * tp / n_true if n_true else 0.0,
    }


def format_stats(stats: dict) -> str:
    if not stats:
        return "  (no ground truth available -- real data)"
    return (
        f"  flagged        {stats['flagged']:>6,}  ({stats['flag_rate_pct']:.1f}% of book)\n"
        f"  true failures  {stats['true_failures']:>6,}\n"
        f"  caught         {stats['caught']:>6,}\n"
        f"  false alarms   {stats['false_alarms']:>6,}\n"
        f"  missed         {stats['missed']:>6,}\n"
        f"  precision      {stats['precision_pct']:>6.1f}%\n"
        f"  recall         {stats['recall_pct']:>6.1f}%\n"
        f"  F1             {stats['f1_pct']:>6.1f}%"
    )
