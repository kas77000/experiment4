"""Score NEW orders against a threshold that was already fitted. No refitting.

    python -m tier3_model.run --csv 2025_q1.csv     # fit once, freezes the threshold
    python score_new.py 2025_q2.csv                 # apply it to the next quarter

This is the operating mode the whole exercise builds toward. Fitting and scoring
the same book is close to circular -- 1.5% of orders flag because 1.5% was
*defined* as flagging. Applying a frozen threshold to orders it has never seen
turns that number into a measurement: if this quarter flags 4% against a gate
set at 1.5%, something actually changed.

What it prints, and why the drift block matters as much as the queue:

    THRESHOLD      when it was fitted, on how many orders, with which settings
    DRIFT          did the new book move away from the one it was fitted on?
    QUEUE          the orders to review, with a diagnosed cause and a cash cost

A frozen threshold decays silently -- nothing errors, the numbers just quietly
stop meaning what they did. The drift block separates the two reasons the flag
rate can move: the market changed (feature medians shifted -> recalibrate) or
your execution changed (features stable, flag rate moved -> a real finding).
Without it you cannot tell those apart, and they call for opposite responses.
"""

from __future__ import annotations
import argparse
import os
import sys

import pandas as pd

import config as root_config
from tca import dataset, pipeline, report, schema
from tier3_model import config as t3cfg, cost_model, diagnostics, persist, scoring

DEFAULT_MODEL = os.path.join(dataset.OUT_DIR, "tier3", "model.json")


def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="New orders to score (.csv, .xlsx, .parquet)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Frozen threshold to apply (default: outputs/tier3/model.json)")
    ap.add_argument("--out", default=None,
                    help="Where to write the review queue (default: outputs/tier3/)")
    ap.add_argument("--examples", type=int, default=3,
                    help="How many flagged orders to narrate in full.")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"No frozen threshold at {args.model}\n"
                 f"Fit one first:  python -m tier3_model.run --csv <history>.csv")
    if not os.path.exists(args.path):
        sys.exit(f"No such file: {args.path}")

    model, cfg, ref = persist.load(args.model, t3cfg.CONFIG)

    print(report.header("SCORING NEW ORDERS AGAINST A FROZEN THRESHOLD"))
    print(f"\n=== Threshold ===")
    print(f"  file           {args.model}")
    print(f"  backend        {model.backend}")
    print(f"  band           tau {cfg.tau_lo:g} / {cfg.tau_hi:g}"
          f"   (nominal {100*(cfg.tau_lo + 1 - cfg.tau_hi):.2f}% flagged)")
    print(f"  fitted on      {model.n_train:,} orders")
    if ref.get("flag_rate_pct") is not None:
        print(f"  flagged then   {ref['flag_rate_pct']:.2f}%")
    print("\n  Nothing is refitted here. The surface below is exactly the one")
    print("  fitted on the history file, applied unchanged.")

    # Same pipeline as fitting: identical mapping, units and cleaning.
    raw = read_any(args.path)
    df, clean_report = pipeline.prepare(
        raw, root_config.COLUMN_MAP, root_config.DATA, root_config.SLIPPAGE_SIGN,
        pre_transform=getattr(root_config, "PRE_TRANSFORM", None))
    print("\n=== New data ===")
    print(clean_report.as_text())

    if not len(df):
        sys.exit("No usable rows after cleaning.")

    preds = cost_model.predict(model, df, cfg)
    scored = scoring.add_scores(df, preds, cfg)
    causes = diagnostics.fit_causes(scored, cfg)
    attributed = diagnostics.attribute(scored, causes)

    # ---------------- drift ----------------
    print(report.header("DRIFT --- is the threshold still valid here?"))
    drift, warnings = persist.drift_report(df, attributed, ref, cfg)
    print(report.frame(drift))
    if warnings:
        print()
        for w in warnings:
            print(f"  WARNING: {w}")
    else:
        print("\n  No drift worth acting on. The threshold still fits this book.")

    # ---------------- the queue ----------------
    print(report.header("REVIEW QUEUE"))
    print(report.zone_summary(attributed))
    print("\n=== Severity ===")
    print(report.frame(scoring.severity_summary(attributed)))
    print("\n=== Causes ===")
    print(report.frame(diagnostics.cause_summary(
        attributed, currency=root_config.DATA.notional_currency)))

    queue = attributed[attributed["review_required"]].sort_values("z")
    if args.examples and len(queue):
        print(report.header("WORST ORDERS"))
        for _, row in queue.head(args.examples).iterrows():
            print()
            print(diagnostics.explain_order(row, model, causes, cfg))

    keep = [c for c in [schema.ORDER_ID, schema.ALGO, schema.BROKER, schema.SYMBOL,
                        schema.SIDE, schema.PCT_ADV, schema.PARTICIPATION,
                        schema.DURATION_MIN, schema.SPREAD_BPS, schema.VOLATILITY,
                        schema.NOTIONAL, schema.SLIPPAGE_BPS, "expected_bps",
                        "band_lo_bps", "band_hi_bps", "band_lo_spreads",
                        "band_hi_spreads", "residual_bps",
                        "shortfall_ccy", "z", "zone", "severity", "flagged",
                        "review_required", "likely_cause", "remedy"]
            if c in attributed.columns]

    out_dir = args.out or os.path.join(dataset.OUT_DIR, "tier3")
    os.makedirs(out_dir, exist_ok=True)
    scored_path = os.path.join(out_dir, "new_scored_orders.csv")
    queue_path = os.path.join(out_dir, "new_review_queue.csv")
    attributed[keep].to_csv(scored_path, index=False)
    queue[keep].to_csv(queue_path, index=False)
    print(f"\nWrote {scored_path}")
    print(f"      {queue_path}")


if __name__ == "__main__":
    main()
