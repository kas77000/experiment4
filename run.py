"""End-to-end demo / driver.

    python run.py                 # run on synthetic HK data
    python run.py --csv path.csv  # run on your real extract (via config COLUMN_MAP)

Outputs:
    outputs/threshold_table.csv   # the range per algo x market x bucket
    outputs/scored_orders.csv     # every order tagged IN_RANGE/OUT_LOW/OUT_HIGH + flagged
"""

from __future__ import annotations
import argparse
import os

import pandas as pd

import config
import synthetic_data
from tca import pipeline, thresholds, report

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="Path to real extract; omit to use synthetic data.")
    ap.add_argument("--n", type=int, default=12000, help="Synthetic row count.")
    args = ap.parse_args()

    if args.csv:
        print(f"Loading extract: {args.csv}")
        df_raw = pd.read_csv(args.csv)
    else:
        print("No --csv given; generating synthetic HK VWAP data.")
        df_raw = synthetic_data.generate(n=args.n)

    cfg = config.CONFIG

    # 1) prepare
    df, clean_report = pipeline.prepare(
        df_raw, config.COLUMN_MAP, cfg, config.SLIPPAGE_SIGN)
    print("\n=== Cleaning ===")
    print(clean_report.as_text())

    # 2) fit thresholds
    table = thresholds.fit(df, cfg)
    print("\n=== Threshold table (range in units of spread) ===")
    print(report.format_threshold_table(table))

    n_untrusted = int((~table["trusted"]).sum())
    if n_untrusted:
        print(f"\nNote: {n_untrusted} group(s) below min_group_n={cfg.min_group_n}"
              f" -> those orders fall back to pooled/global bands when scored.")

    # 3) score every order back against its band
    model = thresholds.ThresholdModel(table, cfg)
    scored = model.score_frame(df)
    print("\n=== Zone distribution (in-sample) ===")
    print(report.zone_summary(scored))

    # 4) show a few concrete flagged examples
    flagged = scored[scored["flagged"]].head(5)
    if len(flagged):
        cols = ["order_id", "algo", "adv_bucket", "spread_bps",
                "slippage_bps", "perf_in_spreads", "zone", "band_level"]
        print("\n=== Example flagged orders ===")
        print(flagged[cols].round(3).to_string(index=False))

    # 5) demo the single-order scoring API
    print("\n=== Single-order scoring API demo ===")
    demo = model.score_order(algo="VWAP", market="HK",
                             slippage_bps=-38.0, spread_bps=9.0, pct_adv=3.2)
    print(f"  VWAP HK, -38bps slippage on 9bps spread, 3.2% ADV -> {demo}")

    # 6) persist
    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(os.path.join(OUT_DIR, "threshold_table.csv"), index=False)
    scored.to_csv(os.path.join(OUT_DIR, "scored_orders.csv"), index=False)
    print(f"\nWrote outputs/ -> threshold_table.csv, scored_orders.csv")


if __name__ == "__main__":
    main()
