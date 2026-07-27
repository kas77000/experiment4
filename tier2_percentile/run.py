"""Tier 2 driver:  python -m tier2_percentile.run  [--peer-csv street.csv]

    --peer-csv   fit the bands on an external peer universe instead of your own
                 history (this is what you are buying from a TCA vendor)
    --metric     perf_in_spreads (default) | perf_norm

Outputs: outputs/tier2/band_table.csv, outputs/tier2/scored_orders.csv
"""

from __future__ import annotations
import argparse
import dataclasses

import pandas as pd

import config as root_config
from tca import dataset, evaluate, pipeline, report, schema
from tier2_percentile import config as t2cfg
from tier2_percentile import thresholds


def main():
    ap = dataset.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--peer-csv", help="External peer universe to fit the bands on.")
    ap.add_argument("--metric", choices=[schema.PERF_IN_SPREADS, schema.PERF_NORM],
                    help="Override the banded metric from tier2_percentile/config.py.")
    args = ap.parse_args()

    cfg = t2cfg.CONFIG
    if args.metric:
        cfg = dataclasses.replace(cfg, metric=args.metric)

    df, clean_report = dataset.load_prepared(args)

    print(report.header("TIER 2 --- EMPIRICAL PERCENTILE BANDS"))
    print("\n=== Cleaning ===")
    print(clean_report.as_text())

    # Reference book: own history, or an external peer universe.
    if args.peer_csv:
        print(f"\nFitting bands on peer universe: {args.peer_csv}")
        peer_raw = pd.read_csv(args.peer_csv)
        ref, peer_clean = pipeline.prepare(
            peer_raw, root_config.COLUMN_MAP, root_config.DATA,
            root_config.SLIPPAGE_SIGN,
            pre_transform=getattr(root_config, "PRE_TRANSFORM", None))
        print(f"  peer rows usable: {len(ref):,}")
    else:
        print("\nFitting bands on your own history (no --peer-csv given).")
        ref = df

    lo, hi = cfg.range_percentiles
    print(f"\n=== Band ===\n  metric={cfg.metric}  percentiles=p{lo:g}/p{hi:g}"
          f"  min_group_n={cfg.min_group_n}")

    table = thresholds.fit(ref, cfg)
    print("\n=== Band table ===")
    show = table.copy()
    for c in ["q_lo", "q_median", "q_hi", "mad"]:
        show[c] = show[c].round(3)
    print(report.frame(show[["level", schema.ALGO, schema.MARKET, schema.ADV_BUCKET,
                             "n", "trusted", "q_lo", "q_median", "q_hi"]], max_rows=40))

    n_untrusted = int((~table["trusted"]).sum())
    if n_untrusted:
        print(f"\n  {n_untrusted} group(s) below min_group_n={cfg.min_group_n}"
              f" -> those orders fall back to a pooled/global band.")

    model = thresholds.ThresholdModel(table, cfg)
    scored = model.score_frame(df)

    print("\n=== Zone distribution ===")
    print(report.zone_summary(scored))

    print("\n=== Flag rate vs order difficulty ===")
    print(report.frame(thresholds.flag_rate_by_bucket(scored)))
    print("\n  Flatter than Tier 1: the band adapts to the bucket. But it is still")
    print("  a step function -- everything inside one bucket shares a threshold.")

    if evaluate.has_truth(scored):
        print("\n=== Detection vs known truth (synthetic only) ===")
        print(evaluate.format_stats(evaluate.detection_stats(scored)))
        print("\n  recall by failure type:")
        print(report.frame(evaluate.recall_by_cause(scored)))
        print("\n  NOTE: these flags are IN-SAMPLE -- the bands were fitted on the")
        print("  same orders being scored, so this flatters Tier 2. Tier 3 reports")
        print("  out-of-sample numbers via cross-fitting.")

    print("\n=== Single-order scoring API demo ===")
    demo = model.score_order(algo="VWAP", market="HK",
                             slippage_bps=-38.0, spread_bps=9.0, pct_adv=3.2)
    print(f"  VWAP HK, -38bps on a 9bps spread, 3.2% ADV ->")
    for k, v in demo.items():
        print(f"    {k:<16} {v}")

    table.to_csv(dataset.out_path("tier2", "band_table.csv"), index=False)
    cols = [c for c in [schema.ORDER_ID, schema.ALGO, schema.ADV_BUCKET,
                        schema.SPREAD_BPS, schema.SLIPPAGE_BPS, cfg.metric,
                        "band_lo", "band_hi", "band_level", "zone", "flagged",
                        "review_required"] if c in scored.columns]
    scored[cols].to_csv(dataset.out_path("tier2", "scored_orders.csv"), index=False)
    print(f"\nWrote outputs/tier2/ -> band_table.csv, scored_orders.csv")


if __name__ == "__main__":
    main()
