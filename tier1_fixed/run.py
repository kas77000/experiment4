"""Tier 1 driver:  python -m tier1_fixed.run  [--rule sigma_multiple] [--csv x.csv]

Outputs: outputs/tier1/scored_orders.csv
"""

from __future__ import annotations
import argparse
import dataclasses

from tca import dataset, evaluate, report, schema
from tier1_fixed import config as t1cfg
from tier1_fixed import rules


def build(args=None):
    """Run Tier 1 end to end. Returns (scored_df, cfg) for reuse by run_all."""
    cfg = t1cfg.CONFIG
    if args is not None and getattr(args, "rule", None):
        cfg = dataclasses.replace(cfg, rule=args.rule)
    return cfg


def main():
    ap = dataset.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--rule", choices=list(rules.RULES),
                    help="Override the fixed rule from tier1_fixed/config.py.")
    args = ap.parse_args()

    cfg = build(args)
    df, clean_report = dataset.load_prepared(args)

    print(report.header("TIER 1 --- FIXED THRESHOLD"))
    print("\n=== Cleaning ===")
    print(clean_report.as_text())
    print(f"\n=== Rule ===\n  {rules.describe(cfg)}")

    scored = rules.score(df, cfg)

    print("\n=== Zone distribution ===")
    print(report.zone_summary(scored))

    print("\n=== The problem with this tier: flag rate vs order difficulty ===")
    print(report.frame(rules.flag_rate_by_bucket(scored)))
    print("\n  A calibrated threshold would show a roughly FLAT flag_rate column.")
    print("  A fixed limit does not adjust for difficulty, so size drives the flag.")

    if evaluate.has_truth(scored):
        print("\n=== Detection vs known truth (synthetic only) ===")
        print(evaluate.format_stats(evaluate.detection_stats(scored)))
        print("\n  recall by failure type:")
        print(report.frame(evaluate.recall_by_cause(scored)))

    cols = [c for c in [schema.ORDER_ID, schema.ALGO, schema.ADV_BUCKET,
                        schema.SPREAD_BPS, schema.SLIPPAGE_BPS, "rule_stat",
                        "rule_limit", "zone", "flagged", "review_required"]
            if c in scored.columns]
    path = dataset.out_path("tier1", "scored_orders.csv")
    scored[cols].to_csv(path, index=False)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
