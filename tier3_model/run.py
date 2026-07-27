"""Tier 3 driver:  python -m tier3_model.run  [--folds 5] [--backend quantreg]

Full Phase 2: volatility-aware normalization -> quantile-regression cost model
-> out-of-sample residual z-scores -> tiered review queue -> slice t-tests ->
cause attribution.

Outputs (all under outputs/tier3/):
    model_coefficients.csv   the fitted quantile surfaces
    calibration.csv          nominal vs realized band coverage, out-of-sample
    scored_orders.csv        every order with expected cost, z, zone, cause
    threshold_table.csv      the fitted band in bps, summarised by group
    review_queue.csv         just what a human needs to look at, worst first
    slice_findings.csv       systematic effects that survive FDR correction
"""

from __future__ import annotations
import argparse
import dataclasses

import config as root_config
from tca import dataset, evaluate, report, schema
from tier3_model import (aggregate, config as t3cfg, cost_model, diagnostics,
                         features, persist, scoring)


def build(df, cfg, seed: int = 0):
    """Fit, cross-fit-score and attribute. Returns (attributed, model, preds)."""
    preds, model = cost_model.cross_fit_predict(df, cfg, seed=seed)
    scored = scoring.add_scores(df, preds, cfg)
    causes = diagnostics.fit_causes(scored, cfg)
    attributed = diagnostics.attribute(scored, causes)
    return attributed, model, preds, causes


def main():
    ap = dataset.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--folds", type=int, help="Cross-fitting folds (1 disables).")
    ap.add_argument("--backend", choices=["auto", "quantreg", "empirical"])
    ap.add_argument("--algo-effect", choices=["absorb", "expose"],
                    help="absorb: judge orders within their algo. "
                         "expose: let a structurally worse algo flag everywhere.")
    ap.add_argument("--examples", type=int, default=3,
                    help="How many flagged orders to narrate in full.")
    args = ap.parse_args()

    cfg = t3cfg.CONFIG
    over = {k: v for k, v in [("n_folds", args.folds),
                              ("backend", args.backend),
                              ("algo_effect", args.algo_effect)] if v is not None}
    if over:
        cfg = dataclasses.replace(cfg, **over)

    df, clean_report = dataset.load_prepared(args)

    print(report.header("TIER 3 --- EXPECTED-COST MODEL + RESIDUAL Z-SCORES"))
    print("\n=== Cleaning ===")
    print(clean_report.as_text())

    print("\n=== Difficulty input coverage ===")
    print(report.frame(features.coverage(df, cfg)))

    attributed, model, preds, causes = build(df, cfg, seed=args.seed)

    # ---------------- model ----------------
    print("\n=== Cost model ===")
    print(f"  backend        {model.backend}")
    print(f"  taus           {model.taus}")
    print(f"  algo_effect    {cfg.algo_effect}")
    print(f"  train rows     {model.n_train:,}  (trimmed {model.n_trimmed:,} = "
          f"{100*cfg.fit_trim_quantile:g}% per tail, from FITTING only; still scored)")
    print(f"  cross-fitting  {cfg.n_folds} folds"
          if cfg.n_folds and cfg.n_folds > 1 else "  cross-fitting  DISABLED (in-sample)")
    if model.pseudo_r1:
        r1 = "  ".join(f"tau={t:g}: {v:.3f}" for t, v in model.pseudo_r1.items())
        print(f"  pseudo-R1      {r1}")

    coefs = model.coef_frame()
    if len(coefs):
        print("\n  fitted coefficients (perf_norm units, features standardized):")
        print(report.frame(coefs))

    # ---------------- calibration ----------------
    print("\n=== Calibration (OUT OF SAMPLE) ===")
    print(report.frame(cost_model.coverage_check(df, preds, cfg)))
    print("\n  Realized should sit close to nominal. Far above -> the model is")
    print("  missing a driver. Far below -> it is overfitting and will under-flag.")
    print("  This is the only validation available on real data, where you never")
    print("  learn which orders were 'really' bad.")

    # ---------------- THE THRESHOLD ----------------
    print("\n=== THE THRESHOLD, in bps ===")
    print(report.frame(scoring.threshold_table(attributed)))
    print("\n  There is no single threshold in Tier 3 -- every order gets its own,")
    print("  predicted from its own size, spread, volatility, duration and urgency.")
    print("  Per-order values are the band_lo_bps / band_hi_bps columns of")
    print("  scored_orders.csv. The table above is the median band per group, so")
    print("  the surface can be read.")
    print("\n  band_lo_p10 vs band_lo_p90 is the point: that is how much the")
    print("  threshold MOVES inside one cell. Tier 2 would have a single number")
    print("  there for every order in the group.")

    # ---------------- the queue ----------------
    print("\n=== Zone distribution ===")
    print(report.zone_summary(attributed))
    print("\n=== Severity tiers ===")
    print(report.frame(scoring.severity_summary(attributed)))

    print("\n=== Flag rate vs order difficulty ===")
    print(report.frame(scoring.flag_rate_by_bucket(attributed)))
    print("\n  This column should be FLAT -- that is the whole point. Difficulty is")
    print("  now in the expectation, so what is left to flag is execution quality.")

    # ---------------- systematic findings ----------------
    print("\n=== Slice findings (mean-z t-tests, BH-corrected) ===")
    slices = aggregate.slice_report(attributed)
    sig = aggregate.significant(slices)
    if len(sig):
        print(report.frame(sig, max_rows=25))
        print("\n  These are the systematic problems. Note they are invisible to any")
        print("  single-order threshold: a 0.2-sigma-per-order effect barely moves")
        print("  the tail rate, but over thousands of orders it is unmistakable.")
    else:
        print("  No slice survives FDR correction.")

    # ---------------- causes ----------------
    print("\n=== Cause attribution across the review queue ===")
    print(report.frame(diagnostics.cause_summary(
        attributed, currency=root_config.DATA.notional_currency)))

    conf = diagnostics.cause_confusion(attributed)
    if len(conf):
        print("\n  attributed cause vs KNOWN cause (synthetic only):")
        print(report.frame(conf))
        print("\n  attribution accuracy:")
        print(report.frame(diagnostics.cause_accuracy(attributed)))

    # ---------------- worked examples ----------------
    queue = attributed[attributed["review_required"]].sort_values("z")
    if args.examples and len(queue):
        print(report.header("WORKED EXAMPLES --- WORST ORDERS IN THE QUEUE"))
        for _, row in queue.head(args.examples).iterrows():
            print()
            print(diagnostics.explain_order(row, model, causes, cfg))

    # ---------------- detection vs truth ----------------
    if evaluate.has_truth(attributed):
        print("\n=== Detection vs known truth (synthetic only, OUT OF SAMPLE) ===")
        print(evaluate.format_stats(evaluate.detection_stats(attributed)))
        print("\n  recall by failure type:")
        print(report.frame(evaluate.recall_by_cause(attributed)))

    # ---------------- persist ----------------
    if len(coefs):
        coefs.to_csv(dataset.out_path("tier3", "model_coefficients.csv"))
    cost_model.coverage_check(df, preds, cfg).to_csv(
        dataset.out_path("tier3", "calibration.csv"), index=False)
    scoring.threshold_table(attributed).to_csv(
        dataset.out_path("tier3", "threshold_table.csv"), index=False)

    # Freeze the fitted threshold so future orders can be scored against it
    # without refitting. This is the artefact the whole exercise produces.
    model_path = persist.save(model, cfg, dataset.out_path("tier3", "model.json"),
                              df=df, scored=attributed)

    keep = [c for c in [schema.ORDER_ID, schema.ALGO, schema.BROKER, schema.SYMBOL,
                        schema.SIDE, schema.PCT_ADV, schema.PARTICIPATION,
                        schema.DURATION_MIN, schema.SPREAD_BPS, schema.VOLATILITY,
                        schema.NOTIONAL, schema.SLIPPAGE_BPS,
                        schema.SIGMA_EXPECTED_BPS, schema.PERF_NORM,
                        "expected_bps", "band_lo_bps", "band_hi_bps",
                        "band_lo_spreads", "band_hi_spreads", "expected_spreads",
                        "residual_bps", "shortfall_ccy", "z", "zone", "severity", "flagged",
                        "review_required", "likely_cause", "remedy"]
            if c in attributed.columns]
    attributed[keep].to_csv(dataset.out_path("tier3", "scored_orders.csv"), index=False)
    queue[keep].to_csv(dataset.out_path("tier3", "review_queue.csv"), index=False)
    if len(slices):
        slices.to_csv(dataset.out_path("tier3", "slice_findings.csv"), index=False)

    print("\nWrote outputs/tier3/ -> model.json, model_coefficients.csv,")
    print("                        calibration.csv, threshold_table.csv,")
    print("                        scored_orders.csv, review_queue.csv,")
    print("                        slice_findings.csv")
    print(f"\nThreshold frozen to {model_path}")
    print("Score future orders against it WITHOUT refitting:")
    print("    python score_new.py next_quarter.csv")


if __name__ == "__main__":
    main()
