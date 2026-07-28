"""Tier 4 driver:  python -m tier4_vwap.run  [--csv x.csv] [--no-debias]

VWAP-native thresholds:
    de-bias the benchmark by 1/(1-PR)  ->  tracking-error scale  ->  quantile
    regression on curve difficulty  ->  z-scores, queue, slices, causes

Outputs under outputs/tier4/, same shape as Tier 3.
"""

from __future__ import annotations
import argparse
import dataclasses

import config as root_config
from tca import dataset, evaluate, report, schema
from tier3_model import aggregate, cost_model, diagnostics, persist, scoring
from tier3_model import config as t3cfg
from tier3_model import features as t3features
from tier4_vwap import config as t4cfg
from tier4_vwap import features as t4features
from tier4_vwap import metric


def build(df, cfg, seed: int = 0):
    """De-bias, rescale, fit, cross-fit-score and attribute."""
    adj, notes = metric.apply(df, cfg, root_config.DATA)
    preds, model = cost_model.cross_fit_predict(adj, cfg, seed=seed,
                                                feats=t4features)
    scored = scoring.add_scores(adj, preds, cfg)
    causes = diagnostics.fit_causes(scored, cfg)
    attributed = diagnostics.attribute(scored, causes)
    return attributed, model, preds, causes, notes


def main():
    ap = dataset.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--no-debias", action="store_true",
                    help="Skip the 1/(1-PR) correction, to see what it is worth.")
    ap.add_argument("--compare-tier3", action="store_true",
                    help="Fit Tier 3 on the same book and show both coefficients.")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    cfg = t4cfg.CONFIG
    if args.no_debias:
        cfg = dataclasses.replace(cfg, debias_benchmark=False)

    df, clean_report = dataset.load_prepared(args)

    print(report.header("TIER 4 --- VWAP-NATIVE THRESHOLDS"))
    print("\n=== Cleaning ===")
    print(clean_report.as_text())

    attributed, model, preds, causes, notes = build(df, cfg, seed=args.seed)

    # ---------------- the de-biasing ----------------
    print("\n=== 1. Benchmark de-biasing ===")
    for n in notes:
        print(f"  {n}")
    print("\n  Against interval VWAP you are part of your own benchmark. With f")
    print("  your share of interval volume, VWAP_total = (1-f)*VWAP_others +")
    print("  f*P_you, so slippage vs the rest of the market is exactly the")
    print("  reported figure divided by (1-f). Algebra, not a model.")

    dil = metric.dilution_summary(attributed)
    if len(dil):
        print()
        print(report.frame(dil))
        print("\n  The gradient is the argument: the correction grows with order")
        print("  size, so the biggest orders -- the ones a contaminated benchmark")
        print("  flatters most -- are exactly the ones being under-measured.")

    # ---------------- the model ----------------
    print("\n=== 2. Cost model (curve difficulty, not impact) ===")
    print(f"  backend        {model.backend}")
    print(f"  scale          sqrt((({cfg.k_spread:g}*spread)^2 + "
          f"({cfg.k_track:g}*vol*sqrt(T/S))^2)")
    print(f"  train rows     {model.n_train:,}")
    if model.pseudo_r1:
        print("  pseudo-R1      " + "  ".join(
            f"tau={t:g}: {v:.3f}" for t, v in model.pseudo_r1.items()))

    coefs = model.coef_frame()
    if len(coefs):
        print("\n  fitted coefficients (features standardized):")
        print(report.frame(coefs))

    if args.compare_tier3:
        print("\n  --- Tier 3 on the same book, for comparison ---")
        t3 = t3cfg.CONFIG
        m3 = cost_model.fit(df, t3, feats=t3features)
        c3 = m3.coef_frame()
        if len(c3):
            print(report.frame(c3))
        print("\n  Look at sqrt_adv. If the VWAP argument holds it should be much")
        print("  weaker here than in Tier 3, because size drives impact (Tier 3's")
        print("  premise) but not curve-tracking error (Tier 4's).")

    # ---------------- calibration ----------------
    print("\n=== 3. Calibration (OUT OF SAMPLE) ===")
    print(report.frame(cost_model.coverage_check(df, preds, cfg)))

    # ---------------- threshold ----------------
    print("\n=== 4. THE THRESHOLD ===")
    tt = scoring.threshold_table(attributed)
    print(report.frame(tt, max_rows=60))
    print("\n  In bps vs the REST OF THE MARKET, not vs the diluted benchmark.")

    # ---------------- queue ----------------
    print("\n=== 5. Zone distribution ===")
    print(report.zone_summary(attributed))
    print("\n=== Flag rate vs order difficulty ===")
    print(report.frame(scoring.flag_rate_by_bucket(attributed)))

    print("\n=== 6a. Slice findings --- BIAS (mean-z t-tests, BH-corrected) ===")
    slices = aggregate.slice_report(attributed)
    sig = aggregate.significant(slices)
    print(report.frame(sig, max_rows=20) if len(sig)
          else "  No slice survives FDR correction.")

    print("\n=== 6b. Slice findings --- CONSISTENCY (Levene on z variance) ===")
    disp = aggregate.dispersion_report(attributed)
    dsig = disp[disp["verdict"] != "no evidence"] if len(disp) else disp
    print(report.frame(dsig, max_rows=20) if len(dsig)
          else "  No slice differs in consistency.")
    print("\n  This is the test that matters for a schedule-following algo, and")
    print("  6a cannot substitute for it. Poor volume-curve tracking does not bias")
    print("  an order in a direction -- it WIDENS the distribution. The order then")
    print("  lands somewhere random on a wider spread, so on average it looks fine")
    print("  while being unreliable order by order.")
    print("\n  A desk that is inconsistent rather than consistently bad shows up")
    print("  here and nowhere else.")

    print("\n=== Cause attribution ===")
    print(report.frame(diagnostics.cause_summary(
        attributed, currency=root_config.DATA.notional_currency)))
    conf = diagnostics.cause_confusion(attributed)
    if len(conf):
        print("\n  attributed vs KNOWN cause (synthetic only):")
        print(report.frame(conf))

    queue = attributed[attributed["review_required"]].sort_values("z")
    if args.examples and len(queue):
        print(report.header("WORKED EXAMPLES"))
        for _, row in queue.head(args.examples).iterrows():
            print()
            print(diagnostics.explain_order(row, model, causes, cfg,
                                            feats=t4features))

    if evaluate.has_truth(attributed):
        print("\n=== Detection vs known truth (OUT OF SAMPLE) ===")
        print(evaluate.format_stats(evaluate.detection_stats(attributed)))
        print("\n  recall by failure type:")
        print(report.frame(evaluate.recall_by_cause(attributed)))

    # ---------------- persist ----------------
    if len(coefs):
        coefs.to_csv(dataset.out_path("tier4", "model_coefficients.csv"))
    cost_model.coverage_check(df, preds, cfg).to_csv(
        dataset.out_path("tier4", "calibration.csv"), index=False)
    tt.to_csv(dataset.out_path("tier4", "threshold_table.csv"), index=False)

    keep = [c for c in [schema.ORDER_ID, schema.ALGO, schema.BROKER, schema.SYMBOL,
                        schema.SIDE, schema.PCT_ADV, schema.PARTICIPATION,
                        schema.DURATION_MIN, schema.SPREAD_BPS, schema.VOLATILITY,
                        schema.NOTIONAL, schema.REPORTED_SLIPPAGE_BPS,
                        schema.DILUTION_FACTOR, schema.DILUTION_CAPPED,
                        schema.SLIPPAGE_BPS, "expected_bps", "band_lo_bps",
                        "band_hi_bps", "band_lo_spreads", "band_hi_spreads",
                        "residual_bps", "shortfall_ccy", "z", "zone", "severity",
                        "flagged", "review_required", "likely_cause", "remedy"]
            if c in attributed.columns]
    attributed[keep].to_csv(dataset.out_path("tier4", "scored_orders.csv"),
                            index=False)
    queue[keep].to_csv(dataset.out_path("tier4", "review_queue.csv"), index=False)
    if len(slices):
        slices.to_csv(dataset.out_path("tier4", "slice_findings.csv"), index=False)
    if len(disp):
        disp.to_csv(dataset.out_path("tier4", "consistency_findings.csv"), index=False)

    path = persist.save(model, cfg, dataset.out_path("tier4", "model.json"),
                        df=attributed, scored=attributed)
    print(f"\nWrote outputs/tier4/  |  threshold frozen to {path}")


if __name__ == "__main__":
    main()
