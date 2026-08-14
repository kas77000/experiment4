"""Run both methods on the SAME orders and compare them head to head.

    python run.py                    # synthetic HK VWAP demo
    python run.py --csv your.csv     # your real extract
    python run.py --budget 3         # hold both methods to a 3% review queue

For a single method with its full report, run it directly:

    python -m tier3_model.run
    python -m tier5_gaussian.run

The comparison has two halves, and the second is the one that matters.

  "At each method's own threshold" is how they would actually behave in
  production -- but it is not a fair test of METHOD, because they flag
  different fractions of the book and recall rises trivially with queue size.

  "At a matched review budget" fixes that: each method ranks the entire book
  by its own severity statistic, the top N% go to the queue, and we ask who
  filled that fixed-size queue with real problems. That is a comparison of
  ranking quality, independent of where anyone set their limit.
"""

from __future__ import annotations
import argparse

import pandas as pd

import config as root_config
from tca import dataset, evaluate, report, schema
from tier3_model import (aggregate, config as t3cfg, cost_model, diagnostics,
                         scoring)
from tier5_gaussian import band, config as t5cfg, normality


def run_tier3(df, seed: int):
    cfg = t3cfg.CONFIG
    preds, model = cost_model.cross_fit_predict(df, cfg, seed=seed)
    scored = scoring.add_scores(df, preds, cfg)
    causes = diagnostics.fit_causes(scored, cfg)
    attributed = diagnostics.attribute(scored, causes)
    desc = (f"quantile regression tau={cfg.tau_lo:g}/{cfg.tau_hi:g}, "
            f"{cfg.n_folds}-fold out-of-sample")
    return attributed, desc, preds, cfg


def run_tier5(df):
    cfg = t5cfg.CONFIG
    table = band.fit(df, cfg)
    model = band.BandModel(table, cfg)
    desc = (f"mu +/- {cfg.k_sigma:g} sigma on {cfg.metric}, "
            f"{cfg.estimator}, {cfg.score_level}, in-sample")
    return model.score_frame(df), desc, table, cfg


def main():
    ap = dataset.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--budget", type=float, default=3.0,
                    help="Matched review-queue size, in %% of orders.")
    args = ap.parse_args()

    df, clean_report = dataset.load_prepared(args)
    print("\n=== Cleaning (shared by both methods) ===")
    print(clean_report.as_text())

    t3, d3, preds3, cfg3 = run_tier3(df, seed=args.seed)
    t5, d5, table5, cfg5 = run_tier5(df)
    methods = [("Tier 3 model", t3, d3), ("Tier 5 gaussian", t5, d5)]

    # ---------------- behaviour at each method's own threshold ----------------
    print(report.header("1. AT EACH METHOD'S OWN THRESHOLD"))
    rows = []
    for name, scored, desc in methods:
        r = {"method": name, "rule": desc,
             "flag_rate_pct": round(100 * scored["flagged"].mean(), 2),
             "review_pct": round(100 * scored["review_required"].mean(), 2)}
        r.update({k: round(v, 1) for k, v in
                  evaluate.detection_stats(scored).items()
                  if k in ("precision_pct", "recall_pct", "f1_pct")})
        rows.append(r)
    print(report.frame(pd.DataFrame(rows)))

    # ---------------- calibration across difficulty ----------------
    print(report.header("2. IS THE THRESHOLD CALIBRATED? (flag rate by %ADV bucket)"))
    order = [b for b in ["<1%", "1-5%", "5-10%", "10-20%", ">20%", "unknown"]
             if b in set(df[schema.ADV_BUCKET])]
    cal = pd.DataFrame({
        name: (100 * scored.groupby(schema.ADV_BUCKET, observed=False)["flagged"].mean())
        for name, scored, _ in methods
    }).reindex(order).round(2)
    cal.insert(0, "n", df.groupby(schema.ADV_BUCKET, observed=False).size().reindex(order))
    print(report.frame(cal))
    cols = [c for c in cal.columns if c != "n"]
    spread = (cal[cols].max() - cal[cols].min()).round(2)
    print("\n  flag-rate spread across buckets (lower = better calibrated):")
    for k, v in spread.items():
        print(f"    {k:<20} {v:>6.2f} pp")
    print("\n  A calibrated threshold flags roughly the same share of easy and hard")
    print("  orders. Anything else means you are measuring difficulty, not quality.")

    # ---------------- matched-budget ranking quality ----------------
    if evaluate.has_truth(df):
        print(report.header(f"3. AT A MATCHED {args.budget:g}% REVIEW BUDGET "
                            f"(the fair comparison)"))
        rows = []
        for name, scored, _ in methods:
            s = evaluate.precision_at_budget(scored, args.budget)
            if s:
                rows.append({"method": name, "queue": s["queue"],
                             "caught": s["caught"],
                             "precision_pct": round(s["precision_pct"], 1),
                             "recall_pct": round(s["recall_pct"], 1)})
        print(report.frame(pd.DataFrame(rows)))
        print(f"\n  {int(df[schema.TRUE_OUTLIER].sum()):,} orders in this book were "
              f"genuinely broken. Each method got the same")
        print("  size queue; the difference is purely how well it ranked.")

        print("\n  recall by failure type, at the matched budget:")
        rec = {}
        for name, scored, _ in methods:
            k = max(int(round(args.budget / 100 * len(scored))), 1)
            top = scored["rank_stat"].fillna(-1e18).nlargest(k).index
            pick = pd.Series(False, index=scored.index)
            pick.loc[top] = True
            real = scored[scored[schema.TRUE_OUTLIER]]
            rec[name] = (100 * pick.loc[real.index]
                         .groupby(real[schema.TRUE_CAUSE]).mean()).round(1)
        print(report.frame(pd.DataFrame(rec)))

    # ---------------- does the Gaussian assumption hold? ----------------
    print(report.header("4. DOES THE 3-SIGMA PROMISE HOLD?"))
    headline = table5[table5["level"] == t5cfg.LEVEL_ALL].iloc[0]
    c = headline[f"centre_{cfg5.estimator}"]
    s = headline[f"scale_{cfg5.estimator}"]
    x = df[cfg5.metric].to_numpy()
    print(report.frame(normality.coverage_table(x, c, s).round(3)))
    req = normality.required_k(x, c, s)
    print(f"\n  To actually flag {100*normality.NOMINAL_OUTSIDE:.2f}% of this book"
          f" you need k = {req['k_symmetric']:.2f}, not {cfg5.k_sigma:g}.")
    print("  That gap is the cost of assuming a shape the data does not have.")

    # ---------------- what only Tier 3 gives you ----------------
    print(report.header("5. WHAT ONLY TIER 3 PRODUCES"))

    print("\n--- Out-of-sample calibration (the check that works on real data) ---")
    print(report.frame(cost_model.coverage_check(df, preds3, cfg3)))

    print("\n--- Systematic effects: mean-z t-tests, BH-corrected ---")
    sig = aggregate.significant(aggregate.slice_report(t3))
    if len(sig):
        cols = [c for c in ["dimension", schema.ALGO, schema.BROKER,
                            schema.ADV_BUCKET, "n", "mean_z", "t_stat",
                            "q_value", "verdict"] if c in sig.columns]
        print(report.frame(sig[cols].head(12)))
        print("\n  Tier 5 cannot produce this table at all: without an expected")
        print("  cost there is no residual to average, so a broker that is")
        print("  consistently 0.2 sigma worse is indistinguishable from one")
        print("  that simply got handed the harder orders.")
    else:
        print("  none significant")

    print("\n--- Cause attribution across the review queue ---")
    print(report.frame(diagnostics.cause_summary(
        t3, currency=root_config.DATA.notional_currency)))
    conf = diagnostics.cause_confusion(t3)
    if len(conf):
        print("\n  attributed vs KNOWN cause (synthetic only):")
        print(report.frame(conf))
        acc = diagnostics.cause_accuracy(t3)
        print("\n  attribution accuracy:")
        print(report.frame(acc))
        hit, tot = acc["attributed_correctly"].sum(), acc["flagged"].sum()
        print(f"\n  {hit}/{tot} flagged true failures got the right diagnosis "
              f"({100*hit/tot:.0f}%).")
        print("  That is the deliverable: not 'this order was bad', but which of")
        print("  four different problems it was, each with a different remedy.")

    for name, scored, _ in methods:
        folder = {"Tier 3 model": "tier3", "Tier 5 gaussian": "tier5"}[name]
        scored.to_csv(dataset.out_path(folder, "scored_orders.csv"), index=False)
    print("\nWrote outputs/tier{3,5}/scored_orders.csv")


if __name__ == "__main__":
    main()
