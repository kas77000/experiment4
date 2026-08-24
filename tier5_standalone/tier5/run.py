"""Tier 5 driver:  python -m tier5.run  [--csv x.csv]

    --metric        perf_in_spreads (default, in spreads) | slippage_bps | perf_norm
    --k             scales either side of the centre (default 3.0)
    --estimator     classical (default) | robust
    --score-level   ALL (default) | algo | adv_bucket | algo_x_adv_bucket
    --self-check    prove the implementation on data with a known answer

Outputs: outputs/tier5/{band_table,scored_orders,normality}.csv, qq_plot.png
"""

from __future__ import annotations
import argparse
import dataclasses

import numpy as np

from tca import dataset, report, schema
from tier5 import band, config as t5cfg, normality


def self_check(n: int = 200_000, mu: float = -8.7, sd: float = 18.4,
               k: float = 3.0, seed: int = 11) -> int:
    """Run the method on data whose answer is known in closed form.

    On genuinely normal data the estimators must recover the parameters, the
    band must match the closed form, the delivered flag rate must be 0.27%,
    and the classical and robust scales must agree. If all four hold, any
    deviation on a real book is the DATA and not a bug -- which is the claim
    the whole report rests on.
    """
    x = np.random.default_rng(seed).normal(mu, sd, n)
    e = band.estimates(x, k)
    lo, hi = e["lo_classical"], e["hi_classical"]
    outside = float(np.mean((x < lo) | (x > hi)))
    theory = 1.0 - normality.promised_inside(k)
    ratio = e["scale_robust"] / e["scale_classical"]

    checks = [
        ("recovered mean", e["centre_classical"], mu, 0.20,
         f"{e['centre_classical']:.3f}  (true {mu})"),
        ("recovered sd", e["scale_classical"], sd, 0.20,
         f"{e['scale_classical']:.3f}  (true {sd})"),
        ("band lo", lo, mu - k * sd, 0.80, f"{lo:.2f}  (closed form {mu - k*sd:.2f})"),
        ("band hi", hi, mu + k * sd, 0.80, f"{hi:.2f}  (closed form {mu + k*sd:.2f})"),
        ("flag rate", outside, theory, 0.0006,
         f"{100*outside:.3f}%  (theory {100*theory:.3f}%)"),
        ("robust/classical", ratio, 1.0, 0.02, f"{ratio:.3f}  (1.000 on normal data)"),
    ]

    print(f"  drew {n:,} samples from N({mu}, {sd}), k = {k}")
    ok = True
    for name, got, want, tol, shown in checks:
        passed = abs(got - want) <= tol
        ok = ok and passed
        print(f"  {name:<18} {shown:<40} {'OK' if passed else 'FAIL'}")
    print(f"\n  {'ALL CHECKS PASSED' if ok else 'SELF-CHECK FAILED'}")
    return 0 if ok else 1


def main():
    ap = dataset.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--metric", choices=[schema.SLIPPAGE_BPS,
                                         schema.PERF_IN_SPREADS,
                                         schema.PERF_NORM],
                    help="Override the banded metric from tier5/config.py.")
    ap.add_argument("--k", type=float, help="Scales either side of the centre.")
    ap.add_argument("--estimator", choices=list(t5cfg.ESTIMATORS),
                    help="Which estimator cuts the zones.")
    ap.add_argument("--score-level", choices=list(t5cfg.LEVEL_KEYS),
                    help="Which fitted level supplies each order's band.")
    ap.add_argument("--self-check", action="store_true",
                    help="Prove the implementation on data with a known answer.")
    args = ap.parse_args()

    if args.self_check:
        print(report.header("TIER 5 --- SELF-CHECK ON NORMAL DATA"))
        raise SystemExit(self_check())

    cfg = t5cfg.CONFIG
    overrides = {}
    if args.metric:
        overrides["metric"] = args.metric
    if args.k is not None:
        overrides["k_sigma"] = args.k
    if args.estimator:
        overrides["estimator"] = args.estimator
    if args.score_level:
        overrides["score_level"] = args.score_level
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    df, clean_report = dataset.load_prepared(args)

    print(report.header("TIER 5 --- GAUSSIAN mu +/- k*sigma BAND"))
    print("\n=== Cleaning ===")
    print(clean_report.as_text())

    print(f"\n=== Band ===\n  metric={cfg.metric}  k={cfg.k_sigma:g}"
          f"  estimator={cfg.estimator}  score_level={cfg.score_level}"
          f"  min_group_n={cfg.min_group_n}")

    table = band.fit(df, cfg)
    headline = table[table["level"] == t5cfg.LEVEL_ALL].iloc[0]
    c = headline[f"centre_{cfg.estimator}"]
    s = headline[f"scale_{cfg.estimator}"]
    lo = headline[f"lo_{cfg.estimator}"]
    hi = headline[f"hi_{cfg.estimator}"]
    print(f"\n  All orders (n = {int(headline['n']):,})")
    print(f"    centre  {c:>9.2f}")
    print(f"    scale   {s:>9.2f}")
    print(f"    RANGE   {lo:>9.2f} .. {hi:.2f}")

    print("\n=== Band table ===")
    show = table.copy()
    for col in show.columns:
        if show[col].dtype.kind == "f":
            show[col] = show[col].round(2)
    print(report.frame(show, max_rows=40))

    n_untrusted = int((~table["trusted"]).sum())
    if n_untrusted:
        print(f"\n  {n_untrusted} group(s) below min_group_n={cfg.min_group_n}"
              f" -> a sigma from that many orders is not a threshold. Those"
              f" orders fall back to the global band.")

    model = band.BandModel(table, cfg)
    scored = model.score_frame(df)

    print("\n=== Zone distribution ===")
    print(report.zone_summary(scored))

    print("\n=== Flag rate vs order difficulty ===")
    print(report.frame(band.flag_rate_by_bucket(scored)))
    if cfg.score_level == t5cfg.LEVEL_ALL:
        print("\n  One band for the whole book, so this column is expected to")
        print("  slope: it does not adjust for difficulty. --score-level"
              " algo_x_adv_bucket flattens it.")

    # ---------------- the evidence ----------------
    x = df[cfg.metric].to_numpy()
    print(report.header("DOES k*sigma MEAN WHAT IT SAYS ON THIS BOOK?"))
    cov = normality.coverage_table(x, c, s)
    print(report.frame(cov.round(3)))

    req = normality.required_k(x, c, s)
    print(f"\n  To actually flag {100*normality.NOMINAL_OUTSIDE:.2f}% of this book"
          f" you need k = {req['k_symmetric']:.2f}, not {cfg.k_sigma:g}.")
    print(f"  Per tail: k_lo = {req['k_lo']:.2f}, k_hi = {req['k_hi']:.2f}."
          f"  A symmetric band cannot serve both when these differ.")

    st = normality.shape_stats(x)
    print(f"\n  skew             {st['skew']:>8.3f}   (0 if normal)")
    print(f"  excess kurtosis  {st['excess_kurtosis']:>8.3f}   (0 if normal)")
    if np.isfinite(st["p_value"]):
        print(f"  D'Agostino K2    {st['dagostino_k2']:>8.1f}   p = {st['p_value']:.3g}")
        print("\n  Treat that p-value with suspicion: at this sample size EVERY")
        print("  formal normality test rejects, because it tests 'exactly normal'")
        print("  and nothing real is. The coverage table above is the evidence.")
    elif st["test_note"]:
        print(f"  {st['test_note']}")

    print("\n=== Classical vs robust ===")
    est_rows = table[table["level"] == t5cfg.LEVEL_ALL][[
        "centre_classical", "scale_classical", "lo_classical", "hi_classical",
        "centre_robust", "scale_robust", "lo_robust", "hi_robust"]].round(2)
    print(report.frame(est_rows))
    sc, sr = headline["scale_classical"], headline["scale_robust"]
    if np.isfinite(sc) and np.isfinite(sr) and sr > 0:
        print(f"\n  sd is {sc/sr:.2f}x the robust scale. On normal data that ratio")
        print("  is 1.00 -- anything above it is the standard deviation being")
        print("  inflated by the very outliers the band exists to catch.")

    if cfg.make_qq_plot:
        print("\n=== QQ plot ===")
        print(normality.qq_plot(x, dataset.out_path("tier5", "qq_plot.png"),
                                f"Tier 5 --- {cfg.metric} vs normal"))

    print("\n  NOTE: these flags are IN-SAMPLE -- the band was fitted on the")
    print("  same orders being scored, so the flag rate is partly circular.")
    print("  Use `tier5.fit` then `tier5.score` to get an out-of-sample number.")

    # ---------------- outputs ----------------
    table.to_csv(dataset.out_path("tier5", "band_table.csv"), index=False)
    normality.evidence(df, cfg).to_csv(
        dataset.out_path("tier5", "normality.csv"), index=False)
    # cfg.metric is often SLIPPAGE_BPS, which is already in the list, so
    # dedupe -- selecting a duplicated column name raises in pandas.
    wanted = [schema.ORDER_ID, schema.ALGO, schema.ADV_BUCKET,
              schema.SPREAD_BPS, schema.SLIPPAGE_BPS, cfg.metric,
              "band_centre", "band_scale", "band_lo", "band_hi",
              "band_level", "zone", "rank_stat", "flagged", "review_required"]
    cols, seen = [], set()
    for col in wanted:
        if col in scored.columns and col not in seen:
            cols.append(col)
            seen.add(col)
    scored[cols].to_csv(dataset.out_path("tier5", "scored_orders.csv"), index=False)
    print("\nWrote outputs/tier5/ -> band_table.csv, scored_orders.csv,"
          " normality.csv")


if __name__ == "__main__":
    main()
