"""Fit a Gaussian band per (region, strategy) and freeze it.

    python -m tier5.fit --csv extracts/year.csv

Region comes from the Sym suffix, strategy from the Strategy column and the
period from the Date column, so the same command works whether the twelve cells
arrive as one file or twelve.

This is the only module that computes a centre or a scale. Everything the
scoring side needs travels in the band file.
"""

from __future__ import annotations

import argparse
import dataclasses
import os

import numpy as np

import config
from tca import dataset, report, schema
from tier5 import (band, budget, cells, compat, config as t5cfg, curve,
                   normality, persist)


def active_rule(cfg) -> str:
    """Which rule sets the bounds. The ONLY place precedence is decided.

    PRECEDENCE, most specific first:
        review_count  -- a workload, in orders per cell per month   (explicit)
        flag_rate     -- a two-sided coverage target, k solved      (explicit)
        absolute      -- an absolute band in metric units           (shipped)
        percentile    -- per side, the wider of mean +/- k*sigma and P(pct)
        fixed         -- a plain fixed multiple

    EXPLICIT OVERRIDES COME BEFORE THE SHIPPED DEFAULTS ON PURPOSE. Three
    separate times a new shipped default silently made a flag inert, because
    the default sat earlier in the chain and the CLI had to remember to clear
    it. Ordering the explicit-only options first makes that structurally
    impossible rather than something to remember.

    And it is a FUNCTION because the chain was written out twice -- once to
    choose the band and once to describe it -- which promptly drifted, so a run
    calibrated to a flag rate printed the absolute band's explanation. Two
    copies of a precedence rule is one copy too many.
    """
    if getattr(cfg, "target_review_count", None) is not None:
        return "review_count"
    if getattr(cfg, "target_flag_rate", None) is not None:
        return "flag_rate"
    if getattr(cfg, "band_abs", None) is not None:
        return "absolute"
    if getattr(cfg, "band_percentile", None) is not None:
        return "percentile"
    if getattr(cfg, "k_sigma", None) is None:
        # No rule left and no multiple to fall back on.
        return "percentile"
    return "fixed"


def fit_frame(df, cfg, *, bands_dir: str, out_dir: str, source_csv: str,
              force: bool = False) -> list[dict]:
    """Fit and freeze every cell in `df`. Returns one result dict per cell."""
    if cfg.metric not in df.columns:
        raise ValueError(f"Tier 5 needs column {cfg.metric!r}, which is absent.")
    if cfg.target_flag_rate is not None and not (0.0 < cfg.target_flag_rate < 100.0):
        raise ValueError(
            f"target_flag_rate must be a percentage strictly between 0 and 100, "
            f"got {cfg.target_flag_rate!r}. It is the share of the fit book you "
            f"are willing to review.")
    if cfg.target_review_count is not None and cfg.target_review_count <= 0:
        raise ValueError(
            f"target_review_count must be a positive number of orders per cell "
            f"per month, got {cfg.target_review_count!r}.")

    results = []
    for region, strategy, g in cells.cells(df):
        period = cells.period_label(g) or "unknown-period"
        x = g[cfg.metric].to_numpy()
        est = band.estimates(x, cfg.k_sigma or 0.0)
        e = cfg.estimator
        lo, hi = est[f"lo_{e}"], est[f"hi_{e}"]

        row = {"region": region, "strategy": strategy, "n": int(est["n"]),
               "period": period,
               "centre": est[f"centre_{e}"], "scale": est[f"scale_{e}"],
               "lo": lo, "hi": hi, "k_used": cfg.k_sigma,
               "flag_rate_pct": float("nan"),
               "band_path": None, "skipped": False, "reason": "",
               "curve_msg": "", "budget": None,
               "k_from_coverage": float("nan"), "k_floored": False,
               "rule": None, "band_abs": None,
               "abs_k_lo": float("nan"), "abs_k_hi": float("nan")}

        if est["n"] == 0:
            # Distinct from "too few orders": the cell has rows, they just
            # carry no metric. Almost always the strategy's source column was
            # left out of the export -- see config.METRIC_BY_STRATEGY.
            row["skipped"] = True
            row["reason"] = (f"{len(g):,} orders but no usable {cfg.metric!r} "
                             f"value in any of them -- check that this "
                             f"strategy's source column is in the extract.")
            results.append(row)
            continue

        if est["n"] < cfg.min_group_n and not force:
            row["skipped"] = True
            row["reason"] = (f"n={est['n']} is below min_group_n="
                             f"{cfg.min_group_n}; a sigma from that many orders "
                             f"is not a threshold. Use --force to override.")
            results.append(row)
            continue

        cell_cfg = cfg
        rule_name = active_rule(cfg)
        if rule_name == "review_count":
            d_lo, d_hi = cells.date_range(g)
            sol = budget.solve(x, est[f"centre_{e}"], est[f"scale_{e}"],
                               per_month=cfg.target_review_count,
                               months=budget.window_months(d_lo, d_hi),
                               k_floor=t5cfg.BUDGET_K_FLOOR)
            row["budget"] = sol
            cell_cfg = dataclasses.replace(cfg, k_sigma=float(sol["k"]))
            est = band.estimates(x, float(sol["k"]))
            lo, hi = est[f"lo_{e}"], est[f"hi_{e}"]
            row["lo"], row["hi"], row["k_used"] = lo, hi, float(sol["k"])
        elif rule_name == "flag_rate":
            k_cov = normality.required_k(
                x, est[f"centre_{e}"], est[f"scale_{e}"],
                target_outside=cfg.target_flag_rate / 100.0)["k_symmetric"]
            if np.isfinite(k_cov) and k_cov > 0:
                # No floor here. An explicit coverage target is an
                # instruction, and flooring it at the shipped band width would
                # make --target-flag-rate silently inert the moment K_SIGMA
                # moved above the k that target implies.
                row["k_from_coverage"] = float(k_cov)
                k = float(k_cov)
                row["k_floored"] = False
                cell_cfg = dataclasses.replace(cfg, k_sigma=float(k))
                est = band.estimates(x, float(k))
                lo, hi = est[f"lo_{e}"], est[f"hi_{e}"]
                row["lo"], row["hi"], row["k_used"] = lo, hi, float(k)
        elif rule_name == "absolute":
            # An absolute band: stated, not fitted. centre and scale are still
            # computed -- scoring needs the scale to rank how far outside an
            # order landed -- but they do not set the bounds.
            a = float(cfg.band_abs)
            lo, hi = -a, a
            est = dict(est)
            for _e in ("classical", "robust"):
                est[f"lo_{_e}"], est[f"hi_{_e}"] = -a, a
            row["lo"], row["hi"], row["band_abs"] = lo, hi, a
            sc = est[f"scale_{e}"]
            if np.isfinite(sc) and sc > 0:
                row["abs_k_hi"] = (a - est[f"centre_{e}"]) / sc
                row["abs_k_lo"] = (est[f"centre_{e}"] + a) / sc
        elif rule_name == "percentile":
            # The fitted rule: per side, the wider of the sigma term and the
            # empirical percentile. k is NOT solved -- the sigma term is
            # literally centre + k*scale.
            est, detail = band.apply_rule(est, x, k=cfg.k_sigma,
                                          percentile=cfg.band_percentile)
            lo, hi = est[f"lo_{e}"], est[f"hi_{e}"]
            row["lo"], row["hi"] = lo, hi
            row["rule"] = detail[e]

        finite = x[np.isfinite(x)]
        flag_rate = (100.0 * float(np.mean((finite < lo) | (finite > hi)))
                     if finite.size else float("nan"))
        row["flag_rate_pct"] = flag_rate

        path = cells.band_path(bands_dir, region, strategy)
        persist.save(est, cell_cfg, path, region=region, strategy=strategy,
                     source_csv=source_csv, period=period, df=g,
                     flag_rate_pct=flag_rate, budget=row["budget"],
                     k_floor=(cfg.k_sigma
                              if cfg.target_flag_rate is not None else None),
                     k_from_coverage=row["k_from_coverage"],
                     k_floored=row["k_floored"], rule=row["rule"],
                     band_abs=row["band_abs"])
        row["band_path"] = path

        cell_out = cells.out_dir(out_dir, "fit", period, region, strategy)
        os.makedirs(cell_out, exist_ok=True)
        normality.evidence(g, cell_cfg).to_csv(
            os.path.join(cell_out, "normality.csv"), index=False)
        row["curve_msg"] = curve.plot(
            x, centre=row["centre"], scale=row["scale"], lo=lo, hi=hi,
            path=os.path.join(cell_out, "curve.png"),
            title=f"{region} / {strategy}  --  {cfg.metric}",
            subtitle=f"fitted on {period}",
            units=t5cfg.units_of(cfg.metric))

        results.append(row)
    return results


def main():
    # Before argparse, before the CSV is read: a half-copied folder must
    # fail in under a second, not after loading 47k rows.
    compat.check_environment()
    ap = dataset.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--metric", choices=[schema.SLIPPAGE_BPS,
                                         schema.PERF_IN_SPREADS,
                                         schema.PERF_NORM],
                    help="Override the banded metric from tier5/config.py.")
    ap.add_argument("--k", type=float, help="Scales either side of the centre.")
    ap.add_argument("--target-flag-rate", type=float, default=None,
                    metavar="PCT",
                    help="Percentage of the fit book to leave outside the band. "
                         "Overrides --k: each cell solves for the k that "
                         "delivers this. Use when k=3 flags more than the desk "
                         "can review, which on a fat-tailed book it will.")
    ap.add_argument("--target-review-count", type=float, default=None,
                    metavar="N",
                    help="Orders per cell per MONTH you are willing to "
                         "explain. Overrides --k: each cell solves for the k "
                         "that delivers this load from its own volume. Use "
                         "this when the answer is 'a couple a month' rather "
                         "than a percentage. --k becomes a floor.")
    ap.add_argument("--percentile", type=float, default=None, metavar="PCT",
                    help="Cut the band from percentiles ALONE: P(PCT) and "
                         "P(100-PCT), with no sigma term and no absolute "
                         "bound. The only rule here that assumes nothing about "
                         "the distribution's shape.")
    ap.add_argument("--estimator", choices=list(t5cfg.ESTIMATORS),
                    help="Which estimator cuts the band.")
    ap.add_argument("--bands-dir", default="bands",
                    help="Where band JSON files are written.")
    ap.add_argument("--out-dir", default="outputs",
                    help="Where curves and evidence are written.")
    ap.add_argument("--force", action="store_true",
                    help="Fit cells below min_group_n anyway.")
    args = ap.parse_args()

    cfg = t5cfg.CONFIG
    overrides = {}
    if args.metric:
        overrides["metric"] = args.metric
    if args.k is not None:
        # An explicit --k means "this multiple, fixed" -- so it must switch the
        # coverage standard OFF, not silently lose to it. Without this line the
        # default in config.py would solve for k anyway and --k would be inert,
        # which is the worst kind of flag: one that appears to work.
        overrides["k_sigma"] = args.k
        overrides["target_flag_rate"] = None
        overrides["band_percentile"] = None
        overrides["band_abs"] = None
    if args.estimator:
        overrides["estimator"] = args.estimator
    if args.target_flag_rate is not None:
        overrides["target_flag_rate"] = args.target_flag_rate
    if args.target_review_count is not None:
        # Outranks the shipped rules by position in active_rule(), so unlike
        # --k it has nothing to clear.
        overrides["target_review_count"] = args.target_review_count
    if args.percentile is not None:
        # Percentiles alone: every other rule has to be switched off, since
        # each would otherwise outrank this one or widen it.
        overrides["band_percentile"] = args.percentile
        overrides["k_sigma"] = None
        overrides["band_abs"] = None
        overrides["target_flag_rate"] = None
        overrides["target_review_count"] = None
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    df, clean_report = dataset.load_prepared(args)
    compat.check_report(clean_report)

    print(report.header("TIER 5 --- FIT AND FREEZE"))
    print("\n=== Cleaning ===")
    print(clean_report.as_text())
    units = t5cfg.units_of(cfg.metric)
    rule_name = active_rule(cfg)
    if rule_name == "absolute":
        k_desc = f"band = {-cfg.band_abs:g} .. {cfg.band_abs:g} {units} (absolute)"
    elif rule_name == "review_count":
        k_desc = (f"k=solved per cell for {cfg.target_review_count:g} "
                  f"order(s)/month  (floor k={t5cfg.BUDGET_K_FLOOR:g})")
        if cfg.target_flag_rate is not None:
            print(f"\n  NOTE: a review count was given, so it overrides the "
                  f"{100.0 - cfg.target_flag_rate:g}% coverage")
            print("  standard in tier5/config.py for this run.")
    elif rule_name == "percentile":
        pc = cfg.band_percentile
        k_desc = (f"band = P{pc:g} / P{100 - pc:g}  (percentiles only)"
                  if cfg.k_sigma is None else
                  f"band = MAX(mean +/- {cfg.k_sigma:g}*sigma, "
                  f"P{pc:g}/P{100 - pc:g})")
    elif rule_name == "flag_rate":
        k_desc = (f"k=solved per cell for "
                  f"{100.0 - cfg.target_flag_rate:g}% coverage")
    else:
        k_desc = f"k={cfg.k_sigma:g}"
    print(f"\n  metric={cfg.metric} ({units})  {k_desc}"
          f"  estimator={cfg.estimator}  min_group_n={cfg.min_group_n}")

    results = fit_frame(df, cfg, bands_dir=args.bands_dir,
                        out_dir=args.out_dir,
                        source_csv=args.csv or "synthetic",
                        force=args.force)

    # Which column each strategy's band was built from. Stated out loud because
    # choosing the wrong one is invisible in the output: the band still fits and
    # the curve still looks like a curve, it is just the wrong benchmark.
    lines = config.metric_source_lines(
        (r["strategy"] for r in results),
        supplied=clean_report.metric_supplied)
    if lines:
        print("\n=== Metric source ===")
        print("\n".join(lines))

    for r in results:
        print(f"\n  {r['region']} / {r['strategy']}   n = {r['n']:,}   {r['period']}")
        if r["skipped"]:
            print(f"    SKIPPED: {r['reason']}")
            continue
        print(f"    centre  {r['centre']:>9.2f}")
        print(f"    scale   {r['scale']:>9.2f}")
        print(f"    RANGE   {r['lo']:>9.2f} .. {r['hi']:.2f} {units}"
              f"      <- frozen to {r['band_path']}")
        if cfg.target_review_count is not None:
            b = r["budget"] or {}
            print(f"    k       {r['k_used']:>9.2f}"
                  + ("      <- HELD AT THE FLOOR" if b.get("floored")
                     else f"      <- what {cfg.target_review_count:g}/month "
                          f"cost on this cell"))
            miss = budget.miss_to_flag(r["lo"], r["hi"], r["centre"])
            print(f"    an order must miss by {miss:.1f} {units} to be flagged")
            if b.get("floored"):
                print(f"    NOTE: {b.get('reason', '')}")
            else:
                print(f"    cut on the {b.get('n_tail', 0)} most extreme "
                      f"order(s) of the fit book")
                if b.get("thin_tail"):
                    print(f"    THIN TAIL: {b.get('n_tail', 0)} order(s) is "
                          f"few to place a bound on. Fit a longer window, or "
                          f"raise the budget.")
        elif r["band_abs"] is not None:
            print(f"    BAND IS STATED, not fitted: "
                  f"{-r['band_abs']:g} .. {r['band_abs']:g} {units}")
            if np.isfinite(r["abs_k_hi"]):
                print(f"    on this book that is {r['abs_k_lo']:.1f} sigma low "
                      f"/ {r['abs_k_hi']:.1f} sigma high"
                      f"   (sigma = {r['scale']:.2f} {units})")
        elif r["rule"] is not None:
            d = r["rule"]
            ks, pc = cfg.k_sigma, cfg.band_percentile
            if ks is None:
                print(f"    P{pc:g} {d['hi']:>9.2f}      "
                      f"P{100 - pc:g} {d['lo']:>9.2f}"
                      f"      (no sigma term; centre and sigma above are "
                      f"context only)")
                print(f"    an order must miss by "
                      f"{budget.miss_to_flag(r['lo'], r['hi'], r['centre']):.1f} "
                      f"{units} to be flagged")
                continue
            print(f"    mean + {ks:g}*sigma  {d['hi_sigma']:>9.2f}"
                  f"      P{pc:g}   {d['hi_pct']:>9.2f}"
                  f"   ->  hi  {d['hi']:>8.2f}  ({d['hi_binds']})")
            print(f"    mean - {ks:g}*sigma  {d['lo_sigma']:>9.2f}"
                  f"      P{100 - pc:g}    {d['lo_pct']:>9.2f}"
                  f"   ->  lo  {d['lo']:>8.2f}  ({d['lo_binds']})")
            print(f"    an order must miss by "
                  f"{budget.miss_to_flag(r['lo'], r['hi'], r['centre']):.1f} "
                  f"{units} to be flagged")
        elif cfg.target_flag_rate is not None:
            cov = 100.0 - cfg.target_flag_rate
            kn = normality.k_if_normal(cov)
            if r["k_floored"]:
                print(f"    k       {r['k_used']:>9.2f}"
                      f"      <- HELD AT THE {cfg.k_sigma:g} FLOOR"
                      f"  ({cov:g}% needed only {r['k_from_coverage']:.2f})")
            else:
                print(f"    k       {r['k_used']:>9.2f}"
                      f"      <- what {cov:g}% coverage cost here"
                      f"  ({kn:.2f} if normal, floor {cfg.k_sigma:g})")
            print(f"    an order must miss by "
                  f"{budget.miss_to_flag(r['lo'], r['hi'], r['centre']):.1f} "
                  f"{units} to be flagged")
        print(f"    in-sample flagged: {r['flag_rate_pct']:.2f}%")
        if r["curve_msg"]:
            print(r["curve_msg"])

    n_ok = sum(1 for r in results if not r["skipped"])
    n_skip = len(results) - n_ok
    print(f"\nFroze {n_ok} band(s) to {args.bands_dir}/"
          + (f", skipped {n_skip}." if n_skip else "."))
    if rule_name == "absolute":
        a = cfg.band_abs
        print(f"\n  The band is STATED: {-a:g} .. {a:g} {units}, set in "
              f"tier5/config.py")
        print("  (BAND_ABS_SPREADS). Nothing about it was estimated from this")
        print("  book, which is the point: a fitted band widens when execution")
        print("  gets worse, forgiving exactly the drift it exists to catch.")
        print("\n  The centre and sigma above ARE measured, and the sigma")
        print("  equivalent printed beside each band says how loose or tight this")
        print("  policy is on that particular cell. Watch it across refits: if it")
        print("  falls, the book is deteriorating underneath a fixed threshold.")
    elif rule_name == "review_count":
        n = cfg.target_review_count
        print(f"\n  k was calibrated to {n:g} order(s) per cell per month, so the "
              f"in-sample")
        print("  rates above hold BY CONSTRUCTION and measure nothing. That is the")
        print("  point: a review load is a staffing decision, not a finding. The")
        print("  number that carries information is what tier5.score reports on a")
        print(f"  period the band has never seen -- a month with {3 * n:.0f}+ is a "
              f"real")
        print("  change, and a run of empty months means the budget was set too")
        print("  tight to detect anything.")
    elif rule_name == "percentile":
        ks, pc = cfg.k_sigma, cfg.band_percentile
        if ks is None:
            print(f"\n  Rule: hi = P{pc:g},  lo = P{100 - pc:g}  -- percentiles "
                  f"alone, no sigma term.")
            print("  Set in tier5/config.py (K_SIGMA = None, PERCENTILE_PCT).")
            print("  This assumes nothing about the shape of the distribution:")
            print("  no centre, no scale, no implied symmetry. The cost is that")
            print(f"  each bound rests on the {100 - pc:g}% of orders beyond it,")
            print("  so it moves with the book -- including when the book gets")
            print("  worse, which is the one time a threshold should not move.")
        else:
            print(f"\n  Rule: hi = MAX(mean + {ks:g}*sigma, P{pc:g}),  "
                  f"lo = MIN(mean - {ks:g}*sigma, P{100 - pc:g})")
            print("  set once in tier5/config.py (K_SIGMA, PERCENTILE_PCT). Both")
            print("  candidates and the winner are printed per side above,")
            print("  because a percentile that never binds is worth knowing")
            print("  about: the sigma term is doing the work and the net is")
            print("  standing by unused.")
        print("\n  These rates are IN-SAMPLE. Run tier5.score on a later period")
        print("  for a number that measures rather than defines.")
    elif rule_name == "flag_rate":
        cov = 100.0 - cfg.target_flag_rate
        print(f"\n  Standard: {cov:g}% coverage, set once in tier5/config.py "
              f"(COVERAGE_PCT).")
        print("  k is an OUTPUT of that rule, not an input, so it differs between")
        print(f"  cells -- correctly: the promise is fixed at {cov:g}%, and the")
        print("  multiple is whatever each book's tail costs to keep it.")
        print(f"\n  The in-sample rates above equal {cfg.target_flag_rate:g}% BY "
              f"CONSTRUCTION and")
        print("  measure nothing. The number that carries information is what")
        print("  tier5.score reports on a period the band has never seen: if July")
        print(f"  flags well above {cfg.target_flag_rate:g}%, something actually "
              f"changed.")
    else:
        print("  These rates are IN-SAMPLE. Run tier5.score on a later period for a")
        print("  number that measures rather than defines.")

    unknown = sorted({r["region"] for r in results
                      if r["region"] not in config.REGION_NAMES})
    if unknown:
        print(f"\n  Unrecognised region code(s): {unknown}. They were fitted "
              f"normally -- check the Sym suffix if that is unexpected.")


if __name__ == "__main__":
    main()
