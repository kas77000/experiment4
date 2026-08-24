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
from tier5 import band, cells, config as t5cfg, curve, normality, persist


def fit_frame(df, cfg, *, bands_dir: str, out_dir: str, source_csv: str,
              force: bool = False) -> list[dict]:
    """Fit and freeze every cell in `df`. Returns one result dict per cell."""
    if cfg.metric not in df.columns:
        raise ValueError(f"Tier 5 needs column {cfg.metric!r}, which is absent.")

    results = []
    for region, strategy, g in cells.cells(df):
        period = cells.period_label(g) or "unknown-period"
        x = g[cfg.metric].to_numpy()
        est = band.estimates(x, cfg.k_sigma)
        e = cfg.estimator
        lo, hi = est[f"lo_{e}"], est[f"hi_{e}"]

        row = {"region": region, "strategy": strategy, "n": int(est["n"]),
               "period": period,
               "centre": est[f"centre_{e}"], "scale": est[f"scale_{e}"],
               "lo": lo, "hi": hi, "flag_rate_pct": float("nan"),
               "band_path": None, "skipped": False, "reason": "",
               "curve_msg": ""}

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

        finite = x[np.isfinite(x)]
        flag_rate = (100.0 * float(np.mean((finite < lo) | (finite > hi)))
                     if finite.size else float("nan"))
        row["flag_rate_pct"] = flag_rate

        path = cells.band_path(bands_dir, region, strategy)
        persist.save(est, cfg, path, region=region, strategy=strategy,
                     source_csv=source_csv, period=period, df=g,
                     flag_rate_pct=flag_rate)
        row["band_path"] = path

        cell_out = cells.out_dir(out_dir, "fit", period, region, strategy)
        os.makedirs(cell_out, exist_ok=True)
        normality.evidence(g, cfg).to_csv(
            os.path.join(cell_out, "normality.csv"), index=False)
        row["curve_msg"] = curve.plot(
            x, centre=row["centre"], scale=row["scale"], lo=lo, hi=hi,
            path=os.path.join(cell_out, "curve.png"),
            title=f"{region} / {strategy}  --  {cfg.metric}",
            subtitle=f"fitted on {period}", k=cfg.k_sigma,
            units=t5cfg.units_of(cfg.metric))

        results.append(row)
    return results


def main():
    ap = dataset.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--metric", choices=[schema.SLIPPAGE_BPS,
                                         schema.PERF_IN_SPREADS,
                                         schema.PERF_NORM],
                    help="Override the banded metric from tier5/config.py.")
    ap.add_argument("--k", type=float, help="Scales either side of the centre.")
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
        overrides["k_sigma"] = args.k
    if args.estimator:
        overrides["estimator"] = args.estimator
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    df, clean_report = dataset.load_prepared(args)

    print(report.header("TIER 5 --- FIT AND FREEZE"))
    print("\n=== Cleaning ===")
    print(clean_report.as_text())
    units = t5cfg.units_of(cfg.metric)
    print(f"\n  metric={cfg.metric} ({units})  k={cfg.k_sigma:g}"
          f"  estimator={cfg.estimator}  min_group_n={cfg.min_group_n}")

    results = fit_frame(df, cfg, bands_dir=args.bands_dir,
                        out_dir=args.out_dir,
                        source_csv=args.csv or "synthetic",
                        force=args.force)

    # Which column each strategy's band was built from. Stated out loud because
    # choosing the wrong one is invisible in the output: the band still fits and
    # the curve still looks like a curve, it is just the wrong benchmark.
    lines = config.metric_source_lines(r["strategy"] for r in results)
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
        print(f"    in-sample flagged: {r['flag_rate_pct']:.2f}%")
        if r["curve_msg"]:
            print(r["curve_msg"])

    n_ok = sum(1 for r in results if not r["skipped"])
    n_skip = len(results) - n_ok
    print(f"\nFroze {n_ok} band(s) to {args.bands_dir}/"
          + (f", skipped {n_skip}." if n_skip else "."))
    print("  These rates are IN-SAMPLE. Run tier5.score on a later period for a")
    print("  number that measures rather than defines.")

    unknown = sorted({r["region"] for r in results
                      if r["region"] not in config.REGION_NAMES})
    if unknown:
        print(f"\n  Unrecognised region code(s): {unknown}. They were fitted "
              f"normally -- check the Sym suffix if that is unexpected.")


if __name__ == "__main__":
    main()
