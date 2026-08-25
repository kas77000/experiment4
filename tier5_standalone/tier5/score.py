"""Apply frozen bands to a later period.

    python -m tier5.score --csv extracts/july.csv --label 2026-07

This module NEVER fits. It reads lo/hi out of a band file and classifies
against them unchanged, which is what turns the flag rate from a definition
into a measurement. If a cell has no band file it is skipped and reported --
scoring HK VWAP orders against the Japan TWAP band would produce
plausible-looking numbers with no warning, which is worse than no answer.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import config
from tca import dataset, report, schema
from tier5 import band, cells, compat, config as t5cfg, curve, persist


class LeakageError(RuntimeError):
    """The scoring window overlaps the window the band was fitted on."""


# Written to outliers.csv when present in the extract. These are the inputs to
# the analyst's explanation -- this module deliberately does not attribute
# cause. Automated attribution is a cost-model job and would defeat the point
# of a standalone folder.
DIAGNOSTIC_COLS = [schema.NOTIONAL, schema.SPREAD_BPS, schema.PCT_ADV,
                   schema.PARTICIPATION, schema.DURATION_MIN,
                   schema.PASSIVE_FILL_PCT, schema.AUCTION_PCT,
                   schema.REVERSION_BPS, schema.MOMENTUM_BPS]

IDENT_COLS = [schema.ORDER_ID, schema.SYMBOL, schema.SIDE, schema.ORDER_DATE]


def n_sigma_outside(x, lo: float, hi: float, scale: float):
    """How far outside the band, in scales. Always >= 0; exactly 0 inside.

    Positive on both sides so a single column ranks the queue regardless of
    which tail an order broke, and comparable across cells because every cell
    is divided by its own scale.
    """
    x = np.asarray(x, dtype=float)
    return np.maximum(0.0, np.maximum((x - hi) / scale, (lo - x) / scale))


def score_frame(df, base_cfg, *, bands_dir: str, out_dir: str,
                label: str | None = None,
                min_notional_review: float | None = None) -> list[dict]:
    """Score every cell in `df` against its frozen band.

    `min_notional_review` overrides the materiality gate frozen with the band.
    The band is a measurement and must not move; how much of the queue the desk
    chooses to work is a policy, and policies change without invalidating the
    measurement. Left as None, the frozen value applies so a rescore of the
    same band reproduces the same queue.
    """
    results = []
    for region, strategy, g in cells.cells(df):
        period = label or cells.period_label(g) or "score"
        path = cells.band_path(bands_dir, region, strategy)
        row = {"region": region, "strategy": strategy, "n": int(len(g)),
               "period": period, "n_flagged": 0, "n_review": 0,
               "min_notional_review": 0.0,
               "flag_rate_pct": float("nan"), "fit_flag_rate_pct": float("nan"),
               "lo": float("nan"), "hi": float("nan"),
               "skipped": False, "reason": "", "out_dir": None,
               "drift_table": None, "drift_warnings": [], "curve_msg": ""}

        if not os.path.exists(path):
            row["skipped"] = True
            row["reason"] = (f"no band at {path} -- fit this cell first. "
                             f"Not scored against another cell's band.")
            results.append(row)
            continue

        frozen, cfg, reference = persist.load(path, base_cfg)

        # A band frozen on a different metric is in different units. Scoring it
        # alongside the others would put bps bounds and spread bounds in one
        # summary table, where nothing marks which row is which.
        if cfg.metric != base_cfg.metric:
            row["skipped"] = True
            row["reason"] = (
                f"band was frozen on {cfg.metric!r} "
                f"({t5cfg.units_of(cfg.metric)}) but this run bands "
                f"{base_cfg.metric!r} ({t5cfg.units_of(base_cfg.metric)}). "
                f"Those are different units -- refit this cell.")
            results.append(row)
            continue

        b_lo, b_hi = cells.date_range(g)
        f_lo = pd.Timestamp(frozen["fit_date_min"]) if frozen["fit_date_min"] else None
        f_hi = pd.Timestamp(frozen["fit_date_max"]) if frozen["fit_date_max"] else None
        if cells.windows_overlap(f_lo, f_hi, b_lo, b_hi):
            raise LeakageError(
                f"{region}/{strategy}: the scoring window "
                f"{b_lo.date()}..{b_hi.date()} overlaps the window the band was "
                f"fitted on ({f_lo.date()}..{f_hi.date()}). Scoring a period the "
                f"band already saw makes the flag rate circular, which is the "
                f"one thing this workflow exists to avoid.")

        if cfg.metric not in g.columns:
            row["skipped"] = True
            row["reason"] = f"extract has no column {cfg.metric!r}"
            results.append(row)
            continue

        lo, hi = float(frozen["lo"]), float(frozen["hi"])
        centre, scale = float(frozen["centre"]), float(frozen["scale"])
        x = pd.to_numeric(g[cfg.metric], errors="coerce").to_numpy(dtype=float)

        scored = g.copy()
        scored["zone"] = [band.classify(v, lo, hi) for v in x]
        scored["band_lo"] = lo
        scored["band_hi"] = hi
        scored["band_centre"] = centre
        scored["band_scale"] = scale
        scored["n_sigma_outside"] = n_sigma_outside(x, lo, hi, scale)
        scored["flagged"] = scored["zone"].isin(list(band.FLAGGED))

        # Materiality. Being outside the band and being worth an analyst's
        # hour are different questions: a 3-spread miss on a $40k order costs
        # almost nothing to fix. The gate shrinks the QUEUE without hiding
        # anything -- every flagged order still appears in outliers.csv, it
        # just carries review_required=False.
        gate = float(min_notional_review
                     if min_notional_review is not None
                     else (getattr(cfg, "min_notional_review", 0.0) or 0.0))
        if gate > 0 and schema.NOTIONAL in scored.columns:
            scored["material"] = (pd.to_numeric(scored[schema.NOTIONAL],
                                                errors="coerce")
                                  .fillna(np.inf) >= gate)
        else:
            scored["material"] = True
        scored["review_required"] = scored["flagged"] & scored["material"]

        row["lo"], row["hi"] = lo, hi
        row["n_flagged"] = int(scored["flagged"].sum())
        row["n_review"] = int(scored["review_required"].sum())
        row["min_notional_review"] = gate
        row["flag_rate_pct"] = 100.0 * float(scored["flagged"].mean())
        row["fit_flag_rate_pct"] = reference.get("flag_rate_pct", float("nan"))

        cell_out = cells.out_dir(out_dir, "score", period, region, strategy)
        os.makedirs(cell_out, exist_ok=True)
        row["out_dir"] = cell_out

        keep = [c for c in (IDENT_COLS
                            + ["zone", cfg.metric, schema.SLIPPAGE_BPS,
                               "band_lo", "band_hi", "n_sigma_outside",
                               "review_required"]
                            + DIAGNOSTIC_COLS) if c in scored.columns]
        keep = list(dict.fromkeys(keep))

        scored.to_csv(os.path.join(cell_out, "scored.csv"), index=False)
        (scored[scored["flagged"]][keep]
         .sort_values("n_sigma_outside", ascending=False)
         .to_csv(os.path.join(cell_out, "outliers.csv"), index=False))

        table, warnings = persist.drift_report(g, scored, reference, cfg)
        row["drift_table"], row["drift_warnings"] = table, warnings

        row["curve_msg"] = curve.plot(
            x, centre=centre, scale=scale, lo=lo, hi=hi,
            path=os.path.join(cell_out, "curve.png"),
            title=f"{region} / {strategy}  --  {period} vs frozen band",
            subtitle=f"band frozen on {frozen.get('fit_period')}"
                     f"  --  the dashed curve is that band, NOT a refit",
            k=float(frozen["k_sigma"]),
            normal_label="frozen band's normal",
            units=frozen.get("metric_units", ""))

        results.append(row)
    return results


def main():
    ap = dataset.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--bands-dir", default="bands",
                    help="Where to look up frozen bands.")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--label", default=None,
                    help="Name the period folder. Defaults to the Date range.")
    ap.add_argument("--min-notional-review", type=float, default=None,
                    metavar="AMOUNT",
                    help="Only queue flagged orders at least this large for "
                         "review. Overrides the gate frozen with the band; the "
                         "band itself does not move. Every flagged order still "
                         "appears in outliers.csv, marked review_required.")
    args = ap.parse_args()

    df, clean_report = dataset.load_prepared(args)
    compat.check_report(clean_report)

    print(report.header("TIER 5 --- SCORE AGAINST FROZEN BANDS"))
    print("\n=== Cleaning ===")
    print(clean_report.as_text())

    try:
        results = score_frame(df, t5cfg.CONFIG, bands_dir=args.bands_dir,
                              out_dir=args.out_dir, label=args.label,
                              min_notional_review=args.min_notional_review)
    except LeakageError as exc:
        # A traceback here would read as a crash. It is a refusal, and the
        # reason matters more than the stack.
        print("\n=== REFUSED: the scoring window overlaps the fit window ===")
        print(f"\n  {exc}")
        print("\n  Export a period the band has not already seen, or refit the "
              "band on\n  an earlier window.")
        raise SystemExit(2)

    units = t5cfg.units_of(t5cfg.CONFIG.metric)
    lines = config.metric_source_lines(
        (r["strategy"] for r in results),
        supplied=clean_report.metric_supplied)
    if lines:
        print("\n=== Metric source ===")
        print("\n".join(lines))

    for r in results:
        print(f"\n  {r['region']} / {r['strategy']}")
        if r["skipped"]:
            print(f"    SKIPPED: {r['reason']}")
            continue
        ratio = (r["flag_rate_pct"] / r["fit_flag_rate_pct"]
                 if r["fit_flag_rate_pct"] else float("nan"))
        print(f"    band        {r['lo']:.2f} .. {r['hi']:.2f} {units}  (frozen)")
        print(f"    fit book:   {r['fit_flag_rate_pct']:.2f}% outside")
        print(f"    {r['period']}:    {r['n']:,} orders, {r['n_flagged']} flagged, "
              f"{r['flag_rate_pct']:.2f}% outside"
              + (f"   <- {ratio:.1f}x" if np.isfinite(ratio) else ""))
        if r["min_notional_review"] > 0:
            print(f"    to review:  {r['n_review']} of those clear the "
                  f"{r['min_notional_review']:,.0f} materiality gate")
        print(f"    wrote {r['out_dir']}")
        if r["drift_warnings"]:
            print("\n    Drift:")
            for w in r["drift_warnings"]:
                print(f"      - {w}")

    n_ok = sum(1 for r in results if not r["skipped"])
    print(f"\nScored {n_ok} cell(s). {len(results) - n_ok} skipped.")
    if n_ok:
        print("  Open outliers.csv in each cell folder: one row per order outside")
        print("  the band, worst first, with the diagnostic columns to explain it.")


if __name__ == "__main__":
    main()
