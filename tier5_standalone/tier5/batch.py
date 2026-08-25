"""Run fit or score over a whole directory of extracts.

    python -m tier5.batch fit   --dir extracts/year
    python -m tier5.batch score --dir extracts/2026-07 --label 2026-07

Only needed when the cells arrive as separate files. Region, strategy and
period still come from the data rather than the filename, so the directory can
be organised however you like -- this just walks it.

One file failing does not abort the rest. With twelve cells, a single malformed
export should cost you that cell, not the run.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import os

import pandas as pd

import config
from tca import pipeline, report, schema
from tier5 import compat, config as t5cfg, fit, score


def _load(path: str):
    """Returns (frame, metric_supplied). The flag says whether the extract
    carried the pre-normalised column or the pipeline had to derive one -- a
    difference that is invisible in the frame afterwards."""
    raw = pd.read_csv(path)
    df, rep = pipeline.prepare(raw, config.COLUMN_MAP, config.DATA,
                               config.SLIPPAGE_SIGN,
                               pre_transform=getattr(config, "PRE_TRANSFORM", None))
    compat.check_report(rep)
    return df, rep.metric_supplied


def run(mode: str, directory: str, *, bands_dir: str, out_dir: str, cfg,
        label: str | None = None, force: bool = False):
    """Walk `directory` for CSVs and fit or score each. Returns (results, failures)."""
    if mode not in ("fit", "score"):
        raise ValueError(f"mode must be 'fit' or 'score', got {mode!r}")

    paths = sorted(glob.glob(os.path.join(directory, "**", "*.csv"),
                             recursive=True))
    results, failures = [], []
    for path in paths:
        try:
            df, supplied = _load(path)
            if mode == "fit":
                new = fit.fit_frame(df, cfg, bands_dir=bands_dir,
                                    out_dir=out_dir, source_csv=path,
                                    force=force)
            else:
                new = score.score_frame(df, cfg, bands_dir=bands_dir,
                                        out_dir=out_dir, label=label)
            for r in new:
                r["metric_supplied"] = supplied
            results += new
        except score.LeakageError as exc:
            # Not a broken file: a refusal. Almost always one systemic mistake
            # (the wrong window exported) hitting every cell at once, so it is
            # tagged separately and summarised once rather than repeated 12
            # times as if twelve different things went wrong.
            failures.append({"file": path, "kind": "leakage", "error": str(exc)})
        except Exception as exc:                      # noqa: BLE001
            # One bad export must not cost the other eleven cells. The error
            # text is kept and printed in the summary rather than swallowed.
            failures.append({"file": path, "kind": "error",
                             "error": f"{type(exc).__name__}: {exc}"})
    return results, failures


def summary_frame(results: list[dict], mode: str) -> pd.DataFrame:
    """One row per cell -- the only place all twelve are comparable."""
    rows = []
    for r in results:
        row = {"region": r["region"], "strategy": r["strategy"],
               "n": r["n"], "lo": r["lo"], "hi": r["hi"],
               "skipped": r["skipped"], "reason": r["reason"]}
        if mode == "fit":
            row["flag_pct"] = r.get("flag_rate_pct")
        else:
            row["n_flagged"] = r.get("n_flagged")
            row["flag_pct"] = r.get("flag_rate_pct")
            row["fit_flag_pct"] = r.get("fit_flag_rate_pct")
            fit_pct = r.get("fit_flag_rate_pct") or 0.0
            row["vs_fit"] = (round(r["flag_rate_pct"] / fit_pct, 2)
                             if fit_pct else None)
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["region", "strategy"]).reset_index(drop=True)
        for col in ("lo", "hi", "flag_pct", "fit_flag_pct"):
            if col in df.columns:
                df[col] = df[col].round(2)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "score"])
    ap.add_argument("--dir", required=True, help="Directory of CSV extracts.")
    ap.add_argument("--bands-dir", default="bands")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--label", default=None,
                    help="Name the period folder. Defaults to the Date range.")
    ap.add_argument("--k", type=float)
    ap.add_argument("--metric", choices=[schema.SLIPPAGE_BPS,
                                         schema.PERF_IN_SPREADS,
                                         schema.PERF_NORM])
    ap.add_argument("--estimator", choices=list(t5cfg.ESTIMATORS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = t5cfg.CONFIG
    overrides = {}
    if args.k is not None:
        overrides["k_sigma"] = args.k
    if args.metric:
        overrides["metric"] = args.metric
    if args.estimator:
        overrides["estimator"] = args.estimator
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    print(report.header(f"TIER 5 --- BATCH {args.mode.upper()}"))
    results, failures = run(args.mode, args.dir, bands_dir=args.bands_dir,
                            out_dir=args.out_dir, cfg=cfg, label=args.label,
                            force=args.force)

    if not results and not failures:
        print(f"\n  No CSV files found under {args.dir}")
        return

    summary = summary_frame(results, args.mode)
    print("\n=== Summary ===")
    print(report.frame(summary, max_rows=60))

    period = args.label
    if period is None:
        periods = {r.get("period") for r in results if r.get("period")}
        period = periods.pop() if len(periods) == 1 else "mixed"
    dest = os.path.join(args.out_dir, args.mode, period, "_summary.csv")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    summary.to_csv(dest, index=False)
    print(f"\nWrote {dest}")

    lines = config.metric_source_lines(
        (r["strategy"] for r in results),
        supplied=all(r.get("metric_supplied", True) for r in results))
    if lines:
        print("\n=== Metric source ===")
        print("\n".join(lines))

    # A region code that is not in REGION_NAMES is usually a new venue -- but it
    # is also exactly what a typo'd Sym suffix looks like, and that would create
    # a phantom cell with its own band. Say so here as well as in tier5.fit:
    # batch is the path people use when they are adding regions.
    unknown = sorted({r["region"] for r in results
                      if r["region"] not in config.REGION_NAMES})
    if unknown:
        print(f"\n  Unrecognised region code(s): {unknown}")
        print("  They were processed normally. If one of those is a typo in the")
        print("  Sym suffix rather than a new venue, it has just been given its")
        print("  own band -- add real ones to REGION_NAMES in config.py.")

    leakage = [f for f in failures if f.get("kind") == "leakage"]
    errors = [f for f in failures if f.get("kind") != "leakage"]

    if leakage:
        print(f"\n=== REFUSED: {len(leakage)} cell(s) overlap their fit window ===")
        print("\n  This is one export mistake, not many broken files: the period")
        print("  you are scoring includes days the band was already fitted on,")
        print("  which would make the flag rate circular.\n")
        print(f"  {leakage[0]['error']}")
        if len(leakage) > 1:
            print(f"\n  ...and {len(leakage) - 1} more cell(s) with the same overlap.")
        print("\n  Re-export the scoring period, or refit on an earlier window.")

    if errors:
        print(f"\n=== {len(errors)} file(s) failed ===")
        for f in errors:
            print(f"  {f['file']}\n    {f['error']}")


if __name__ == "__main__":
    main()
