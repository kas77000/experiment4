"""Preflight your extract BEFORE running any tier.

    python check_extract.py your_file.csv

Reads the file through the same PRE_TRANSFORM and COLUMN_MAP the tiers use, then
reports:

  1. which canonical fields resolved, and which are missing
  2. the two settings a column NAME cannot tell you, inferred from the data:
       - SLIPPAGE_SIGN   (is +ve good, or is +ve a cost?)
       - volatility_unit / pct_adv_unit
  3. magnitude sanity checks on every numeric field
  4. what pipeline.clean() would drop, and why
  5. the exact config.py settings to use

The sign check is the important one. It is the single setting that silently
inverts every result: the flag rate looks identical either way, but you end up
reviewing your BEST orders. It is inferred from a fact that has to be true in
any real book -- bigger and wider-spread orders perform worse -- so the sign
that makes that relationship hold is the sign your data uses.

Accepts .csv, .xlsx/.xls and .parquet.
"""

from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import config
from tca import pipeline, schema

BAR = "=" * 74


def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def section(title: str):
    print(f"\n{BAR}\n{title}\n{BAR}")


# ---------------------------------------------------------------------------
# 1. mapping
# ---------------------------------------------------------------------------

def check_mapping(raw: pd.DataFrame) -> pd.DataFrame:
    transformed = config.PRE_TRANSFORM(raw)
    rows = []
    for canon, src in config.COLUMN_MAP.items():
        found = src in transformed.columns
        derived = src.startswith("_")
        rows.append({
            "canonical": canon,
            "source": src,
            "derived": derived,
            "found": found,
            "essential": canon in schema.ESSENTIAL,
        })
    return pd.DataFrame(rows).sort_values(
        ["essential", "found"], ascending=[False, False])


# ---------------------------------------------------------------------------
# 2. units
# ---------------------------------------------------------------------------

def infer_volatility_unit(vol: pd.Series) -> tuple[str, str]:
    """Daily equity vol is ~1-3%. In bps that is ~100-300."""
    med = float(vol.median())
    if med > 20:
        return "bps", f"median {med:.1f} -> looks like bps (1-3%/day = 100-300bps)"
    if med > 0.2:
        return "pct", f"median {med:.3f} -> looks like percent (1.8 = 1.8%/day)"
    return "fraction", f"median {med:.5f} -> looks like a fraction (0.018 = 1.8%/day)"


def infer_pct_adv_unit(adv: pd.Series) -> tuple[str, str]:
    med = float(adv.median())
    p95 = float(adv.quantile(0.95))
    if p95 > 1.5:
        return "pct", f"median {med:.3f}, p95 {p95:.2f} -> looks like percent (3.5 = 3.5% ADV)"
    return "fraction", f"median {med:.4f}, p95 {p95:.4f} -> looks like a fraction (0.035 = 3.5% ADV)"


def infer_participation_unit(pr: pd.Series) -> tuple[str, str]:
    """A fraction cannot exceed 1; a percent routinely reaches 10-40."""
    med = float(pr.median())
    p95 = float(pr.quantile(0.95))
    mx = float(pr.max())
    if p95 > 1.5 or mx > 1.5:
        return "pct", f"median {med:.2f}, p95 {p95:.2f}, max {mx:.2f} -> looks like percent (15 = 15% of volume)"
    return "fraction", f"median {med:.4f}, p95 {p95:.4f}, max {mx:.4f} -> looks like a fraction (0.15 = 15% of volume)"


# ---------------------------------------------------------------------------
# 3. sign convention
# ---------------------------------------------------------------------------

def infer_sign(df: pd.DataFrame) -> tuple[str, list[str], int]:
    """Which sign convention makes the data behave like a real execution book?

    Two facts must hold in any equity book, whatever the convention:
        bigger orders perform WORSE      -> corr(perf, %ADV)   < 0
        wider-spread names perform WORSE -> corr(perf, spread) < 0

    We test them on the raw column. If the correlations come out POSITIVE, the
    column is a cost (+ve = worse) and needs flipping. Spearman, so a few
    outliers cannot swing it.
    """
    perf = pd.to_numeric(df[schema.SLIPPAGE_BPS], errors="coerce")
    votes, notes = [], []

    for col, label in [(schema.PCT_ADV, "%ADV"), (schema.SPREAD_BPS, "spread")]:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        ok = perf.notna() & x.notna()
        if ok.sum() < 50:
            continue
        rho, p = stats.spearmanr(x[ok], perf[ok])
        if not np.isfinite(rho):
            continue
        verdict = "positive_is_good" if rho < 0 else "cost"
        strength = "clear" if p < 0.01 and abs(rho) > 0.03 else "WEAK"
        votes.append(verdict if strength == "clear" else None)
        notes.append(f"  corr(raw, {label:<6}) = {rho:+.3f}  (p={p:.1e})  "
                     f"-> {verdict:<17} [{strength}]")

    med = float(perf.median())
    notes.append(f"  median raw slippage   = {med:+.2f} bps"
                 f"  -> {'positive_is_good' if med < 0 else 'cost'} "
                 f"[WEAK, most books lose to VWAP on average]")

    real = [v for v in votes if v]
    if not real:
        return "UNKNOWN", notes, 0
    if len(set(real)) > 1:
        return "CONFLICTED", notes, 0
    return real[0], notes, len(real)


REVERSION_CONVENTIONS = ["raw", "raw_inverted", "signed", "signed_inverted"]


def _apply_reversion(rev: pd.Series, side_sign: pd.Series | None,
                     convention: str) -> pd.Series:
    out = rev.astype(float)
    if convention in ("raw", "raw_inverted"):
        if side_sign is None:
            return pd.Series(np.nan, index=rev.index)
        out = out * side_sign
    if convention in ("raw_inverted", "signed_inverted"):
        out = -out
    return out


def infer_reversion_sign(raw: pd.DataFrame, named: pd.DataFrame):
    """Score all four reversion conventions against the book.

    The physics: bigger and faster orders push the price harder, so under the
    CORRECT convention reversion must RISE with size and participation. The
    wrong sign inverts that; an unsigned raw column on a mixed buy/sell book
    washes it out to roughly zero. So the convention with the strongest
    positive correlation is the one your data uses.
    """
    if "Rev30min" not in raw.columns:
        return None, []

    rev = pd.to_numeric(raw["Rev30min"], errors="coerce")
    side_sign = None
    if schema.SIDE in named.columns:
        side_sign = named[schema.SIDE].map({"buy": 1.0, "sell": -1.0})

    drivers = [(c, l) for c, l in [(schema.PCT_ADV, "%ADV"),
                                   (schema.PARTICIPATION, "POV")]
               if c in named.columns]
    if not drivers:
        return None, ["  no %ADV or participation column to test against"]

    rows = []
    for conv in REVERSION_CONVENTIONS:
        adj = _apply_reversion(rev, side_sign, conv)
        if adj.isna().all():
            rows.append({"convention": conv, "score": np.nan,
                         "note": "needs a Side column"})
            continue
        cors = {}
        for col, label in drivers:
            x = pd.to_numeric(named[col], errors="coerce")
            ok = adj.notna() & x.notna()
            cors[label] = (stats.spearmanr(x[ok], adj[ok])[0]
                           if ok.sum() > 50 else np.nan)
        score = np.nanmean(list(cors.values()))
        rows.append({"convention": conv, **{f"corr_{k}": round(v, 3)
                                            for k, v in cors.items()},
                     "score": round(float(score), 3), "note": ""})

    table = pd.DataFrame(rows)
    valid = table.dropna(subset=["score"])
    best = valid.loc[valid["score"].idxmax(), "convention"] if len(valid) else None
    return best, table


# ---------------------------------------------------------------------------
# 4. magnitudes
# ---------------------------------------------------------------------------

EXPECTED = {
    schema.SLIPPAGE_BPS: ("bps vs interval VWAP", -200, 200),
    schema.SPREAD_BPS:   ("bps", 0.5, 300),
    schema.PCT_ADV:      ("percent of ADV", 0.001, 100),
    schema.VOLATILITY:   ("daily vol, bps", 20, 1500),
    schema.PARTICIPATION: ("fraction of volume", 0.002, 0.9),
    schema.DURATION_MIN: ("minutes", 1, 400),
    schema.NOTIONAL:     ("currency", 1e3, 1e11),
    schema.QUANTITY:     ("shares", 1, 1e10),
}


def check_magnitudes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, (unit, lo, hi) in EXPECTED.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        med = float(s.median()) if s.notna().any() else np.nan
        rows.append({
            "column": col,
            "expect": unit,
            "n_null": int(s.isna().sum()),
            "min": round(float(s.min()), 3) if s.notna().any() else np.nan,
            "median": round(med, 3),
            "max": round(float(s.max()), 3) if s.notna().any() else np.nan,
            "plausible": "yes" if (pd.notna(med) and lo <= abs(med) <= hi) else "CHECK",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Your extract (.csv, .xlsx, .parquet)")
    ap.add_argument("--rows", type=int, help="Read only the first N rows.")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        sys.exit(f"No such file: {args.path}")

    raw = read_any(args.path)
    if args.rows:
        raw = raw.head(args.rows)

    section(f"EXTRACT: {args.path}")
    print(f"  {len(raw):,} rows x {len(raw.columns)} columns")
    print(f"\n  columns present: {', '.join(map(str, raw.columns))}")

    # ---- 1. mapping ----
    section("1. COLUMN MAPPING")
    m = check_mapping(raw)
    print(m.to_string(index=False))

    missing_essential = m[(m["essential"]) & (~m["found"])]["canonical"].tolist()
    if missing_essential:
        print(f"\n  STOP: essential columns unresolved: {missing_essential}")
        print("  Fix COLUMN_MAP / PRE_TRANSFORM in config.py before continuing.")
        sys.exit(1)
    print("\n  All essential columns resolved.")

    missing_opt = m[(~m["essential"]) & (~m["found"])]["canonical"].tolist()
    if missing_opt:
        print(f"  Absent (features degrade, nothing breaks): {', '.join(missing_opt)}")

    # Named frame, no unit scaling and no sign flip yet -- the checks below need
    # the raw column exactly as it arrived.
    named = pipeline.load_orders(raw, config.COLUMN_MAP,
                                 pre_transform=config.PRE_TRANSFORM)

    # ---- 2. units ----
    section("2. UNITS  (set these in config.DataConfig)")
    vol_unit = pct_unit = pr_unit = None
    if schema.VOLATILITY in named.columns:
        vol_unit, why = infer_volatility_unit(
            pd.to_numeric(named[schema.VOLATILITY], errors="coerce").dropna())
        flag = "" if vol_unit == config.DATA.volatility_unit else "   <-- CHANGE"
        print(f"  volatility_unit = {vol_unit!r:<12} {why}")
        print(f"    currently set to {config.DATA.volatility_unit!r}{flag}")
    if schema.PCT_ADV in named.columns:
        pct_unit, why = infer_pct_adv_unit(
            pd.to_numeric(named[schema.PCT_ADV], errors="coerce").dropna())
        flag = "" if pct_unit == config.DATA.pct_adv_unit else "   <-- CHANGE"
        print(f"\n  pct_adv_unit    = {pct_unit!r:<12} {why}")
        print(f"    currently set to {config.DATA.pct_adv_unit!r}{flag}")
    if schema.PARTICIPATION in named.columns:
        pr_unit, why = infer_participation_unit(
            pd.to_numeric(named[schema.PARTICIPATION], errors="coerce").dropna())
        flag = "" if pr_unit == config.DATA.participation_unit else "   <-- CHANGE"
        print(f"\n  participation_unit = {pr_unit!r:<9} {why}")
        print(f"    currently set to {config.DATA.participation_unit!r}{flag}")
        print("    (thresholds are invariant to this -- log(POV) absorbs a constant"
              " scale.\n     It only affects how POV is displayed.)")

    # ---- 3. sign ----
    section("3. SLIPPAGE SIGN  (the setting that silently inverts everything)")
    verdict, notes, nvotes = infer_sign(named)
    for n in notes:
        print(n)
    print()
    if verdict in ("UNKNOWN", "CONFLICTED"):
        print(f"  RESULT: {verdict} -- the data does not settle it.")
        print("  Confirm from the source system's documentation. As a fallback,")
        print("  pull five orders you KNOW went badly and check the sign of Pvwap.")
    else:
        agree = verdict == config.SLIPPAGE_SIGN
        print(f"  RESULT: SLIPPAGE_SIGN = {verdict!r}   ({nvotes}/2 checks agree)")
        print(f"  currently set to {config.SLIPPAGE_SIGN!r}"
              f"{'  -- correct' if agree else '   <-- CHANGE THIS'}")

    # ---- 3b. reversion convention ----
    rev_best = None
    if "Rev30min" in raw.columns:
        section("3b. REVERSION CONVENTION  (Rev30min)")
        rev_best, rev_table = infer_reversion_sign(raw, named)
        if isinstance(rev_table, pd.DataFrame) and len(rev_table):
            print(rev_table.to_string(index=False))
        elif rev_table:
            print("\n".join(rev_table))
        print("\n  Correct convention = the one where reversion RISES with size and")
        print("  participation, because bigger and faster orders push the price")
        print("  harder and it snaps back further. A negative score means the sign")
        print("  is inverted; a score near zero means buys and sells are cancelling")
        print("  (unsigned raw data).")
        if rev_best:
            agree = rev_best == config.REVERSION_SIGN
            print(f"\n  RESULT: REVERSION_SIGN = {rev_best!r}")
            print(f"  currently set to {config.REVERSION_SIGN!r}"
                  f"{'  -- correct' if agree else '   <-- CHANGE THIS'}")

        if schema.SIDE in named.columns:
            vc = named[schema.SIDE].value_counts(dropna=False)
            print(f"\n  side parsed: {vc.to_dict()}")
            n_bad = int(named[schema.SIDE].isna().sum())
            if n_bad:
                unmapped = sorted(set(
                    raw.loc[named[schema.SIDE].isna(), "Side"].astype(str)))[:8]
                print(f"  WARNING: {n_bad:,} rows have an unrecognised Side value: "
                      f"{unmapped}")
                print("  Add them to BUY_TOKENS / SELL_TOKENS in config.py.")

    # ---- 4. magnitudes ----
    section("4. MAGNITUDE SANITY  (after unit conversion)")
    scaled = pipeline.normalize_units(named, config.DATA)
    print(check_magnitudes(scaled).to_string(index=False))
    print("\n  Volatility, %ADV and participation are shown AFTER conversion by the")
    print("  units currently set in config.py, so this cross-checks section 2: if a")
    print("  unit setting is wrong, the converted median lands outside its range and")
    print("  reads 'CHECK'. Otherwise 'CHECK' usually means a mapping problem.")
    if schema.DURATION_MIN in scaled.columns:
        med = float(pd.to_numeric(scaled[schema.DURATION_MIN],
                                  errors="coerce").median())
        print(f"\n  duration median {med:.1f} min = "
              f"{med / config.DATA.minutes_per_day:.0%} of a session. If that reads")
        print("  like hours, (endtime - starttime) was already in minutes: drop the /60.")

    if schema.MARKET in named.columns:
        vc = named[schema.MARKET].value_counts()
        print(f"\n  markets parsed from Sym: {vc.to_dict()}")
        odd = [k for k in vc.index if not (isinstance(k, str) and k.isalpha())]
        if odd:
            print(f"  WARNING: these do not look like venue codes: {odd}")
            print("  Check the Sym format -- PRE_TRANSFORM takes the last 2 chars.")

    if schema.ALGO in named.columns:
        print(f"\n  strategies: {named[schema.ALGO].value_counts().to_dict()}")

    # ---- 5. what the pipeline would do ----
    section("5. WHAT THE PIPELINE WOULD KEEP")
    df, rep = pipeline.prepare(raw, config.COLUMN_MAP, config.DATA,
                               config.SLIPPAGE_SIGN,
                               pre_transform=config.PRE_TRANSFORM)
    print(rep.as_text())

    if len(df):
        pn = df[schema.PERF_NORM]
        print(f"\n  perf_norm  median {pn.median():+.2f}  sd {pn.std():.2f}"
              f"  p1 {pn.quantile(0.01):+.2f}  p99 {pn.quantile(0.99):+.2f}")
        print("  Healthy is sd roughly 0.7-1.5. Much larger means sigma_expected is")
        print("  too small (check units); much smaller means it is too large.")

        n_groups = df.groupby([schema.ALGO, schema.MARKET,
                               schema.ADV_BUCKET]).size()
        print(f"\n  algo x market x bucket groups: {len(n_groups)}"
              f"  ({int((n_groups >= 200).sum())} with >= 200 orders)")
        if len(df) < 500:
            print(f"\n  NOTE: {len(df):,} usable rows. Tier 3 needs min_fit_n=500 to")
            print("  regress; below that it falls back to empirical bands.")

    # ---- verdict ----
    section("SETTINGS TO USE")
    print("  config.py:")
    print(f"    SLIPPAGE_SIGN = {verdict!r}"
          if verdict not in ("UNKNOWN", "CONFLICTED")
          else "    SLIPPAGE_SIGN = <confirm from source docs>")
    if vol_unit:
        print(f"    DataConfig.volatility_unit = {vol_unit!r}")
    if pct_unit:
        print(f"    DataConfig.pct_adv_unit    = {pct_unit!r}")
    if pr_unit:
        print(f"    DataConfig.participation_unit = {pr_unit!r}")
    print(f"    DataConfig.minutes_per_day = {config.DATA.minutes_per_day:g}"
          f"   <- set for the market you trade most")
    print(f"    DataConfig.default_duration_min = {config.DATA.default_duration_min}"
          f"   <- assumption; no duration column present")
    print("\n  Then:")
    print("    python run.py --csv " + args.path)
    print("    python -m tier3_model.run --csv " + args.path)


if __name__ == "__main__":
    main()
