"""Plot the performance distribution of an orders file. Seaborn, saved to PNG.

    python distribution.py                              # synthetic demo book
    python distribution.py 2025_q2.csv                  # a raw extract
    python distribution.py outputs/tier4/scored_orders.csv --metric z
    python distribution.py 2025_q2.csv --by broker --out outputs/dist

Everything else in this repo answers "which orders are outliers?". This answers
the question you should ask BEFORE that one: what does the book actually look
like? A threshold fitted on a bimodal book, or on one whose left tail is three
orders wide, is a number with no meaning behind it -- and no summary statistic
will tell you, because a mean of -2bps is a mean of -2bps whether it came from a
tight cloud or from two populations that never touch.

It reads whatever you hand it through the SAME pipeline the tiers use, so the
units, the sign convention and the cleaning are identical -- the shape you see
here is the shape they fit on, not an adjacent one. Files that are already
scored output (they carry `z`, `perf_norm`) are detected and passed through
unprepared.

Four figures, each answering a different question:

    01 overall     what is the shape -- tails, skew, how much is below benchmark
    02 by group    which algo/broker sits where, and how wide is each
    03 ecdf        the same comparison read off precisely at any point
    04 difficulty  does the shape change with order size (%ADV)

Two conventions carried through all of them:

  * HIGHER IS BETTER. `pipeline.prepare` flips whatever sign convention your
    extract uses (config.SLIPPAGE_SIGN), so the LEFT tail is underperformance
    everywhere on these plots regardless of what the source file meant by it.
  * The VIEW is clipped, the DATA is not. TCA distributions have tails that will
    squash the body into one bin. `--clip` trims the x-axis only, and every
    figure states in a footnote how many orders fell outside the frame. Nothing
    is dropped silently; `--clip 0` shows the full range.
"""

from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")           # write files; never try to open a window
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    sys.exit("This script needs seaborn:  pip install seaborn matplotlib")

import config
from tca import dataset, pipeline, report, schema


# --- palette -------------------------------------------------------------
# Fixed slot order, both modes. Categorical hues are assigned by group name and
# never cycled, so a broker keeps its colour between figure 02 and figure 03.
LIGHT = {
    "surface": "#fcfcfb", "text": "#0b0b0b", "text_muted": "#52514e",
    "grid": "#e3e2dd", "good": "#2a78d6", "bad": "#e34948",
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
}
DARK = {
    "surface": "#1a1a19", "text": "#ffffff", "text_muted": "#c3c2b7",
    "grid": "#383835", "good": "#3987e5", "bad": "#e66767",
    "series": ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"],
}
MAX_GROUPS = 8      # one per palette slot; the rest fold into "Other"


# --- metrics -------------------------------------------------------------
# label, axis unit, one line on what the number means. Order is the --metric
# auto preference: raw bps first, because it is the one a desk quotes.
METRICS = {
    schema.SLIPPAGE_BPS: (
        "Slippage vs benchmark", "bps",
        "raw bps, sign-normalized so higher is better"),
    schema.PERF_NORM: (
        "Performance vs expected noise", "sigma",
        "slippage / sigma_expected (spread and vol-over-horizon in quadrature)"),
    schema.PERF_IN_SPREADS: (
        "Performance in spreads", "spreads",
        "slippage / spread"),
    "z": (
        "Residual vs expected cost", "sigma",
        "Tier 3/4 residual: how far off the fitted expected-cost surface"),
    schema.REPORTED_SLIPPAGE_BPS: (
        "Reported slippage (pre de-biasing)", "bps",
        "Tier 4 keeps the extract's own figure here"),
}


def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def looks_prepared(df: pd.DataFrame) -> bool:
    """Already been through the pipeline (or is a tier's scored output)?

    Re-preparing those would re-apply the sign convention to a series that has
    already been flipped, which silently mirrors the whole distribution.
    """
    return any(c in df.columns for c in (schema.PERF_IN_SPREADS, schema.PERF_NORM, "z"))


def load(args) -> tuple[pd.DataFrame, str]:
    """Returns (frame, one-line provenance note)."""
    if args.path:
        if not os.path.exists(args.path):
            sys.exit(f"No such file: {args.path}")
        raw = read_any(args.path)
        source = args.path
    else:
        import synthetic_data
        raw = synthetic_data.generate(n=args.n, seed=args.seed)
        source = f"synthetic ({args.n:,} orders, seed {args.seed})"

    prepared = looks_prepared(raw) if args.prepare == "auto" else args.prepare == "never"
    if prepared:
        df = raw.copy()
        note = f"{source} -- already prepared, used as-is"
    else:
        df, clean = pipeline.prepare(
            raw, config.COLUMN_MAP, config.DATA, config.SLIPPAGE_SIGN,
            pre_transform=getattr(config, "PRE_TRANSFORM", None))
        note = f"{source} -- {clean.rows_out:,} of {clean.rows_in:,} rows survived cleaning"
        print(report.header("PREPARING"))
        print(clean.as_text())

    if not len(df):
        sys.exit("Nothing left to plot after cleaning.")

    # Scored output drops adv_bucket; recreate it so figure 04 still works.
    if schema.ADV_BUCKET not in df.columns:
        df = pipeline.add_buckets(df, config.DATA)
    return df, note


def pick_metric(df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        numeric = list(df.select_dtypes("number").columns)
        if requested not in df.columns:
            sys.exit(f"Column {requested!r} is not in the file. Numeric columns "
                     f"available:\n  " + ", ".join(numeric))
        if requested not in numeric:
            sys.exit(f"Column {requested!r} is not numeric -- nothing to plot a "
                     f"distribution of. Numeric columns available:\n  "
                     + ", ".join(numeric))
        return requested
    for m in METRICS:
        if m in df.columns and df[m].notna().any():
            return m
    sys.exit("None of the known performance columns are present. Pass --metric <column>.")


def pick_group(df: pd.DataFrame, requested: str) -> str | None:
    """The grouping column, or None if it would produce a single panel."""
    if requested == "none":
        return None
    order = [requested] if requested != "auto" else [schema.ALGO, schema.BROKER, schema.MARKET]
    for c in order:
        if c in df.columns and df[c].nunique(dropna=True) > 1:
            return c
    if requested != "auto":
        print(f"  note: {requested!r} is absent or constant -- group figures skipped.")
    return None


def fold_groups(df: pd.DataFrame, by: str, top: int) -> tuple[pd.DataFrame, list[str]]:
    """Keep the `top` largest groups by order count; fold the rest into Other.

    Says out loud what it folded. A cap that hides itself reads as coverage.
    """
    counts = df[by].astype("object").fillna("(missing)").value_counts()
    keep = list(counts.index[:top])
    rest = list(counts.index[top:])
    out = df.copy()
    out[by] = out[by].astype("object").fillna("(missing)")
    if rest:
        n_folded = int(counts[rest].sum())
        print(f"  folding {len(rest)} small {by} values into 'Other' "
              f"({n_folded:,} orders): {', '.join(map(str, rest[:6]))}"
              f"{' ...' if len(rest) > 6 else ''}")
        out.loc[out[by].isin(rest), by] = "Other"
        keep.append("Other")
    return out, keep


def view_limits(s: pd.Series, clip_pct: float) -> tuple[float, float]:
    """x-axis limits. Trims the VIEW at a percentile; the data is untouched."""
    s = s.dropna()
    lo, hi = float(s.min()), float(s.max())
    if clip_pct > 0 and len(s) > 20:
        lo, hi = np.percentile(s, [clip_pct, 100 - clip_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(s.min()), float(s.max())
    pad = 0.04 * (hi - lo) if hi > lo else 1.0
    return lo - pad, hi + pad


def offview_note(s: pd.Series, lo: float, hi: float, metric: str) -> str:
    s = s.dropna()
    below = int((s < lo).sum())
    above = int((s > hi).sum())
    if not (below or above):
        return f"All {len(s):,} orders are inside the frame."
    parts = []
    if below:
        parts.append(f"{below} below (worst {s.min():,.1f})")
    if above:
        parts.append(f"{above} above (best {s.max():,.1f})")
    return ("Axis clipped for readability -- data is not: "
            + " and ".join(parts) + " fall outside the frame.")


# --- numbers to sit alongside the pictures --------------------------------
def describe(s: pd.Series) -> dict:
    s = s.dropna()
    q = np.percentile(s, [5, 25, 50, 75, 95]) if len(s) else [np.nan] * 5
    return {
        "n": len(s),
        "mean": s.mean(), "median": q[2], "sd": s.std(ddof=1) if len(s) > 1 else np.nan,
        "p05": q[0], "p25": q[1], "p75": q[3], "p95": q[4],
        "iqr": q[3] - q[1],
        "skew": s.skew() if len(s) > 2 else np.nan,
        "pct_below_0": 100.0 * (s < 0).mean() if len(s) else np.nan,
    }


def summary_table(df: pd.DataFrame, metric: str, by: str | None) -> pd.DataFrame:
    rows = [{"group": "ALL", **describe(df[metric])}]
    if by:
        for g, sub in df.groupby(by, dropna=False):
            rows.append({"group": str(g), **describe(sub[metric])})
        rows[1:] = sorted(rows[1:], key=lambda r: (np.nan_to_num(r["median"], nan=0.0)))
    return pd.DataFrame(rows).round(3)


# --- figures --------------------------------------------------------------
def _wrap(text: str, fig, size: float = 8.5, frac: float = 1.0) -> str:
    """Wrap a caption to the figure width so it never runs off the page.

    0.55*size is the average glyph width in points for this font at this size --
    close enough, and it beats discovering the overflow in a screenshot.
    """
    import textwrap
    chars = max(40, int(fig.get_figwidth() * 72 * frac / (0.55 * size)))
    return "\n".join(textwrap.wrap(text, chars))


def _finish(fig, ax, title: str, subtitle: str, footnote: str, C: dict):
    # Subtitle starts at the axes edge, not the figure edge -- hence the 0.88.
    sub = _wrap(subtitle, fig, 9.5, 0.88).split("\n")[:2]
    ax.set_title(title, loc="left", fontsize=13, fontweight="semibold",
                 color=C["text"], pad=18 + 13 * (len(sub) - 1))
    ax.text(0, 1.015, "\n".join(sub), transform=ax.transAxes, fontsize=9.5,
            color=C["text_muted"], va="bottom")
    note = _wrap(footnote, fig)
    fig.text(0.008, 0.008, note, fontsize=8.5, color=C["text_muted"], va="bottom")
    fig.tight_layout(rect=(0, 0.03 + 0.018 * note.count("\n"), 1, 1))


def fig_overall(df, metric, C, args, outdir) -> str:
    label, unit, gloss = METRICS.get(metric, (metric, "", ""))
    s = df[metric].dropna()
    lo, hi = view_limits(s, args.clip)

    d = pd.DataFrame({metric: s})
    d["side"] = np.where(s < 0, "below benchmark", "at or above")

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    sns.histplot(data=d, x=metric, hue="side", bins=args.bins, binrange=(lo, hi),
                 multiple="stack", palette={"below benchmark": C["bad"],
                                            "at or above": C["good"]},
                 hue_order=["below benchmark", "at or above"],
                 edgecolor=C["surface"], linewidth=0.6, ax=ax)
    # One density line over the top, in ink rather than a series colour: it
    # describes the whole book, it is not a third category.
    ax2 = ax.twinx()
    sns.kdeplot(x=s, ax=ax2, color=C["text_muted"], linewidth=2, cut=0)
    ax2.set_ylabel("")
    ax2.set_yticks([])
    ax2.grid(False)

    stats = describe(s)
    ax.axvline(0, color=C["text_muted"], linestyle="--", linewidth=1.4, zorder=5)
    ax.axvline(stats["median"], color=C["text"], linestyle="-", linewidth=1.6, zorder=5)
    for x, txt in ((0, "benchmark"),
                   (stats["median"], f"median {stats['median']:,.2f}")):
        if lo < x < hi:
            ax.annotate(f" {txt}", xy=(x, 0.60), xycoords=("data", "axes fraction"),
                        rotation=90, fontsize=9, color=C["text_muted"],
                        ha="right", va="bottom",
                        bbox={"facecolor": C["surface"], "edgecolor": "none",
                              "pad": 1.5, "alpha": 0.85})

    ax.set_xlim(lo, hi)
    ax.set_xlabel(f"{label} ({unit})   -- higher is better")
    ax.set_ylabel("orders")
    sns.move_legend(ax, "upper left", frameon=False, title=None, fontsize=9.5)

    _finish(fig, ax,
            f"Performance distribution -- {label}",
            f"n = {stats['n']:,}   mean {stats['mean']:,.2f}   median "
            f"{stats['median']:,.2f}   sd {stats['sd']:,.2f}   skew {stats['skew']:,.2f}"
            f"   |   {stats['pct_below_0']:.1f}% below benchmark   |   {gloss}",
            offview_note(s, lo, hi, metric), C)
    return save(fig, outdir, f"01_overall_{metric}", args)


def fig_by_group(df, metric, by, C, colors, order, args, outdir) -> str:
    label, unit, _ = METRICS.get(metric, (metric, "", ""))
    s = df[metric].dropna()
    lo, hi = view_limits(s, args.clip)

    # Worst median at the top: the eye starts there, and so should the review.
    med = df.groupby(by)[metric].median().reindex(order)
    y_order = list(med.sort_values().index)

    fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(y_order) + 3.0))
    sns.boxplot(data=df, x=metric, y=by, order=y_order, hue=by, hue_order=y_order,
                palette=colors, legend=False, orient="h", width=0.62,
                linewidth=1.2, fliersize=2.0, showmeans=True,
                meanprops={"marker": "|", "markeredgecolor": C["text"],
                           "markersize": 14, "markeredgewidth": 1.6}, ax=ax)
    ax.axvline(0, color=C["text_muted"], linestyle="--", linewidth=1.4, zorder=0)

    counts = df[by].value_counts()
    ax.set_yticks(range(len(y_order)))
    ax.set_yticklabels([f"{g}   n={counts.get(g, 0):,}" for g in y_order],
                       fontsize=10, color=C["text"])
    ax.set_xlim(lo, hi)
    ax.set_xlabel(f"{label} ({unit})   -- higher is better")
    ax.set_ylabel("")

    _finish(fig, ax,
            f"{label} by {by}",
            "box = IQR with median; whiskers 1.5x IQR; | = mean; dots beyond are "
            "individual orders. Ordered worst median first.",
            offview_note(s, lo, hi, metric)
            + "  Width matters as much as position: a wide box is an inconsistent "
              "process, not a bad average.", C)
    return save(fig, outdir, f"02_by_{by}_{metric}", args)


def fig_ecdf(df, metric, by, C, colors, order, args, outdir) -> str:
    label, unit, _ = METRICS.get(metric, (metric, "", ""))
    s = df[metric].dropna()
    lo, hi = view_limits(s, args.clip)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    sns.ecdfplot(data=df, x=metric, hue=by, hue_order=order, palette=colors,
                 linewidth=2, ax=ax)
    ax.axvline(0, color=C["text_muted"], linestyle="--", linewidth=1.4, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"{label} ({unit})   -- higher is better")
    ax.set_ylabel("share of orders at or below")
    sns.move_legend(ax, "lower right", frameon=False, title=by, fontsize=9.5)

    _finish(fig, ax,
            f"Cumulative distribution by {by}",
            "Read it vertically at x=0: the height of each curve is that group's "
            "share of orders below benchmark. A curve left of another is worse "
            "at every point.",
            offview_note(s, lo, hi, metric), C)
    return save(fig, outdir, f"03_ecdf_{by}_{metric}", args)


def fig_by_difficulty(df, metric, C, args, outdir) -> str | None:
    if schema.ADV_BUCKET not in df.columns or df[schema.ADV_BUCKET].nunique() < 2:
        return None
    label, unit, _ = METRICS.get(metric, (metric, "", ""))
    s = df[metric].dropna()
    lo, hi = view_limits(s, args.clip)

    buckets = [b for b in config.DATA.adv_bucket_labels
               if (df[schema.ADV_BUCKET] == b).any()]
    if (df[schema.ADV_BUCKET] == "unknown").any():
        buckets.append("unknown")

    g = sns.displot(data=df, x=metric, col=schema.ADV_BUCKET, col_order=buckets,
                    col_wrap=min(3, len(buckets)), bins=max(20, args.bins // 2),
                    binrange=(lo, hi), color=C["good"], edgecolor=C["surface"],
                    linewidth=0.5, height=2.9, aspect=1.3,
                    facet_kws={"sharey": False})
    g.set_titles("")     # drop seaborn's "adv_bucket = 1-5%" before writing our own
    counts = df[schema.ADV_BUCKET].value_counts()
    for b, ax in zip(buckets, g.axes.flat):
        sub = df.loc[df[schema.ADV_BUCKET] == b, metric].dropna()
        ax.axvline(0, color=C["text_muted"], linestyle="--", linewidth=1.2)
        if len(sub):
            ax.axvline(sub.median(), color=C["text"], linewidth=1.4)
        ax.set_title(f"{b}    n={counts.get(b, 0):,}    median "
                     f"{sub.median():,.2f}" if len(sub) else str(b),
                     loc="left", fontsize=10.5, color=C["text"])
        ax.set_xlim(lo, hi)
    g.set_axis_labels(f"{label} ({unit})", "orders")

    fig = g.figure
    fig.suptitle(f"{label} by order size (%ADV)", x=0.008, ha="left",
                 fontsize=13, fontweight="semibold", color=C["text"])
    note = _wrap("Per-panel y-scales differ -- compare shape and centre, not bar "
                 "height. " + offview_note(s, lo, hi, metric), fig)
    fig.text(0.008, 0.008, note, fontsize=8.5, color=C["text_muted"], va="bottom")
    fig.tight_layout(rect=(0, 0.04 + 0.015 * note.count("\n"), 1, 0.955))
    return save(fig, outdir, f"04_by_adv_bucket_{metric}", args)


def save(fig, outdir: str, stem: str, args) -> str:
    path = os.path.join(outdir, f"{stem}.{args.format}")
    fig.savefig(path, dpi=args.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def style(C: dict):
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
        "savefig.facecolor": C["surface"], "savefig.bbox": "standard",
        "text.color": C["text"], "axes.labelcolor": C["text_muted"],
        "xtick.color": C["text_muted"], "ytick.color": C["text_muted"],
        "axes.edgecolor": C["grid"], "grid.color": C["grid"],
        "grid.linewidth": 0.7, "axes.grid.axis": "both",
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10.5, "legend.fontsize": 9.5,
    })


def main():
    ap = argparse.ArgumentParser(
        description="Performance distribution of an orders file (seaborn).")
    ap.add_argument("path", nargs="?",
                    help="Orders file (.csv, .xlsx, .parquet). Omit for synthetic.")
    ap.add_argument("--metric", default="auto",
                    help="Column to plot. auto -> " + ", ".join(METRICS))
    ap.add_argument("--by", default="auto",
                    help="Grouping column for figures 02/03 (auto | none | any column).")
    ap.add_argument("--out", default=None,
                    help="Output directory (default: outputs/distribution).")
    ap.add_argument("--clip", type=float, default=1.0,
                    help="Percentile trimmed off each end of the VIEW only "
                         "(default 1; use 0 for the full range).")
    ap.add_argument("--top", type=int, default=MAX_GROUPS,
                    help=f"Largest N groups plotted; rest fold into Other (max {MAX_GROUPS}).")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--dark", action="store_true", help="Dark surface.")
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--prepare", default="auto", choices=["auto", "always", "never"],
                    help="Run the shared TCA pipeline first (auto: skip if the "
                         "file is already prepared/scored output).")
    ap.add_argument("--n", type=int, default=12000, help="Synthetic row count.")
    ap.add_argument("--seed", type=int, default=7, help="Synthetic seed.")
    args = ap.parse_args()
    args.top = max(1, min(args.top, MAX_GROUPS))

    print(report.header("PERFORMANCE DISTRIBUTION"))
    df, note = load(args)
    metric = pick_metric(df, args.metric)
    by = pick_group(df, args.by)
    if by:
        df, order = fold_groups(df, by, args.top)
    else:
        order = []

    C = DARK if args.dark else LIGHT
    style(C)
    colors = {g: C["series"][i % len(C["series"])] for i, g in enumerate(order)}
    outdir = args.out or os.path.join(dataset.OUT_DIR, "distribution")
    os.makedirs(outdir, exist_ok=True)

    label, unit, gloss = METRICS.get(metric, (metric, "", "not a known TCA metric"))
    print(f"\n  source    {note}")
    print(f"  metric    {metric}  --  {label} ({unit}); {gloss}")
    print(f"  grouping  {by or '(none)'}")
    print(f"  view      clipped at the {args.clip:g}/{100 - args.clip:g} percentile "
          f"(data untouched)")

    table = summary_table(df, metric, by)
    print("\n=== Distribution ===")
    print(report.frame(table))
    print("\n  sd and IQR are the columns to read next to the median. Two groups "
          "with\n  the same median and different IQR are two different execution "
          "processes.")

    csv_path = os.path.join(outdir, f"summary_{metric}.csv")
    table.to_csv(csv_path, index=False)

    written = [fig_overall(df, metric, C, args, outdir)]
    if by:
        written.append(fig_by_group(df, metric, by, C, colors, order, args, outdir))
        written.append(fig_ecdf(df, metric, by, C, colors, order, args, outdir))
    written.append(fig_by_difficulty(df, metric, C, args, outdir))

    print("\n=== Written ===")
    for p in [csv_path] + [w for w in written if w]:
        print(f"  {p}")
    if by is None:
        print("\n  No usable grouping column, so the by-group figures were skipped.")


if __name__ == "__main__":
    main()
