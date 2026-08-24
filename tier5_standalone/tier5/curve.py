"""The band, drawn.

One picture with two shapes on it: what the data actually looks like (a KDE of
the observed metric) and what the band assumes it looks like (the fitted normal
PDF). The gap between them IS the non-normality that the coverage table reports
numerically -- a reader who will not read a kurtosis figure can see a peaked
middle and fat ends immediately.

The band bounds are drawn as vertical lines with the out-of-band regions
shaded, so `lo` and `hi` are located on the same axes as the distribution they
came from.

matplotlib and seaborn are imported INSIDE the function on purpose. Nothing in
the scoring path may depend on a plotting library being installed.
"""

from __future__ import annotations

import os

import numpy as np


def view_range(x, lo: float, hi: float, scale: float) -> tuple[float, float]:
    """The x window to draw.

    A handful of extreme orders otherwise stretch the axis so far that the band
    and the bulk of the book collapse into an unreadable spike. Frame on the
    band plus the near tails instead. Whatever falls outside the VIEW is still
    counted in the numbers -- clipping the view must never read as clipping the
    data, which is why plot() reports the offscreen count in the caption.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return (min(lo - 0.5 * scale, float(np.percentile(x, 0.2))),
            max(hi + 0.5 * scale, float(np.percentile(x, 99.8))))


# Roughly what fits across a 9-inch figure at the default label size. A longer
# line is not wrapped by matplotlib -- it is silently cut off at BOTH ends, so
# the reader loses the start of the sentence as well as the end.
MAX_CAPTION_CHARS = 82


def caption(*, n: int, k: float, outside: float, n_offscreen: int,
            units: str, data_min: float | None = None,
            data_max: float | None = None) -> str:
    """The line under the x-axis: what the axis is in, and what it summarises.

    The units matter more than they look. A band drawn at -5.59 is five and a
    half SPREADS wide, not five and a half bps, and a reader who assumes bps
    reads the picture as a rounding error.

    The offscreen note needs care for the opposite reason. `view_range` clips
    at the 0.2nd and 99.8th percentiles, so the number of orders beyond the
    frame is ALWAYS about 0.4% of n -- 188 on a 47,000-order book, whatever
    those orders look like. Printed as a bare count it reads as a finding, and
    a reader reasonably asks how 188 orders escaped. So say the rule that
    produced it, give the share, and spend the space on the fact the clipped
    view actually hides: how far the worst orders really reach.
    """
    axis = f"metric in {units}" if units else "metric"
    # A solved k arrives as 4.339160209. Two decimals is the whole signal, and
    # the extra digits read as false precision on a quantile of 47,000 orders.
    k_txt = f"{k:g}" if float(k).is_integer() else f"{k:.2f}"
    lines = [f"{axis}    n = {n:,}    k = {k_txt}    "
             f"outside the band: {100 * outside:.2f}%"]
    if n_offscreen:
        lines.append(f"view clipped at the 0.2/99.8 percentiles, so {n_offscreen} "
                     f"orders ({100 * n_offscreen / n:.1f}%) lie beyond it")
        if data_min is not None and data_max is not None:
            lines.append(f"They reach {data_min:.1f} / {data_max:.1f}, and are "
                         f"counted in every number above.")
        else:
            lines.append("They are counted in every number above.")
    return "\n".join(lines)


def plot(x, *, centre: float, scale: float, lo: float, hi: float,
         path: str, title: str, subtitle: str | None = None,
         k: float = 3.0, normal_label: str = "fitted normal",
         units: str = "") -> str:
    """Write the curve. Returns the line to print (never raises on a missing lib).

    `normal_label` names the dashed curve. It matters: at fit time the normal IS
    fitted to the data underneath it, but at score time it is the FROZEN band's
    normal drawn over a different period's data. Calling both "fitted" would
    suggest the band adapts to each period, which is exactly what it does not do.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "  Curve skipped (no finite values)."
    if not np.isfinite(scale) or scale <= 0:
        return "  Curve skipped (scale is not positive)."

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        return f"  Curve skipped ({exc.name} not installed)."

    outside = float(np.mean((x < lo) | (x > hi)))

    x_min, x_max = view_range(x, lo, hi, scale)
    n_offscreen = int(np.sum((x < x_min) | (x > x_max)))

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    # Observed shape.
    sns.kdeplot(x=x, ax=ax, fill=True, color="#2c7fb8", alpha=0.35,
                linewidth=1.6, label="observed (KDE)")

    # Assumed shape: the normal the band is built on. Plain numpy so scipy is
    # not required just to draw a Gaussian.
    grid = np.linspace(x_min, x_max, 512)
    pdf = np.exp(-0.5 * ((grid - centre) / scale) ** 2) / (scale * np.sqrt(2 * np.pi))
    ax.plot(grid, pdf, linestyle="--", linewidth=1.8, color="#d95f0e",
            label=f"{normal_label}  N({centre:.2f}, {scale:.2f}$^2$)")

    # The band.
    ax.axvline(lo, color="#b2182b", linewidth=1.5)
    ax.axvline(hi, color="#b2182b", linewidth=1.5)
    if lo > x_min:
        ax.axvspan(x_min, lo, color="#b2182b", alpha=0.07)
    if hi < x_max:
        ax.axvspan(hi, x_max, color="#b2182b", alpha=0.07)

    # Mid-height, not the top: the legend lives up there, and the density at
    # the band edges is near zero so this space is empty on every real book.
    top = ax.get_ylim()[1]
    label_kw = dict(color="#b2182b", fontsize=9, va="center",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="none", alpha=0.85))
    ax.text(lo, top * 0.55, f" lo {lo:.2f}", ha="left", **label_kw)
    ax.text(hi, top * 0.55, f"hi {hi:.2f} ", ha="right", **label_kw)

    ax.set_xlim(x_min, x_max)

    ax.set_title(title, fontsize=13)
    text = caption(n=int(x.size), k=k, outside=outside,
                   n_offscreen=n_offscreen, units=units,
                   data_min=float(x.min()), data_max=float(x.max()))
    if subtitle:
        text = f"{subtitle}\n{text}"
    ax.set_xlabel(text)
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return (f"  Wrote {path}\n"
            f"  The dashed line is what the band assumes; the filled shape is "
            f"what the data is. The gap between them is the non-normality.")
