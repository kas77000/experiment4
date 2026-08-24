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


def plot(x, *, centre: float, scale: float, lo: float, hi: float,
         path: str, title: str, subtitle: str | None = None,
         k: float = 3.0, normal_label: str = "fitted normal") -> str:
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
    caption = (f"n = {x.size:,}    k = {k:g}    "
               f"outside the band: {100 * outside:.2f}%")
    if n_offscreen:
        caption += (f"    ({n_offscreen} order(s) beyond this view, still "
                    f"counted above)")
    if subtitle:
        caption = f"{subtitle}\n{caption}"
    ax.set_xlabel(caption)
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return (f"  Wrote {path}\n"
            f"  The dashed line is what the band assumes; the filled shape is "
            f"what the data is. The gap between them is the non-normality.")
