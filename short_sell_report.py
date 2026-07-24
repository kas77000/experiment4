"""Daily short-sell order report, by market (real-time RDB).

For a fixed set of Asian markets, pull the day's short-sell (``side="sellshort"``)
orders from the real-time ``target`` table and, per market, report:

  * how many short-sell orders we had,
  * the percentage of completion (executed qty / order qty), and
  * the number of rejections that happened.

Data sources (all on the order RDB — one connection):
  * ``target``        : the orders. ``size`` is the order qty; ``sym`` suffix
                        identifies the market; filter ``side="sellshort"``.
  * ``target_state``  : linked by ``id_target``; the *last* row gives the current
                        ``open`` (remaining qty) and ``make`` (executed qty).
  * ``workorder``     : linked by ``id_target``; a split whose ``state`` contains
                        "reject"/"rejected" is counted as a rejection.

Market is taken from the ``sym`` suffix:
  .HK Hong Kong   .JP Japan   .KS Korea   .MK Malaysia   .TB Thailand

Outputs three files (default prefix ``short_sell_report``):
  * ``<prefix>_by_market.csv`` : one row per market (the summary).
  * ``<prefix>_orders.csv``    : one row per short-sell order (the detail).
  * ``<prefix>.pdf``           : a stylish one-pager, per market, for emailing.

Example
-------
    python short_sell_report.py --host rdb-order --port 5010
    python short_sell_report.py --host dc2nix2p424 --port 5010 \\
        --out-prefix ss_2026-07-24
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

import pandas as pd
import pykx as kx


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# The side value that marks a short-sell order. Assumed to be a q *symbol*
# column (`sellshort). If your `side` column is a string, drop the `$ cast in
# ORDER_FN (compare against the raw string instead).
SHORT_SELL_SIDE = "sellshort"

# sym suffix -> market name, in display order.
MARKETS = {
    ".HK": "Hong Kong",
    ".JP": "Japan",
    ".KS": "Korea",
    ".MK": "Malaysia",
    ".TB": "Thailand",
}
MARKET_ORDER = list(MARKETS.values())


# --------------------------------------------------------------------------- #
# Palette (validated data-viz reference palette, light surface)
# --------------------------------------------------------------------------- #

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"     # completion (magnitude)
CRITICAL = "#d03b3b"  # rejections (a concern metric)
GOOD = "#006300"      # positive delta / high completion


# --------------------------------------------------------------------------- #
# Real-time query (order RDB). Returns one enriched row per short-sell target.
# --------------------------------------------------------------------------- #
#   remaining : last `open` in target_state  (qty still to do)
#   executed  : last `make` in target_state  (qty done)
#   nReject   : # workorders whose `state` contains "reject"/"rejected"
# `last` uses natural (append/chronological) RDB order; if target_state has a
# `time` column you'd rather trust, prepend `` `time xasc `` to that select.
ORDER_FN = """
{[ss]
  s:`$ss;
  o:select id_target, sym, side, size
      from target
      where side=s,
        any sym like/: ("*.HK";"*.JP";"*.KS";"*.MK";"*.TB");
  st:select remaining:last open, executed:last make
       by id_target
       from target_state
       where id_target in exec id_target from o;
  wo:select nReject:sum state like "*reject*"
       by id_target
       from workorder
       where id_target in exec id_target from o;
  o:o lj st;
  o:o lj wo;
  update executed:0N^executed, remaining:0N^remaining, nReject:0^nReject from o }
"""


# --------------------------------------------------------------------------- #
# Fetch + shape
# --------------------------------------------------------------------------- #

def fetch_orders(conn: "kx.SyncQConnection") -> pd.DataFrame:
    """Enriched short-sell orders from the live RDB, as a pandas frame."""
    return conn(ORDER_FN, SHORT_SELL_SIDE).pd()


def market_of(sym: str) -> str:
    """Map a symbol to its market via suffix (e.g. '5.HK' -> 'Hong Kong')."""
    if isinstance(sym, str) and "." in sym:
        return MARKETS.get("." + sym.rsplit(".", 1)[1], "Unknown")
    return "Unknown"


def shape_orders(raw: pd.DataFrame) -> pd.DataFrame:
    """Add market + completion, tidy column names/order for the detail CSV."""
    if raw.empty:
        return pd.DataFrame(columns=[
            "market", "id_target", "sym", "side", "order_qty",
            "executed_qty", "remaining_qty", "completion_pct", "n_rejections",
        ])

    df = raw.copy()
    df["market"] = df["sym"].map(market_of)

    # Executed / remaining may be null if a target has no target_state row yet.
    df["executed"] = pd.to_numeric(df["executed"], errors="coerce").fillna(0)
    df["remaining"] = pd.to_numeric(df["remaining"], errors="coerce").fillna(0)
    df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0)
    df["nReject"] = pd.to_numeric(df["nReject"], errors="coerce").fillna(0).astype(int)

    df["completion_pct"] = 100.0 * df["executed"].where(df["size"] > 0, 0) / \
        df["size"].where(df["size"] > 0, 1)

    df = df.rename(columns={
        "size": "order_qty",
        "executed": "executed_qty",
        "remaining": "remaining_qty",
        "nReject": "n_rejections",
    })
    return df[[
        "market", "id_target", "sym", "side", "order_qty",
        "executed_qty", "remaining_qty", "completion_pct", "n_rejections",
    ]].sort_values(["market", "sym"]).reset_index(drop=True)


def summarise_by_market(orders: pd.DataFrame) -> pd.DataFrame:
    """One row per market: order count, qty, completion %, rejections.

    Every configured market appears, even with zero orders, so the report is
    always complete.
    """
    base = pd.DataFrame({"market": MARKET_ORDER})
    if orders.empty:
        agg = pd.DataFrame(columns=[
            "market", "n_orders", "order_qty",
            "executed_qty", "remaining_qty", "n_rejections",
        ])
    else:
        agg = orders.groupby("market", as_index=False).agg(
            n_orders=("id_target", "nunique"),
            order_qty=("order_qty", "sum"),
            executed_qty=("executed_qty", "sum"),
            remaining_qty=("remaining_qty", "sum"),
            n_rejections=("n_rejections", "sum"),
        )

    out = base.merge(agg, on="market", how="left")
    for col in ("n_orders", "order_qty", "executed_qty", "remaining_qty", "n_rejections"):
        out[col] = out[col].fillna(0).astype(int)

    out["completion_pct"] = 100.0 * out["executed_qty"].where(out["order_qty"] > 0, 0) / \
        out["order_qty"].where(out["order_qty"] > 0, 1)

    # Keep the configured display order.
    out["market"] = pd.Categorical(out["market"], categories=MARKET_ORDER, ordered=True)
    return out.sort_values("market").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# PDF (matplotlib) — a stylish one-pager for emailing
# --------------------------------------------------------------------------- #

def _fmt_int(n) -> str:
    return f"{int(round(n)):,}"


def build_pdf(summary: pd.DataFrame, out_path: str, as_of: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "text.color": INK,
        "axes.edgecolor": BASELINE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })

    total_orders = int(summary["n_orders"].sum())
    total_rejections = int(summary["n_rejections"].sum())
    total_size = int(summary["order_qty"].sum())
    total_exec = int(summary["executed_qty"].sum())
    overall_completion = (100.0 * total_exec / total_size) if total_size else 0.0

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait

    # --- Title band ------------------------------------------------------- #
    fig.text(0.06, 0.955, "Short-Sell Order Report", fontsize=22, fontweight="bold",
             color=INK)
    fig.text(0.06, 0.932, f"By market  ·  {as_of}", fontsize=11, color=INK2)
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.917, 0.917], color=GRID, lw=1,
                              transform=fig.transFigure))

    # --- KPI row ---------------------------------------------------------- #
    kpis = [
        ("Short-sell orders", _fmt_int(total_orders), INK),
        ("Overall completion", f"{overall_completion:.1f}%",
         GOOD if overall_completion >= 50 else INK),
        ("Rejections", _fmt_int(total_rejections),
         CRITICAL if total_rejections else INK),
    ]
    for i, (label, value, colour) in enumerate(kpis):
        x = 0.06 + i * 0.30
        fig.text(x, 0.885, value, fontsize=26, fontweight="bold", color=colour)
        fig.text(x, 0.862, label, fontsize=10.5, color=MUTED)

    # --- Per-market table ------------------------------------------------- #
    ax_t = fig.add_axes([0.06, 0.575, 0.88, 0.235])
    ax_t.axis("off")
    cols = ["Market", "Orders", "Order qty", "Executed", "Completion", "Rejections"]
    rows = []
    for _, r in summary.iterrows():
        comp = f"{r['completion_pct']:.1f}%" if r["n_orders"] else "—"
        rows.append([
            r["market"], _fmt_int(r["n_orders"]), _fmt_int(r["order_qty"]),
            _fmt_int(r["executed_qty"]), comp, _fmt_int(r["n_rejections"]),
        ])

    # bbox=[0,0,1,1] makes the table fill its axes exactly (no internal gap).
    table = ax_t.table(cellText=rows, colLabels=cols, cellLoc="right",
                       colLoc="right", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        if col == 0:
            cell.get_text().set_ha("left")
        if row == 0:  # header
            cell.set_facecolor(INK)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(SURFACE if row % 2 else "#f4f3f0")
            if col == 5 and rows[row - 1][5] != "0":  # rejections > 0
                cell.get_text().set_color(CRITICAL)
                cell.get_text().set_fontweight("bold")

    # --- Two bar charts --------------------------------------------------- #
    def _bar(ax, values, labels, colour, title, fmt):
        order = values.sort_values(ascending=True).index
        y = range(len(order))
        vals = values.loc[order].values
        ax.barh(list(y), vals, color=colour, height=0.62, zorder=3)
        ax.set_yticks(list(y))
        ax.set_yticklabels([labels[i] for i in order], fontsize=9.5, color=INK2)
        ax.set_title(title, fontsize=12, fontweight="bold", color=INK, loc="left",
                     pad=10)
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(length=0)
        ax.set_xticks([])
        span = max(vals.max(), 1)
        for yi, v in zip(y, vals):
            ax.text(v + span * 0.02, yi, fmt(v), va="center", fontsize=9.5,
                    color=INK, fontweight="bold")
        ax.set_xlim(0, span * 1.18)
        ax.margins(y=0.08)

    labels = summary["market"].astype(str).tolist()
    comp_series = pd.Series(summary["completion_pct"].values, index=range(len(summary)))
    rej_series = pd.Series(summary["n_rejections"].values, index=range(len(summary)))

    ax1 = fig.add_axes([0.14, 0.16, 0.32, 0.30])
    _bar(ax1, comp_series, labels, BLUE, "Completion by market",
         lambda v: f"{v:.0f}%")
    ax2 = fig.add_axes([0.60, 0.16, 0.32, 0.30])
    _bar(ax2, rej_series, labels, CRITICAL, "Rejections by market",
         lambda v: _fmt_int(v))

    # --- Footer ----------------------------------------------------------- #
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.085, 0.085], color=GRID, lw=1,
                              transform=fig.transFigure))
    fig.text(0.06, 0.06, f"Generated {as_of}  ·  real-time snapshot",
             fontsize=8.5, color=MUTED)

    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="dc2nix2p424",
                    help="Host of the order RDB (target/target_state/workorder).")
    ap.add_argument("--port", type=int, required=True,
                    help="Port of the order RDB.")
    ap.add_argument("--user", default=None, help="Optional username.")
    ap.add_argument("--password", default=None, help="Optional password.")
    ap.add_argument("--out-prefix", default=None,
                    help="Output file prefix (default: short_sell_report_<YYYY-MM-DD>, "
                         "date-stamped so daily runs don't overwrite each other).")
    args = ap.parse_args(argv)

    if args.out_prefix is None:
        args.out_prefix = f"short_sell_report_{dt.date.today():%Y-%m-%d}"

    creds = {}
    if args.user is not None:
        creds["username"] = args.user
    if args.password is not None:
        creds["password"] = args.password

    as_of = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"Short-sell report (real-time)  ·  {args.host}:{args.port}")

    with kx.SyncQConnection(host=args.host, port=args.port, **creds) as conn:
        raw = fetch_orders(conn)

    orders = shape_orders(raw)
    summary = summarise_by_market(orders)

    orders_csv = f"{args.out_prefix}_orders.csv"
    market_csv = f"{args.out_prefix}_by_market.csv"
    pdf_path = f"{args.out_prefix}.pdf"

    orders.to_csv(orders_csv, index=False)
    summary.to_csv(market_csv, index=False)
    build_pdf(summary, pdf_path, as_of)

    print(f"\n=== {len(orders)} short-sell orders across "
          f"{int((summary['n_orders'] > 0).sum())} markets ===")
    print(summary[["market", "n_orders", "completion_pct", "n_rejections"]]
          .to_string(index=False))
    print(f"\nWrote:\n  {market_csv}\n  {orders_csv}\n  {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
