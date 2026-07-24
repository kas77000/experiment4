"""Detect OMS orders that traded limit up / limit down (or locked) stocks.

Python + pykx port of ``temp.q``. For a user-selected date range it replays the
historical logic per trading day and concatenates the results:

    1. Pull eligible target orders from the ORDER-history server.
    2. Pull the Up/Down/Locked quote states for those symbols from the
       QUOTE-history server.
    3. Join orders to quote-states within each order's [t_start, t_end] window,
       so an order surfaces when its stock was in a limit state while it traded.

The join is done one date at a time (as in the original single-date q script);
crossing orders and quotes across different dates would be wrong.

Only host + port are needed per server (pykx IPC — no q license required).

Examples
--------
    python detect_limit_orders.py --start 2026.07.01 --end 2026.07.15
    python detect_limit_orders.py --start 2026-07-01 --end 2026-07-15 \\
        --host dc2nix2p424 --order-port 17072 --quote-port 17034 \\
        --states up,down --out limit_orders.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

import pandas as pd
import pykx as kx


# --------------------------------------------------------------------------- #
# Remote queries (sent to the historical servers via IPC).
# These are the exact selects from temp.q, wrapped as parameterized functions.
# --------------------------------------------------------------------------- #

# Eligible target orders for one date `x`: Asian names with a non-null algo.
# Each target is enriched with its child-split stats from `workorder`:
#   nChild : number of workorders created for this target (0 if none)
#   venue  : venue of the latest-created workorder for this target (null if none)
# The workorder rows are sorted by `time` (creation time) ascending so `last
# venue` is the most-recently-created venue. The aggregation is filtered to the
# eligible targets, then left-joined so targets with no workorder still appear
# (nChild=0, venue=null).
ORDER_FN = """
{[x]
  o:select date, id_target, trader, basket, sym, side, size,
           limit_price, algo, t_start, t_end
      from target
      where date=x,
        any sym like/: ("*.IN";"*.JP";"*.C1";"*.C2";"*.CH";"*.TT"),
        not null algo;
  w:select nChild:count i, venue:last venue
      by id_target
      from `time xasc select id_target, venue, time
        from workorder
        where date=x, id_target in exec id_target from o;
  update nChild:0^nChild from o lj w }
"""

# Up/Down/Locked quote states on date `d`, restricted to the traded `syms`.
#   up     : bid only, no ask   (qask=0 & qbid>0)   -> limit up
#   down   : ask only, no bid   (qbid=0 & qask>0)   -> limit down
#   locked : bid == ask > 0     (qbid=qask & qbid>0)
QUOTE_FN = """
{[d;syms]
  select sym, time,
         state:?[qask=0&qbid>0;`up;?[qbid=0&qask>0;`down;`locked]],
         lastPrice, qbid, qask
    from qatt
    where date=d,
      sym in syms,
      (qask=0&qbid>0) | (qbid=0&qask>0) | ((qbid=qask)&qbid>0) }
"""

# Order-attribute columns that identify a single order (the q `by` clause,
# minus the derived `targetSym`/`state`).
ORDER_KEYS = [
    "date", "id_target", "trader", "basket", "sym", "side",
    "size", "limit_price", "algo", "t_start", "t_end",
]


# --------------------------------------------------------------------------- #
# Date helpers
# --------------------------------------------------------------------------- #

def parse_date(s: str) -> dt.date:
    """Accept both q-style (2026.07.01) and ISO (2026-07-01) dates."""
    s = s.strip()
    for sep in (".", "-", "/"):
        try:
            y, m, d = (int(p) for p in s.split(sep))
            return dt.date(y, m, d)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Unrecognized date {s!r}; use YYYY.MM.DD or YYYY-MM-DD.")


def trading_dates(start: dt.date, end: dt.date, include_weekends: bool):
    """Yield each date in [start, end], skipping weekends unless asked not to."""
    day = start
    step = dt.timedelta(days=1)
    while day <= end:
        if include_weekends or day.weekday() < 5:  # Mon-Fri
            yield day
        day += step


# --------------------------------------------------------------------------- #
# Per-date fetch + join
# --------------------------------------------------------------------------- #

def fetch_orders(conn: "kx.SyncQConnection", date: dt.date) -> pd.DataFrame:
    """Eligible target orders for one date, as a pandas frame."""
    return conn(ORDER_FN, date).pd()


def fetch_quote_states(conn: "kx.SyncQConnection", date: dt.date,
                       syms: list[str]) -> pd.DataFrame:
    """Up/Down/Locked quote states for the given symbols on one date."""
    return conn(QUOTE_FN, date, kx.SymbolVector(syms)).pd()


def join_orders_quotes(orders: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """Match orders to same-symbol limit states inside [t_start, t_end].

    Mirrors: ``t1 cross q1 where targetSym=qattSym, time within (t_start;t_end)``
    then aggregate per order x state.
    """
    if orders.empty or quotes.empty:
        return pd.DataFrame()

    merged = orders.merge(quotes, on="sym", how="inner")
    in_window = (merged["time"] >= merged["t_start"]) & \
                (merged["time"] <= merged["t_end"])
    merged = merged[in_window]
    if merged.empty:
        return pd.DataFrame()

    # `last` == latest-in-time value within the group. bid/ask are the qbid/qask
    # of that last one-sided (or locked) status; nChild/venue are per-target
    # constants carried through from the order side.
    merged = merged.sort_values("time")
    grouped = merged.groupby(ORDER_KEYS + ["state"], as_index=False, dropna=False)
    return grouped.agg(
        stateStart=("time", "min"),
        stateEnd=("time", "max"),
        lastPrice=("lastPrice", "last"),
        bid=("qbid", "last"),
        ask=("qask", "last"),
        nChild=("nChild", "first"),
        venue=("venue", "first"),
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run(order_conn, quote_conn, start, end, states, include_weekends) -> pd.DataFrame:
    results = []
    for date in trading_dates(start, end, include_weekends):
        orders = fetch_orders(order_conn, date)
        if orders.empty:
            print(f"  {date}: no eligible orders")
            continue

        syms = orders["sym"].dropna().unique().tolist()
        quotes = fetch_quote_states(quote_conn, date, syms)
        day = join_orders_quotes(orders, quotes)

        if states is not None and not day.empty:
            day = day[day["state"].astype(str).isin(states)]

        n = 0 if day.empty else day["id_target"].nunique()
        print(f"  {date}: {len(orders):>5} orders -> {n:>4} traded limit stocks")
        if not day.empty:
            results.append(day)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=parse_date, required=True,
                    help="Range start date (inclusive), YYYY.MM.DD or YYYY-MM-DD.")
    ap.add_argument("--end", type=parse_date, required=True,
                    help="Range end date (inclusive).")
    ap.add_argument("--host", default="dc2nix2p424",
                    help="Shared host for both servers (default: dc2nix2p424).")
    ap.add_argument("--order-host", default=None,
                    help="Override host for the order-history server.")
    ap.add_argument("--quote-host", default=None,
                    help="Override host for the quote-history server.")
    ap.add_argument("--order-port", type=int, default=17072,
                    help="Port of the order-history server (default: 17072).")
    ap.add_argument("--quote-port", type=int, default=17034,
                    help="Port of the quote-history server (default: 17034).")
    ap.add_argument("--user", default=None, help="Optional username for both servers.")
    ap.add_argument("--password", default=None, help="Optional password for both servers.")
    ap.add_argument("--states", default="up,down,locked",
                    help="Comma list of states to keep: up,down,locked "
                         "(default: all three).")
    ap.add_argument("--include-weekends", action="store_true",
                    help="Query Sat/Sun too (skipped by default).")
    ap.add_argument("--out", default="limit_orders.csv",
                    help="Output CSV path (default: limit_orders.csv).")
    args = ap.parse_args(argv)

    if args.end < args.start:
        ap.error("--end is before --start")

    states = {s.strip().lower() for s in args.states.split(",") if s.strip()}
    order_host = args.order_host or args.host
    quote_host = args.quote_host or args.host

    creds = {}
    if args.user is not None:
        creds["username"] = args.user
    if args.password is not None:
        creds["password"] = args.password

    print(f"Detecting limit up/down orders  {args.start} .. {args.end}")
    print(f"  orders: {order_host}:{args.order_port}   "
          f"quotes: {quote_host}:{args.quote_port}   states={sorted(states)}")

    with kx.SyncQConnection(host=order_host, port=args.order_port, **creds) as order_conn, \
         kx.SyncQConnection(host=quote_host, port=args.quote_port, **creds) as quote_conn:
        result = run(order_conn, quote_conn, args.start, args.end,
                     states, args.include_weekends)

    if result.empty:
        print("\nNo orders traded limit up/down stocks in this range.")
        return 0

    result = result.sort_values(["date", "id_target", "state"]).reset_index(drop=True)
    result.to_csv(args.out, index=False)

    print(f"\n=== {len(result)} order x state rows "
          f"({result['id_target'].nunique()} distinct orders) ===")
    print(result["state"].value_counts().to_string())
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
