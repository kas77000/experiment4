"""Detect OMS orders trading limit up / limit down stocks — real-time version.

Same logic as ``detect_limit_orders.py`` but pointed at the *real-time*
databases (RDBs) instead of the historical servers. The RDBs hold only the
current session's data in memory, so there is no date to select: the queries
drop the ``date`` predicate and run once against a live snapshot.

The join is unchanged and reused from ``detect_limit_orders``:
    1. Pull eligible target orders from the ORDER RDB, enriched with each
       target's workorder (child-split) stats.
    2. Pull the Up/Down/Locked quote states for those symbols from the
       QUOTE RDB.
    3. Join orders to quote-states within each order's [t_start, t_end] window.

The ORDER RDB (target + workorder) and the QUOTE RDB (qatt) live on their own
host/port — different from both each other and from the historical servers.

Only host + port are needed per server (pykx IPC — no q license required).

Examples
--------
    python detect_limit_orders_rt.py \\
        --order-host rdb-order --order-port 5010 \\
        --quote-host rdb-quote --quote-port 5011
    python detect_limit_orders_rt.py --host dc2nix2p424 \\
        --order-port 5010 --quote-port 5011 --states up,down --out live.csv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
import pykx as kx

# Reuse the shared join, key list, and time formatter from the batch script so
# the two stay in lock-step.
from detect_limit_orders import (
    ORDER_KEYS,
    format_time_of_day,
    join_orders_quotes,
)


# --------------------------------------------------------------------------- #
# Real-time queries (no date predicate — the RDB holds one session in memory).
# These mirror the historical selects in detect_limit_orders.py, minus `date`.
# --------------------------------------------------------------------------- #

# Eligible target orders (Asian names with a non-null algo), each enriched with
# its workorder child-split stats:
#   nChild            : number of workorders for the target (0 if none)
#   venue             : venue of the latest-created workorder (null if none)
#   first_worker_time : `time` of the first workorder created
#   last_worker_time  : `time` of the last workorder created
# Workorders are sorted by `time` (creation time) so first/last are correct,
# then left-joined so targets with no workorder still appear.
#
# The trailing `[]` immediately *invokes* the niladic lambda: `conn(ORDER_FN)`
# with no args only *evaluates* the lambda literal (returning the function
# itself), so we self-apply here to get the table back.
ORDER_FN = """
{[]
  o:select date, id_target, trader, basket, sym, side, size,
           limit_price, algo, t_gen, t_start, t_end
      from target
      where any sym like/: ("*.IN";"*.JP";"*.C1";"*.C2";"*.CH";"*.TT"),
        not null algo;
  w:select nChild:count i, venue:last venue,
           first_worker_time:first time, last_worker_time:last time
      by id_target
      from `time xasc select id_target, venue, time
        from workorder
        where id_target in exec id_target from o;
  update nChild:0^nChild from o lj w }[]
"""

# Up/Down/Locked quote states, restricted to the traded `syms`.
#   up     : bid only, no ask   (qask=0 & qbid>0)   -> limit up
#   down   : ask only, no bid   (qbid=0 & qask>0)   -> limit down
#   locked : bid == ask > 0     (qbid=qask & qbid>0)
QUOTE_FN = """
{[syms]
  select sym, time,
         state:?[qask=0&qbid>0;`up;?[qbid=0&qask>0;`down;`locked]],
         lastPrice, qbid, qask
    from qatt
    where sym in syms,
      (qask=0&qbid>0) | (qbid=0&qask>0) | ((qbid=qask)&qbid>0) }
"""


# --------------------------------------------------------------------------- #
# Fetch + drive (single live snapshot — no date loop)
# --------------------------------------------------------------------------- #

def fetch_orders(conn: "kx.SyncQConnection") -> pd.DataFrame:
    """Eligible target orders from the live RDB, as a pandas frame."""
    return conn(ORDER_FN).pd()


def fetch_quote_states(conn: "kx.SyncQConnection", syms: list[str]) -> pd.DataFrame:
    """Up/Down/Locked quote states for the given symbols from the live RDB."""
    return conn(QUOTE_FN, kx.SymbolVector(syms)).pd()


def run(order_conn, quote_conn, states) -> pd.DataFrame:
    orders = fetch_orders(order_conn)
    if orders.empty:
        print("  no eligible orders")
        return pd.DataFrame()

    syms = orders["sym"].dropna().unique().tolist()
    quotes = fetch_quote_states(quote_conn, syms)
    day = join_orders_quotes(orders, quotes)

    if states is not None and not day.empty:
        day = day[day["state"].astype(str).isin(states)]

    n = 0 if day.empty else day["id_target"].nunique()
    print(f"  {len(orders):>5} orders -> {n:>4} traded limit stocks")
    return day


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="dc2nix2p424",
                    help="Shared host for both RDBs (default: dc2nix2p424).")
    ap.add_argument("--order-host", default=None,
                    help="Override host for the order RDB (target + workorder).")
    ap.add_argument("--quote-host", default=None,
                    help="Override host for the quote RDB (qatt).")
    ap.add_argument("--order-port", type=int, required=True,
                    help="Port of the order RDB (target + workorder).")
    ap.add_argument("--quote-port", type=int, required=True,
                    help="Port of the quote RDB (qatt).")
    ap.add_argument("--user", default=None, help="Optional username for both servers.")
    ap.add_argument("--password", default=None, help="Optional password for both servers.")
    ap.add_argument("--states", default="up,down,locked",
                    help="Comma list of states to keep: up,down,locked "
                         "(default: all three).")
    ap.add_argument("--out", default="limit_orders_rt.csv",
                    help="Output CSV path (default: limit_orders_rt.csv).")
    args = ap.parse_args(argv)

    states = {s.strip().lower() for s in args.states.split(",") if s.strip()}
    order_host = args.order_host or args.host
    quote_host = args.quote_host or args.host

    creds = {}
    if args.user is not None:
        creds["username"] = args.user
    if args.password is not None:
        creds["password"] = args.password

    print("Detecting limit up/down orders (real-time snapshot)")
    print(f"  orders: {order_host}:{args.order_port}   "
          f"quotes: {quote_host}:{args.quote_port}   states={sorted(states)}")

    with kx.SyncQConnection(host=order_host, port=args.order_port, **creds) as order_conn, \
         kx.SyncQConnection(host=quote_host, port=args.quote_port, **creds) as quote_conn:
        result = run(order_conn, quote_conn, states)

    if result.empty:
        print("\nNo orders are trading limit up/down stocks right now.")
        return 0

    result = result.sort_values(["date", "id_target", "state"]).reset_index(drop=True)

    # Render time-of-day columns as HH:MM:SS.sss instead of "0 days HH:MM:SS.sss".
    for col in ("t_gen", "t_start", "t_end", "stateStart", "stateEnd",
                "first_worker_time", "last_worker_time"):
        if col in result.columns:
            result[col] = result[col].map(format_time_of_day)

    result.to_csv(args.out, index=False)

    print(f"\n=== {len(result)} order x state rows "
          f"({result['id_target'].nunique()} distinct orders) ===")
    print(result["state"].value_counts().to_string())
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
