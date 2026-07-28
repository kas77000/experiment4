"""A VWAP-native synthetic book: volume curve, price path, schedule deviation.

`synthetic_data.py` generates from an IMPACT model -- cost grows with
sqrt(%ADV) and participation. That is the right DGP for an arrival-price
benchmark and the wrong one for interval VWAP, so it cannot be used to test
Tier 4 without rigging the answer in Tier 3's favour.

This generator simulates the mechanism instead of asserting a cost formula:

  1. an intraday volume curve v_t (U-shaped: busy open, quiet midday, heavy
     close, which is the HK shape)
  2. a price path P_t (random walk at the order's own volatility)
  3. the algo's schedule w_t = the curve plus tracking error. A good VWAP algo
     has small deviation, a bad one drifts off the curve
  4. slippage computed from the IDENTITY, not from a cost model:

         slippage = - SUM_t (w_t - v_t)(P_t - VWAP) / VWAP

  5. the benchmark VWAP INCLUDES the order's own prints, so the
     self-benchmarking dilution emerges rather than being imposed

Two consequences follow for free, and they are what Tier 4 exists to exploit:

  - an algo that tracks the curve has near-zero slippage AT ANY SIZE, so
    sqrt(%ADV) is not the driver
  - a large order dilutes its own benchmark by exactly (1-f), so its reported
    slippage understates the truth

Failure cohorts are curve failures, not impact failures:
    curve_drift    | missed_close | spread_bleed | benchmark_error
"""

from __future__ import annotations
import numpy as np
import pandas as pd

SESSION_MIN = 330.0
N_BUCKETS = 22            # 15-minute buckets across the HK continuous session

ALGOS = ["VWAP", "VWAP_Passive", "VWAP_Aggressive"]
ALGO_P = [0.55, 0.30, 0.15]
# (schedule tracking error, spread capture in spreads)
ALGO_PROFILE = {
    "VWAP":            (0.14,  0.05),
    "VWAP_Passive":    (0.11,  0.25),
    "VWAP_Aggressive": (0.22, -0.15),
}

BROKERS = ["BRK_A", "BRK_B", "BRK_C", "BRK_D"]
BROKER_P = [0.30, 0.28, 0.22, 0.20]
# Extra tracking error, i.e. worse curve following. BRK_C is genuinely worse.
BROKER_SLOPPINESS = {"BRK_A": 0.94, "BRK_B": 1.00, "BRK_C": 1.30, "BRK_D": 0.98}

FAILURE_MIX = {
    "curve_drift":     0.012,   # schedule wandered off the volume curve
    "missed_close":    0.010,   # skipped the closing auction
    "spread_bleed":    0.010,   # crossed all day instead of posting
    "benchmark_error": 0.008,   # bad marks, both tails
}


def _volume_curve(rng, n_buckets: int) -> np.ndarray:
    """U-shaped intraday volume profile, normalized to sum to 1."""
    x = np.linspace(0, 1, n_buckets)
    base = 0.6 + 1.8 * (x - 0.5) ** 2      # busy at both ends
    base[-1] += 1.4                        # HK closing auction is large
    base[0] += 0.35
    base = base * rng.uniform(0.85, 1.15, n_buckets)   # daily noise
    return base / base.sum()


def generate(n: int = 12000, market: str = "HK", seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    algo = rng.choice(ALGOS, size=n, p=ALGO_P)
    broker = rng.choice(BROKERS, size=n, p=BROKER_P)
    symbol_id = rng.integers(1, 400, size=n)
    side = rng.choice(["buy", "sell"], size=n)
    side_sign = np.where(side == "buy", 1.0, -1.0)

    spread_bps = np.clip(np.round(rng.lognormal(2.3, 0.6, n), 2), 1.0, 200.0)
    pct_adv = np.clip(np.round(rng.lognormal(0.3, 1.0, n), 3), 0.01, 60.0)
    duration_min = np.clip(rng.lognormal(4.2, 0.7, n), 15.0, SESSION_MIN)
    volatility = np.round(rng.lognormal(5.2, 0.35, n), 1)     # daily vol, bps

    # Participation over the interval. Related to %ADV but NOT identical: a
    # short window concentrates the same shares into a smaller slice of volume.
    coverage = duration_min / SESSION_MIN
    participation = np.clip((pct_adv / 100.0) / np.maximum(coverage, 0.05)
                            * rng.lognormal(0, 0.35, n), 0.002, 0.85)

    track_sd = np.array([ALGO_PROFILE[a][0] for a in algo])
    track_sd = track_sd * np.array([BROKER_SLOPPINESS[b] for b in broker])
    edge_algo = np.array([ALGO_PROFILE[a][1] for a in algo])

    true_cause = np.full(n, "none", dtype=object)
    available = rng.permutation(n)
    cursor = 0
    cohorts = {}
    for cause, frac in FAILURE_MIX.items():
        k = int(round(frac * n))
        idx = available[cursor:cursor + k]
        cursor += k
        true_cause[idx] = cause
        cohorts[cause] = idx

    # Sloppier schedules for the curve-drift cohort.
    track_sd[cohorts["curve_drift"]] *= rng.uniform(3.0, 6.0,
                                                    len(cohorts["curve_drift"]))

    slippage_bps = np.empty(n)
    auction_pct = np.empty(n)
    close_frac_used = np.empty(n)

    for i in range(n):
        nb = max(int(round(N_BUCKETS * coverage[i])), 3)
        v = _volume_curve(rng, nb)

        # The algo's schedule: the curve plus correlated tracking error.
        w = v + rng.normal(0, track_sd[i] / np.sqrt(nb), nb)
        if true_cause[i] == "missed_close":
            w[-1] *= rng.uniform(0.0, 0.08)          # skipped the close
        w = np.clip(w, 1e-6, None)
        w = w / w.sum()

        # Price path over the window, at this name's own volatility.
        step = volatility[i] * np.sqrt(coverage[i] / nb)
        p = np.cumsum(rng.normal(0, step, nb))       # in bps, relative

        # The benchmark includes our own prints: mix market and own volume.
        f = participation[i]
        v_total = (1 - f) * v + f * w
        vwap_total = float(np.dot(v_total, p))
        px_you = float(np.dot(w, p))

        # Identity: performance vs the benchmark we are part of. Sign so that
        # positive = good for our side.
        slippage_bps[i] = -side_sign[i] * (px_you - vwap_total)
        auction_pct[i] = w[-1]
        close_frac_used[i] = v[-1]

    # Spread capture / cost, and the algos' style edge.
    slippage_bps += edge_algo * spread_bps

    passive_base = np.select(
        [algo == "VWAP_Passive", algo == "VWAP_Aggressive"], [0.75, 0.25], 0.50)
    passive_fill_pct = np.clip(passive_base - 0.3 * participation
                               + rng.normal(0, 0.10, n), 0.0, 1.0)

    idx = cohorts["spread_bleed"]
    passive_fill_pct[idx] = rng.uniform(0.0, 0.10, len(idx))
    slippage_bps[idx] -= rng.uniform(0.8, 2.0, len(idx)) * spread_bps[idx]

    idx = cohorts["benchmark_error"]
    scale = 0.35 * volatility[idx] * np.sqrt(coverage[idx])
    slippage_bps[idx] += rng.choice([-1.0, 1.0], size=len(idx),
                                    p=[0.55, 0.45]) * rng.uniform(2.0, 5.0,
                                                                  len(idx)) * scale

    # Reversion: an order that ran ahead of the curve leaves a footprint.
    reversion_bps = np.clip(
        0.25 * np.abs(slippage_bps) * rng.normal(1.0, 0.4, n), 0, None)
    reversion_bps[cohorts["curve_drift"]] *= rng.uniform(1.5, 3.0,
                                                         len(cohorts["curve_drift"]))

    notional = np.round(rng.lognormal(15.0, 1.1, n), 0)      # USD
    open_share = np.clip(rng.uniform(0.05, 0.30, n) * auction_pct, 0, 1)

    out = pd.DataFrame({
        "aggrTgtId": [f"HK{seed}{i:07d}" for i in range(n)],
        "Strategy": algo,
        "broker": broker,
        "Side": np.where(side == "buy", "BUY",
                         np.where(rng.random(n) < 0.25, "SSH", "SELL")),
        "Sym": [f"{s:04d} {market}" for s in symbol_id],
        "Pvwap": np.round(slippage_bps, 2),
        "Sprd": spread_bps,
        "%Adv": pct_adv,
        "Vol": np.round(volatility / 100.0, 4),          # percent
        "PR": np.round(participation * 100.0, 3),        # percent
        "Dur": np.round(duration_min, 1),
        "$Mln": np.round(notional / 1e6, 4),
        "#Shares": np.round(rng.integers(1000, 500000, size=n) / 1000.0, 3),
        "%POST": np.round(passive_fill_pct * 100, 2),
        "%CLOSE": np.round(auction_pct * 100, 3),
        "%OPEN": np.round(open_share * 100, 3),
        "Rev30min": np.round(-reversion_bps, 2),         # signed, "+ is good"
        "_true_cause": true_cause,
        "_true_outlier": true_cause != "none",
    })
    return out


if __name__ == "__main__":
    df = generate()
    print(df.head())
    print(f"\n{len(df):,} rows")
    print(f"true failures: {int(df['_true_outlier'].sum()):,}")
    print(df["_true_cause"].value_counts().to_dict())
    print(f"\nmedian PR {df['PR'].median():.1f}%   "
          f"median |Pvwap| {df['Pvwap'].abs().median():.1f} bps")
