"""Generate a realistic year of HK VWAP orders for demo/testing.

The data-generating process is written down explicitly below, because the whole
point of the three tiers is to see which one recovers it. Nothing here is tuned
to make any particular tier win -- it is tuned to look like an equity execution
book.

DGP
---
1. Difficulty features are drawn independently (spread, %ADV, POV, duration,
   daily vol).
2. The natural scale of slippage is
       sigma_true = sqrt( (0.5*spread)^2 + (0.18*vol*sqrt(T))^2 )
   i.e. spread dominates for small/fast orders, volatility-over-horizon
   dominates for long ones. This is a fact about interval benchmarks, not a
   modelling choice -- Tier 1 and Tier 2 ignore it and pay for it.
3. Expected cost follows the square-root law:
       impact = -c_algo * vol * sqrt(%ADV/100) * (POV/0.10)^0.25
4. Passive algos earn some spread back; aggressive ones pay it.
5. Latent, UNOBSERVED effects the models cannot see:
       - a per-symbol liquidity effect (model misspecification, on purpose)
       - a per-broker skill effect (BRK_C is genuinely worse)
       - VWAP_Aggressive degrades above 20% participation
6. Four cohorts of injected *true failures*, each with its own causal
   fingerprint in the diagnostic columns, so cause attribution can be scored:
       over_aggressive | spread_bleed | missed_close | benchmark_error

The `_true_outlier` / `_true_cause` columns are demo-only ground truth. Real
extracts obviously won't have them; every tier ignores them when fitting.

Deterministic: pass a seed for reproducibility.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

SESSION_MIN = 330.0   # HK continuous session: 09:30-12:00 + 13:00-16:00

ALGOS = ["VWAP", "VWAP_Passive", "VWAP_Aggressive"]
ALGO_P = [0.55, 0.30, 0.15]

# (impact coefficient, spread capture in spreads)
ALGO_PROFILE = {
    "VWAP":            (0.40,  0.05),
    "VWAP_Passive":    (0.30,  0.25),   # works the bid/offer, earns spread
    "VWAP_Aggressive": (0.55, -0.15),   # pays for immediacy
}

BROKERS = ["BRK_A", "BRK_B", "BRK_C", "BRK_D"]
BROKER_P = [0.30, 0.28, 0.22, 0.20]
# Systematic skill, in units of sigma. Small per order -- invisible to any
# single-order threshold, obvious to a t-test over a few thousand orders.
BROKER_SKILL = {"BRK_A": 0.06, "BRK_B": 0.00, "BRK_C": -0.18, "BRK_D": 0.02}

# Injected failure cohorts and their share of the book.
FAILURE_MIX = {
    "over_aggressive": 0.012,   # traded too fast -> own impact, price reverts
    "spread_bleed":    0.010,   # crossed when it should have posted
    "missed_close":    0.008,   # under-participated in the closing auction
    "benchmark_error": 0.008,   # bad marks / stale benchmark, both tails
}


def generate(n: int = 12000, market: str = "HK", seed: int = 7,
             with_diagnostics: bool = True,
             start_date: str = "2025-06-02",
             end_date: str = "2026-05-29") -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- 1) difficulty features ------------------------------------------
    algo = rng.choice(ALGOS, size=n, p=ALGO_P)
    broker = rng.choice(BROKERS, size=n, p=BROKER_P)
    symbol_id = rng.integers(1, 400, size=n)          # HK-style numeric tickers
    side = rng.choice(["buy", "sell"], size=n)

    spread_bps = np.clip(np.round(rng.lognormal(2.3, 0.6, n), 2), 1.0, 200.0)
    pct_adv = np.clip(np.round(rng.lognormal(0.3, 1.0, n), 3), 0.01, 60.0)
    participation = np.clip(rng.lognormal(-2.3, 0.5, n), 0.005, 0.60)
    duration_min = np.clip(rng.lognormal(4.0, 0.7, n), 1.0, SESSION_MIN)
    # Daily volatility in bps: median ~180bps (1.8%), a realistic HK large/mid cap.
    volatility = np.round(rng.lognormal(5.2, 0.35, n), 1)

    horizon = duration_min / SESSION_MIN
    sigma_true = np.sqrt((0.5 * spread_bps) ** 2
                         + (0.18 * volatility * np.sqrt(horizon)) ** 2)

    # --- 2) expected cost: square-root law --------------------------------
    c_algo = np.array([ALGO_PROFILE[a][0] for a in algo])
    edge_algo = np.array([ALGO_PROFILE[a][1] for a in algo])
    pov_factor = (participation / 0.10) ** 0.25
    impact_bps = -c_algo * volatility * np.sqrt(pct_adv / 100.0) * pov_factor
    edge_bps = edge_algo * spread_bps

    # --- 3) latent effects the models cannot observe ----------------------
    sym_effect_by_id = rng.normal(0.0, 0.25, size=401)      # per-symbol liquidity
    sym_effect = sym_effect_by_id[symbol_id]
    skill = np.array([BROKER_SKILL[b] for b in broker])
    # VWAP_Aggressive falls apart at high participation.
    degrade = np.where((algo == "VWAP_Aggressive") & (participation > 0.20),
                       -0.45, 0.0)

    noise_z = rng.normal(0.0, 1.0, n) + sym_effect + skill + degrade
    slippage_bps = impact_bps + edge_bps + noise_z * sigma_true

    # --- 4) baseline diagnostic columns -----------------------------------
    # Reversion: price gives back a share of the impact you caused.
    reversion_bps = 0.35 * np.abs(impact_bps) * np.clip(
        rng.normal(1.0, 0.4, n), 0.0, None)
    base_passive = np.select(
        [algo == "VWAP_Passive", algo == "VWAP_Aggressive"], [0.75, 0.25], 0.50)
    passive_fill_pct = np.clip(
        base_passive - 0.4 * participation + rng.normal(0, 0.10, n), 0.0, 1.0)
    auction_pct = np.clip(rng.lognormal(-2.6, 0.8, n), 0.0, 0.6)
    momentum_bps = rng.normal(0, 1, n) * 0.6 * volatility * np.sqrt(horizon)

    # --- 5) injected true failures, each with a causal fingerprint --------
    true_cause = np.full(n, "none", dtype=object)
    available = rng.permutation(n)
    cursor = 0
    for cause, frac in FAILURE_MIX.items():
        k = int(round(frac * n))
        idx = available[cursor:cursor + k]
        cursor += k
        true_cause[idx] = cause

        if cause == "over_aggressive":
            # Pushed too hard: extra impact now, price snaps back after.
            participation[idx] = np.clip(participation[idx] * rng.uniform(2.0, 3.5, k),
                                         0.0, 0.9)
            hit = rng.uniform(2.0, 5.0, k)
            slippage_bps[idx] -= hit * sigma_true[idx]
            reversion_bps[idx] += rng.uniform(0.6, 1.4, k) * hit * sigma_true[idx]
            passive_fill_pct[idx] = np.clip(passive_fill_pct[idx] * 0.4, 0.0, 1.0)

        elif cause == "spread_bleed":
            # Crossed the spread all day instead of posting. No reversion:
            # you paid the spread, you did not move the price.
            passive_fill_pct[idx] = np.clip(rng.uniform(0.0, 0.10, k), 0.0, 1.0)
            slippage_bps[idx] -= rng.uniform(1.5, 3.5, k) * sigma_true[idx]

        elif cause == "missed_close":
            # HK does a lot of volume on the close; skipping it misses the VWAP.
            auction_pct[idx] = rng.uniform(0.0, 0.01, k)
            slippage_bps[idx] -= rng.uniform(2.0, 4.0, k) * sigma_true[idx]

        elif cause == "benchmark_error":
            # Stale marks / wrong benchmark window. Both tails, no fingerprint --
            # this is the cohort that should surface as "unexplained".
            sign = rng.choice([-1.0, 1.0], size=k, p=[0.55, 0.45])
            slippage_bps[idx] += sign * rng.uniform(3.0, 8.0, k) * sigma_true[idx]

    notional = np.round(rng.lognormal(15.0, 1.1, n), 0)   # USD, median ~3.3m

    # Trading dates spread uniformly across the window, so the demo exercises
    # period labelling and the overlap check rather than only the fallback.
    span = pd.bdate_range(start_date, end_date)
    order_date = pd.Series(span[rng.integers(0, len(span), size=n)]).dt.strftime("%Y-%m-%d")

    out = pd.DataFrame({
        "order_id": [f"HK{seed}{i:07d}" for i in range(n)],
        "market": market,
        # Literally "Date", matching COLUMN_MAP: synthetic_data stands in for a
        # raw extract, so it uses the extract's column names where they matter.
        "Date": order_date.to_numpy(),
        "algo": algo,
        "broker": broker,
        "benchmark_type": "interval_vwap",
        "symbol": [f"{s:04d}.HK" for s in symbol_id],
        "side": side,
        "slippage_bps": np.round(slippage_bps, 2),   # +ve = beat benchmark
        "spread_bps": spread_bps,
        "quantity": rng.integers(1000, 500000, size=n),
        "notional": notional,
        "pct_adv": pct_adv,
        # Emitted in PERCENT (12.5 = 12.5% of volume) to match the real
        # extract's convention; normalize_units scales it back to a fraction.
        "participation": np.round(participation * 100.0, 3),
        "duration_min": np.round(duration_min, 1),
        # Emitted in PERCENT (1.81 = 1.81%/day) to match the real extract's
        # convention; pipeline.normalize_units scales it to bps internally per
        # DataConfig.volatility_unit. The DGP above works in bps throughout.
        "volatility": np.round(volatility / 100.0, 4),
    })

    if with_diagnostics:
        out["reversion_bps"] = np.round(reversion_bps, 2)
        out["passive_fill_pct"] = np.round(passive_fill_pct, 3)
        out["auction_pct"] = np.round(auction_pct, 4)
        out["momentum_bps"] = np.round(momentum_bps, 2)

    out["_true_cause"] = true_cause
    out["_true_outlier"] = true_cause != "none"
    return out


if __name__ == "__main__":
    df = generate()
    print(df.head())
    print(f"\n{len(df):,} rows")
    print(f"algos:   {df['algo'].value_counts().to_dict()}")
    print(f"brokers: {df['broker'].value_counts().to_dict()}")
    print(f"true failures: {int(df['_true_outlier'].sum()):,} "
          f"({100*df['_true_outlier'].mean():.1f}%)")
    print(df['_true_cause'].value_counts().to_dict())
