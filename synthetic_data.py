"""Generate a realistic-ish year of HK VWAP orders for demo/testing.

Emits the CANONICAL column names (see tca/schema.py) so it runs against the
default identity COLUMN_MAP. Performance is engineered to depend on difficulty
(bigger %ADV -> worse mean, wider dispersion) and a handful of true outliers
are injected so the flagging has something to catch.

Deterministic: pass a seed for reproducibility (avoids Math.random-style drift).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

ALGOS = ["VWAP", "VWAP_Passive", "VWAP_Aggressive"]
# Per-algo behaviour: (mean edge in spreads, base noise in spreads, adv sensitivity)
ALGO_PROFILE = {
    "VWAP":            (0.05, 0.45, 0.9),
    "VWAP_Passive":    (0.15, 0.35, 0.6),   # captures spread, tighter, less adv-sensitive
    "VWAP_Aggressive": (-0.10, 0.60, 1.4),  # pays for immediacy, more adv-sensitive
}


def generate(n: int = 12000, market: str = "HK", seed: int = 7,
             outlier_frac: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    algo = rng.choice(ALGOS, size=n, p=[0.55, 0.30, 0.15])
    symbol = rng.integers(1, 400, size=n)  # HK-style numeric tickers
    side = rng.choice(["buy", "sell"], size=n)

    # Spread (bps): lognormal, HK large caps tight, small caps wide.
    spread_bps = np.round(rng.lognormal(mean=2.3, sigma=0.6, size=n), 2)  # ~10 bps median
    spread_bps = np.clip(spread_bps, 1.0, 200.0)

    # Size as %ADV: lognormal, mostly small with a fat tail of hard orders.
    pct_adv = np.round(rng.lognormal(mean=0.3, sigma=1.0, size=n), 3)
    pct_adv = np.clip(pct_adv, 0.01, 60.0)

    participation = np.clip(rng.lognormal(mean=-2.3, sigma=0.5, size=n), 0.005, 0.5)
    duration_min = np.clip(rng.lognormal(mean=4.0, sigma=0.7, size=n), 1, 390)
    volatility = np.round(rng.lognormal(mean=2.7, sigma=0.4, size=n), 1)  # daily vol bps

    # Performance in spreads: algo edge - difficulty drag + noise.
    perf_spreads = np.empty(n)
    for a, (edge, noise, adv_sens) in ALGO_PROFILE.items():
        m = algo == a
        drag = adv_sens * 0.04 * pct_adv[m]           # bigger orders cost more
        perf_spreads[m] = edge - drag + rng.normal(0, noise, size=m.sum())

    # Inject genuine outliers (bad fills / benchmark errors), both tails.
    n_out = int(outlier_frac * n)
    idx = rng.choice(n, size=n_out, replace=False)
    signs = rng.choice([-1, 1], size=n_out, p=[0.7, 0.3])  # mostly bad
    perf_spreads[idx] += signs * rng.uniform(3.0, 8.0, size=n_out)

    slippage_bps = np.round(perf_spreads * spread_bps, 2)
    notional = np.round(rng.lognormal(mean=13.5, sigma=1.0, size=n), 0)  # ~ HKD

    return pd.DataFrame({
        "order_id": [f"HK{seed}{i:07d}" for i in range(n)],
        "market": market,
        "algo": algo,
        "benchmark_type": "interval_vwap",
        "symbol": [f"{s:04d}.HK" for s in symbol],
        "side": side,
        "slippage_bps": slippage_bps,          # +ve = beat benchmark (positive_is_good)
        "spread_bps": spread_bps,
        "quantity": rng.integers(1000, 500000, size=n),
        "notional": notional,
        "pct_adv": pct_adv,
        "participation": np.round(participation, 3),
        "duration_min": np.round(duration_min, 1),
        "volatility": volatility,
    })


if __name__ == "__main__":
    df = generate()
    print(df.head())
    print(f"\n{len(df):,} rows | algos: {df['algo'].value_counts().to_dict()}")
