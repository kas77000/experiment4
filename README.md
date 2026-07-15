# TCA Performance Thresholds

Define, per **algo × market**, a performance **range** on the benchmark
slippage of executed orders. Orders inside the range are normal (**GREEN**),
on the edge are borderline (**GREY**), and outside are flagged (**RED**) —
two-sided, so both unusually *bad* and suspiciously *good* orders surface.

Built for TCA on algo executions (e.g. a year of HK VWAP orders vs interval VWAP).

## The method (Phase 1 — empirical percentile bands)

1. **Normalize slippage by spread.** Raw bps aren't comparable across names, so
   the working metric is performance in *units of spread*:

   `perf_in_spreads = signed_slippage_bps / spread_bps`   (higher = better)

2. **Control for order difficulty.** A big illiquid order is naturally worse
   than a small liquid one, so orders are banded within **%ADV buckets**
   (`algo × market × adv_bucket`) — not one range for the whole algo.

3. **Bands = percentiles of that metric within each group.**
   - **GREEN**  : inside the inner percentiles (default p25–p75)
   - **GREY**   : between inner and outer (p10–p25 and p75–p90)
   - **RED**    : beyond the outer percentiles (below p10 = underperformance,
     above p90 = suspiciously good) → **flagged**

   Percentiles (not mean ± k·σ) so the bands are robust to outliers and handle
   the natural skew of cost distributions without assuming symmetry.

4. **Graceful fallback for thin slices.** Bands are fitted at three levels —
   `bucketed → pooled (algo×market) → global`. Any group with fewer than
   `min_group_n` orders is not trusted; those orders are scored against the
   nearest trusted parent level instead.

> In-sample, ~20% of orders fall in RED **by construction** (p10/p90). The value
> is (a) the fitted table becomes a *fixed* rule applied to *future* orders,
> where the flag rate reflects real drift, and (b) it says *which* orders and
> *how* to compare a hard order fairly against its peers.

## Run it

```bash
pip install -r requirements.txt
python run.py                    # synthetic HK VWAP demo
python run.py --csv your.csv     # your real extract
```

Outputs:
- `outputs/threshold_table.csv` — the bands per group (hand this to your boss)
- `outputs/scored_orders.csv`   — every order tagged + flagged

## Point it at your data

Everything you change lives in **`config.py`**:
- `COLUMN_MAP` — map canonical fields to *your* column names
- `SLIPPAGE_SIGN` — `"positive_is_good"` or `"cost"` (positive = worse)
- band percentiles, %ADV bucket edges, `min_group_n`

No other file needs editing. Canonical field names are in `tca/schema.py`;
essentials are `order_id, market, algo, slippage_bps, spread_bps`.

## Layout

```
config.py            # <- the only file you edit for real data
synthetic_data.py    # realistic demo dataset (canonical columns)
run.py               # end-to-end driver + scoring API demo
tca/
  schema.py          # canonical column names
  pipeline.py        # load -> clean -> metric -> buckets
  thresholds.py      # fit bands + ThresholdModel scoring (with fallback)
  report.py          # human-readable table + zone summary
```

## Scoring a new order

```python
model.score_order(algo="VWAP", market="HK",
                  slippage_bps=-38.0, spread_bps=9.0, pct_adv=3.2)
# -> {'zone': 'RED_LOW', 'perf_in_spreads': -4.22, 'flagged': True, ...}
```

## Phase 2 (planned) — regression cost model

Replace the bucketed percentiles with a **quantile regression** of
`perf_in_spreads` on difficulty features (%ADV, participation, duration,
volatility, spread, time-of-day, momentum). The predicted conditional
p10/p25/p50/p75/p90 *are* the bands — continuous instead of bucketed, and
naturally asymmetric. Same pipeline, `clean`/`metric` steps unchanged; only
`thresholds.fit`/scoring get a regression backend. Adds `statsmodels`.
