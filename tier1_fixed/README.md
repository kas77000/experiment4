# Tier 1 --- Fixed thresholds

```bash
python -m tier1_fixed.run
python -m tier1_fixed.run --rule sigma_multiple
```

## Method

One limit, applied to every order:

| rule | test | typical value |
|---|---|---|
| `abs_bps` | `\|slippage_bps\| > L` | 25-50 bps |
| `spread_multiple` | `\|slippage / spread\| > L` | 2-3 spreads |
| `sigma_multiple` | `\|slippage / sigma_expected\| > L` | 3 sigma |

Plus a materiality gate: orders below `min_notional_review` never reach the
queue, however bad they look. No fitting, no reference data, no history needed.

## Why it exists

This is what most best-execution policies actually say in writing, and there are
real reasons for that: it is trivially auditable, a compliance officer can
verify it by hand, and it needs no model to defend to a regulator. If you are
writing an exception policy from scratch, this is where you start.

## Why it fails

The limit does not scale with difficulty, so it measures order size rather than
execution quality. On the demo book, `abs_bps = 50`:

| %ADV bucket | n | flag rate |
|---|---|---|
| <1% | 4,566 | 4.20% |
| 1-5% | 6,258 | 5.42% |
| 5-10% | 863 | 8.92% |
| 10-20% | 245 | 17.55% |
| >20% | 44 | 27.27% |

A 6.5x gradient. Every trader learns within a quarter that big orders always
flag and small ones never do, at which point the report stops being read.

`run.py` prints this table on every run rather than asserting the problem.

## Making it less bad

Switch `rule` to `sigma_multiple`. It is still a single fixed number, but it
divides by `sigma_expected` --- spread and volatility-over-horizon in quadrature
--- so at least the yardstick adapts to the order. That one change costs nothing
and recovers most of the gap to Tier 2 without fitting anything.

## Knobs

`tier1_fixed/config.py` --- `rule`, the three limits, `min_notional_review`.
