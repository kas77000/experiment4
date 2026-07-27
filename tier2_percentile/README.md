# Tier 2 --- Empirical percentile bands

```bash
python -m tier2_percentile.run
python -m tier2_percentile.run --metric perf_norm      # better denominator
python -m tier2_percentile.run --peer-csv street.csv   # rank vs a peer universe
```

## Method

1. Group comparable orders: `algo x market x %ADV bucket`.
2. Take percentiles of the normalized metric **within each group**. Default
   p2/p98: inside is acceptable, outside is flagged.
3. Fall back for thin slices: `bucketed -> pooled (algo x market) -> global`.
   Any group below `min_group_n` is not trusted and its orders are scored
   against the nearest trusted parent.

Percentiles rather than mean +/- k*sigma, because execution cost distributions
are skewed with fat left tails. The mean and sigma are both dragged by the very
outliers you are hunting --- a few disasters widen the band until it stops
catching anything. Percentiles are order statistics: a handful of extreme values
cannot move p2/p98. They also give an asymmetric band, which matches the real
shape.

## Peer universe

`fit()` takes whatever reference book you hand it, so `--peer-csv` reproduces the
vendor product (Abel Noser, Virtu/ITG): you are ranked against the street rather
than against your own history, which means you cannot hide behind your own bad
quarter. The peer extract goes through the identical `COLUMN_MAP` and pipeline.

## Two things to get right

**The percentile.** p10/p90 flags 20% of the book *by construction*. If a human
must justify each flag, that queue is unworkable. The default here is p2/p98
(~4%), which is where production exception reports actually run. This is a
calibration choice, not a discovery --- the arithmetic guarantees the flag rate
before you see any data.

**The denominator.** This matters more than the band:

```
--metric perf_in_spreads   ->  F1 27.3%     (slippage / spread)
--metric perf_norm         ->  F1 54.1%     (slippage / sigma_expected)
```

Same code, same percentiles, double the F1. Spread is the right scale for small
fast orders; for interval-VWAP orders worked over hours it is the wrong one. The
default is left at `perf_in_spreads` because that is the convention Tier 2
represents --- switch it in `config.py` and you get most of the benefit for free.

## Why it still falls short of Tier 3

- **The band is a step function.** Every order in the 1-5% ADV bucket shares one
  threshold, so a 1.1% order and a 4.9% order are held to the same standard, and
  the threshold jumps discontinuously at the bucket edge.
- **Bucketing costs you resolution.** Each extra conditioning variable multiplies
  the number of groups; with `min_group_n = 200` you run out of data after two
  or three. Tier 3 conditions on six features continuously.
- **In-sample by construction.** The bands are fitted on the orders being scored,
  so the flag rate is guaranteed, not earned. Tier 3 cross-fits.
- **No expected cost, therefore no residual, therefore no aggregation.** You
  cannot average a percentile rank into "this broker is 0.18 sigma worse", so
  systematic problems stay invisible.

## Knobs

`tier2_percentile/config.py` --- `range_percentiles`, `metric`, `min_group_n`,
`group_keys`, `min_notional_review`.
