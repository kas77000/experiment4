# Tier 3 --- Expected-cost model + residual z-scores

The serious version. Instead of banding realized performance, model what the
order *should* have cost given its difficulty, and flag on the residual.

```bash
python -m tier3_model.run
python -m tier3_model.run --algo-effect expose    # compare algos, not orders
python -m tier3_model.run --folds 1               # disable cross-fitting
python -m tier3_model.run --backend empirical     # no statsmodels needed
```

## The pipeline

```
        slippage_bps
             |
      [ 1. normalize ]      / sigma_expected = sqrt((k*spread)^2 + (w*vol*sqrt(T))^2)
             |
         perf_norm
             |
      [ 2. cost model ]     quantile regression on sqrt(%ADV), log(POV),
             |              log(duration), log(spread), log(vol), size x urgency,
             |              algo, market, side  ->  q_lo, q_med, q_hi
             |
      [ 3. residual ]       z = (perf_norm - q_med) / sigma_hat
             |
      [ 4. queue ]          zone + severity + materiality  ->  review list
             |
      [ 5. aggregate ]      mean-z t-tests per broker/algo/bucket  ->  systematic findings
             |
      [ 6. attribute ]      reversion / passive fill / auction / drift  ->  cause + remedy
```

## 1. Normalization

`sigma_expected` combines the spread and volatility-over-horizon in quadrature,
so short orders are spread-scaled and long ones volatility-scaled with no manual
switch. Set in the shared `config.py`. Raise `vol_horizon_weight` toward 1.0 for
arrival / implementation-shortfall benchmarks.

## 2. The cost model

Three quantiles of `perf_norm` are regressed on difficulty. The **median surface
is the expected cost**; the **outer two surfaces are the threshold**. Continuous,
asymmetric, and conditioned on six features at once --- no bucket edges.

Features follow the empirical impact literature. `sqrt_adv` is the square-root
law (Almgren et al.; Kissell's I-Star): impact grows with the *square root* of
size, not linearly. `sqrt_adv x log_pov` captures that a big order traded fast is
worse than the two effects added. Numeric features are standardized on training
statistics, which keeps the fit conditioned and makes coefficients directly
comparable when attributing cost drivers.

**Quantile crossing.** The three quantiles are fitted independently, so in sparse
corners of feature space they can cross (`q_lo > q_med`). Fixed by rearrangement
--- sorting each row's predicted triple --- the standard remedy from Chernozhukov,
Fernandez-Val & Galichon. No refitting required.

**`algo_effect` is a real modelling decision, not a tuning knob:**

| | algo dummies | a flag means | systematic algo problems |
|---|---|---|---|
| `absorb` (default) | in the model | "bad for this algo" | caught by the slice t-tests |
| `expose` | out of the model | "bad for any algo" | flags on every order |

Use `absorb` for an order-level exception queue, `expose` when the question is
which algo to keep.

## 3. Cross-fitting, and why the calibration table is the important output

In-sample quantile fits are optimistically tight: a p2 band fitted and evaluated
on the same rows will always look near-perfect. `cross_fit_predict` scores every
order with a model fitted on the other K-1 folds, so the numbers are honest.

On real data you never learn which orders were "really" bad --- so **coverage is
the only validation you have**:

```
                 bound  nominal_pct  realized_pct
 below q_lo (tau=0.02)          2.0          2.02
 above q_hi (tau=0.99)          1.0          1.06
 below q_med (tau=0.5)         50.0         50.06
```

Realized far *above* nominal means the model is missing a driver. Far *below*
means it is overfitting and will under-flag next quarter. Read this table before
you read anything else.

One consequence, worth knowing: `fit_trim_quantile` defaults to **0**. Trimming
the training tail biases exactly the quantile you are estimating (measured:
0.5% per tail moves a 2.0% nominal gate to 2.56% realized), and quantile
regression's check loss is already outlier-robust --- unlike OLS, an extreme
observation has bounded influence. Gross errors are handled in `pipeline.clean()`
instead.

## 4. The queue

| severity | condition | action |
|---|---|---|
| `OK` | inside the band | none |
| `MONITOR` | outside, `\|z\| < escalate_z` | logged, trended |
| `ESCALATE` | `\|z\| >= escalate_z` | written justification |

Plus a notional materiality gate. Default taus are asymmetric (`0.02 / 0.99`):
underperformance is what you act on, the upper gate exists mainly to surface data
and benchmark errors. Total queue ~3% of the book.

## 5. Slice tests --- where systematic problems live

A single order's z is mostly market noise. Averaged over hundreds of orders it
is not:

```
t = mean_z / (sd_z / sqrt(n))
```

The demo book makes broker `BRK_C` worse by 0.18 sigma per order. That is
invisible to *any* single-order threshold --- it barely shifts the tail rate ---
but over 2,616 orders it is **t = -9.3**. This asymmetry is why the exception
report and the slice report are different products, and why only one of them
finds root causes.

Testing ~40 slices at once means ~2 false positives for free, so p-values are
corrected to Benjamini-Hochberg q-values and the verdict column uses q.

**Tiers 1 and 2 cannot produce this table at all.** Without an expected cost
there is no residual to average, so a broker that is consistently worse is
indistinguishable from one that was handed the harder orders.

## 6. Cause attribution

A z-score says an order was bad. It does not say what to change.

| cause | evidence | remedy |
|---|---|---|
| `over_aggressive` | high post-trade reversion for the order's own scale, corroborated by high POV | cap participation, lengthen the horizon |
| `spread_bleed` | passive fill far below the algo's norm, **and no reversion** | post more, cross less |
| `missed_close` | auction share near zero where peers do real volume | route a closing slice |
| `adverse_momentum` | large interval drift | execution was fine; revisit decision timing |
| `unexplained` | no fingerprint | check marks, benchmark window, fill timestamps |
| `suspiciously_good` | flagged on the upside | almost always a data question |

Rules trigger at percentiles **of your own book**, so nothing needs retuning for
a new market or year. Competing rules are compared in percentile units, which is
what makes "pick the largest excess" defensible.

**The reversion split is the important one.** High reversion means you moved the
price and it came back --- trade slower. No reversion means you paid the spread
or were adversely selected --- a different fix entirely. Without post-trade marks
these two look identical and have opposite remedies, which is why
`reversion_bps` is the one optional column worth fighting for.

On the demo book, 92% of flagged true failures get the correct diagnosis
(`over_aggressive` 100%, `missed_close` 95.6%, `spread_bleed` 90.5%,
`benchmark_error` 85.7% --- the last is injected without a fingerprint, so
"unexplained" *is* the right answer for it).

`explain_order()` renders one order as a paragraph a human can act on: actual vs
expected, the band in bps, which features made it expensive, the diagnostic
evidence with percentiles, the likely cause and the remedy.

## Compatibility

Everything degrades and the degradation is reported:

| missing | behaviour |
|---|---|
| statsmodels | `backend="auto"` falls back to bucketed empirical quantiles of `perf_norm` |
| volatility / duration | `sigma_expected` uses the spread term only, per row |
| a difficulty column | dropped from the spec; individual NaNs get the training median |
| a diagnostic column | the rules needing it are skipped |
| < `min_fit_n` rows | empirical backend |

The coverage table is printed on every run, so a degraded run is visible.

## Knobs

`tier3_model/config.py` --- taus, `algo_effect`, `include_interactions`,
`n_folds`, `fit_trim_quantile`, `escalate_z`, `min_notional_review`, `backend`,
and the five cause-rule percentiles.

## Outputs

```
outputs/tier3/
  model_coefficients.csv   the fitted quantile surfaces --- the artefact you hand over
  calibration.csv          nominal vs realized coverage, out-of-sample
  scored_orders.csv        every order: expected cost, band in bps, z, zone, cause
  review_queue.csv         just what a human should look at, worst first
  slice_findings.csv       systematic effects with t-stats and q-values
```

## What is still missing

- **Post-trade reversion at multiple horizons** (5 / 30 min) would separate
  temporary from permanent impact, which the single reversion column cannot.
- **Venue and fill-level data** would let `spread_bleed` point at the routing
  decision rather than the algo.
- **Order clustering.** The slice t-tests assume independence; many orders in the
  same name on the same day are correlated. Borderline q-values are optimistic.
  A cluster-robust standard error keyed on symbol-day would fix it.
