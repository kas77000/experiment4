# Tier 5 --- Gaussian 3-sigma band

```bash
python -m tier5_gaussian.run --csv your_file.csv
python -m tier5_gaussian.run --csv your_file.csv --metric perf_norm       # band a normalized metric
python -m tier5_gaussian.run --csv your_file.csv --k 2.5                  # a different number of sigmas
python -m tier5_gaussian.run --csv your_file.csv --estimator robust       # median +/- 3 x 1.4826 x MAD
python -m tier5_gaussian.run --csv your_file.csv --score-level algo_x_adv_bucket
python -m tier5_gaussian.run --self-check                                 # prove the implementation
```

## The method

1. Take the pVWAP slippage for every order in the book: `x1 .. xn`.
2. Assume `x ~ Normal(mu, sigma^2)`; estimate `mu` and `sigma`.
3. The acceptable range is `mu - 3*sigma .. mu + 3*sigma`.
4. Anything outside is flagged and must be justified.

`k = 3` is not an arbitrary number. Under a normal distribution 99.73% of
observations fall within +/- 3 sigma, so choosing `k` is choosing a flag rate:
**0.27% of orders, about 1 in 370**. That is the whole appeal --- the threshold
comes with a promise attached, before you have looked at any data.

On the demo book:

```
  All orders (n = 11,976)
    centre     -11.44
    scale       24.15
    RANGE      -83.88 .. 61.00
```

## Why anyone wants this

Two numbers, one formula, no model. A compliance officer can recompute it in a
spreadsheet, there is nothing to defend to a regulator beyond the arithmetic,
and it is what statistical process control has said in writing since Shewhart.
Those are real advantages and this tier does not pretend otherwise.

The band table decomposes the headline by algo, by %ADV bucket and by the
cross, so the single range comes with the detail underneath it. Groups below
`min_group_n` (200) stay in the table marked `trusted=False` rather than being
dropped --- a sigma estimated from 44 orders is visible, and is not used.

## Does 3 sigma mean what it says here?

No. Not close.

```
   k  promised_inside_pct  actual_inside_pct  promised_outside_pct  actual_outside_pct    ratio  n_outside
 1.0               68.269             78.908                31.731              21.092    0.665       2526
 2.0               95.450             95.274                 4.550               4.726    1.039        566
 3.0               99.730             98.330                 0.270               1.670    6.186        200
 4.0               99.994             99.324                 0.006               0.676  106.777         81
```

At `k = 3` the band flags **1.67%** where it promised 0.27% --- **6.2x** the
queue anyone sizing a review team from the normal table would have planned for.
At `k = 4` it is 107x.

Read the `k = 1` row alongside it: **78.9%** of orders fall within one sigma
where a normal distribution puts 68.3%. The book is simultaneously *more*
concentrated in the middle and *fatter* in the tails than a normal. That is the
classic leptokurtic shape, and it is exactly the shape a two-parameter Gaussian
cannot represent.

The single number that settles the discussion:

```
  To actually flag 0.27% of this book you need k = 5.03, not 3.
  Per tail: k_lo = 5.70, k_hi = 4.09.
```

`k_lo` and `k_hi` differ by 1.6 sigma, which is the asymmetry a symmetric band
structurally cannot express. No single `k` serves both tails.

```
  skew               -0.958   (0 if normal)
  excess kurtosis     9.509   (0 if normal)
  D'Agostino K2      3359.9   p = 0
```

**Treat that p-value with suspicion.** At n = 11,976 every formal normality
test rejects, because it is testing "exactly normal" and nothing real is. The
p-value tells you the sample is large, not that the departure matters. The
coverage table and the QQ plot are the evidence; the test is reported because
somebody always asks for it.

`outputs/tier5/qq_plot.png` is the version that needs no statistics
background: a straight line means normal, and the curl at both ends is the fat
tail.

## Classical vs robust

Both estimators are computed for every group and both are always reported:

```
  centre_classical  scale_classical  lo_classical  hi_classical  centre_robust  scale_robust  lo_robust  hi_robust
            -11.44            24.15        -83.88         61.00          -9.32         17.12     -60.69      42.05
```

The robust row is `median +/- k * 1.4826 * MAD`. The constant `1.4826` is
`1 / Phi^-1(0.75)`, which makes the scaled MAD a **consistent estimator of
sigma under normality** --- so on genuinely normal data the two rows agree, and
any gap between them is the non-normality, measured in the band's own units.

```
  sd is 1.41x the robust scale.
```

The standard deviation is 41% larger than the tail-resistant estimate of the
same quantity. That is the standard deviation being inflated by the very
outliers the band exists to catch: every extreme order widens the band that is
supposed to flag it. Scoring on the robust estimator instead flags 4.3% of the
book rather than 1.7%.

`--self-check` confirms the two agree to three decimal places on 200,000 draws
from a known normal, so the 1.41x is a property of the data and not of the code.

## Where it breaks

**The band is symmetric by construction.** `mu +/- k*sigma` is the same
distance either side of the centre. Slippage is not symmetric --- skew is
-0.96 here --- so the two tails are not equally served. `k_lo = 5.70` against
`k_hi = 4.09` quantifies the gap.

**One band for the whole book does not adjust for difficulty.** The flag rate
climbs with order size, which is the same failure Tier 1 had:

| %ADV bucket | orders | flagged (ALL) | flagged (algo x adv_bucket) |
|---|---|---|---|
| <1% | 4,566 | 1.62% | 1.66% |
| 1-5% | 6,258 | 1.49% | 1.71% |
| 5-10% | 863 | 2.20% | 1.62% |
| 10-20% | 245 | 4.49% | 4.49% |
| >20% | 44 | 6.82% | 6.82% |

`--score-level algo_x_adv_bucket` flattens the first three rows and **leaves
the last two untouched**, which is worth understanding rather than working
around: those cross-cells hold 6 to 143 orders, fall below `min_group_n`, and
correctly fall back to the global band. The buckets that most need their own
threshold are precisely the ones without enough orders to fit a trustworthy
sigma. Conditioning harder does not fix this; it runs out of data first. That
is the argument for Tier 3, which conditions on six variables at once by
fitting a surface instead of counting cells.

**Fitted and scored on the same orders.** The band is fitted on the book it
scores, so "1.67% flagged" is partly circular --- some of that 1.67% is the
band being dragged toward the outliers it then measures. `tier3_model` reports
an out-of-sample number via cross-fitting; this tier cannot.

**No expected cost, so no aggregation.** There is no residual to average, so a
broker who is consistently slightly worse is invisible here. Tier 3's slice
tests are what catch that.

## Knobs

`tier5_gaussian/config.py` --- `k_sigma`, `metric`, `estimator`,
`score_level`, `min_group_n`, `group_levels`, `min_notional_review`,
`make_qq_plot`.
