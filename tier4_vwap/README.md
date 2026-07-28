# Tier 4 --- VWAP-native thresholds

```bash
python -m tier4_vwap.run --csv your_file.csv
python -m tier4_vwap.run --csv your_file.csv --compare-tier3   # coefficients side by side
python -m tier4_vwap.run --csv your_file.csv --no-debias       # what the correction is worth
```

Tier 3 borrows its machinery from **market-impact** models --- I-Star,
Almgren-Chriss, the square-root law. Those were derived for an **arrival-price /
implementation-shortfall** benchmark: *how far did you push the price away from
the decision point?* Interval VWAP is a different question, and two of Tier 3's
premises do not survive the change.

## What is wrong with an impact model here

**1. The benchmark moves with you.** Interval VWAP is contemporaneous, so much
of what an impact model predicts is already inside the benchmark.

**2. Slippage vs interval VWAP is an identity, not a cost.**

```
slippage  =  - SUM_t (w_t - v_t)(P_t - VWAP)
```

`w_t` is your share of your own order in bucket `t`; `v_t` is the market's share
of interval volume. Match the curve --- `w_t = v_t` --- and slippage is
**exactly zero, at any size**. The error is a covariance between schedule
deviation and the intraday price path. `sqrt(%ADV)` is the wrong functional form.

**3. Participation is an output, not a decision.** For a VWAP algo, POV is
whatever the volume curve dictates. Tier 3's `log_pov` and `sqrt_adv × log_pov`
model it as an urgency choice, which is a POV/IS framing that does not apply.

`synthetic_vwap.py` generates a book from this mechanism rather than from a cost
formula --- U-shaped volume curve, price path, schedule with tracking error,
benchmark including own prints. On that book:

```
corr(Pvwap, %Adv) = +0.001   p = 0.93
corr(Pvwap, PR)   = -0.003   p = 0.76
```

Size explains nothing. An impact model has no signal to find.

## What Tier 4 does instead

### 1. De-bias the benchmark --- exact, no model

You are part of your own benchmark. With `f` your share of interval volume:

```
VWAP_total = (1-f)·VWAP_others + f·P_you
      =>    P_you - VWAP_others = (P_you - VWAP_total) / (1 - f)
```

Reported slippage understates performance against the rest of the market by
exactly `1/(1-f)`. This is algebra, not a fit.

So participation *does* matter enormously --- just not as a cost driver. It is
the **dilution factor**, and it belongs in the metric. On the demo book the
correction scales cleanly with size:

| %ADV bucket | median PR | correction | reported | de-biased |
|---|---|---|---|---|
| <1% | 2.5% | ×1.03 | 0.94 | 0.98 |
| 1-5% | 10.4% | ×1.12 | 0.95 | 1.14 |
| 5-10% | 33.4% | ×1.50 | 0.98 | 1.68 |
| 10-20% | 63.7% | ×2.00 | 0.67 | 1.27 |
| >20% | 85.0% | ×2.00 | 1.19 | 2.37 |

The correction **amplifies whatever sign is there**. An underperforming large
order looks worse; an outperforming one looks better. Either way the biggest
orders are the ones whose true performance is most obscured, because they are
the ones most made of their own prints.

Participation above `max_dilution_participation` (default 50%) is capped and
flagged in `dilution_capped` --- beyond that the factor explodes and the
participation figure is usually unreliable anyway.

### 2. Scale by tracking error, not impact

```
sigma_track = sqrt( (0.25·spread)² + (0.35·vol·sqrt(T/S))² )
```

Same quadrature shape as Tier 3 but reweighted: the volatility-over-horizon term
dominates, because that is what schedule deviation interacts with. The spread
term only covers per-child-order cost.

### 3. Curve difficulty, not impact features

| dropped | why |
|---|---|
| `log_pov`, `sqrt_adv × log_pov` | participation is an output, and is now used in the metric |

| added | why |
|---|---|
| `session_coverage` | an order spanning the whole session tracks the curve almost by construction; one squeezed into twenty minutes carries far more curve risk, since a single misjudged bucket is a large share of the schedule |

`sqrt_adv` is kept but demoted --- no interaction, no POV partner --- so you can
check whether size still matters rather than assume either way.

Deliberately **not** features: `%POST` and auction share. Those are execution
*choices*, not order difficulty. Putting them in the expectation would let an
algo that crosses the spread all day lower its own bar --- the same trap as
absorbing the algo effect. They stay diagnostic.

### 4. Test consistency, not just bias

**This is the most important difference, and it was found by testing rather than
by reasoning.**

Poor curve tracking does not bias an order in a direction. It **widens the
distribution**. The order then lands somewhere random on a wider spread, so on
average it looks fine while being unreliable order by order. On the demo book:

| cohort | n | mean z | **sd z** |
|---|---|---|---|
| clean | 11,520 | +0.014 | 0.487 |
| injected `curve_drift` | 144 | −0.212 | **1.356** |

Nearly 3× the dispersion, almost no shift in the mean. A mean-z t-test
**structurally cannot see this**. At broker level, where one broker was made
1.30× sloppier at curve tracking:

| broker | sloppiness | mean-z t | Levene p on variance |
|---|---|---|---|
| BRK_A | 0.94× | +0.40 | 8.8e-03 |
| BRK_B | 1.00× | −1.05 | 9.0e-01 |
| **BRK_C** | **1.30×** | **−1.96** | **1.3e-04** |

`t = -1.96` does not survive FDR correction across ~40 slices. Levene at 1.3e-04
does, and comes out as `INCONSISTENT (strong)` at q = 0.0012.

So Tier 4 reports slices twice: **6a bias** (mean-z t-tests, as Tier 3) and
**6b consistency** (Levene on z variance). A desk that is inconsistent rather
than consistently bad shows up in 6b and nowhere else.

Levene rather than an F-test because it uses absolute deviations from the group
centre and does not assume normality --- z-scores from a quantile fit have
heavier tails than a normal.

## The honest results

On the VWAP-native book, at a matched ~1.7% queue:

| | precision | recall | F1 |
|---|---|---|---|
| Tier 3 | 78.3% | 33.3% | 46.8% |
| Tier 4, no de-biasing | 77.5% | 33.3% | 46.6% |
| **Tier 4** | 76.2% | 34.4% | **47.4%** |

**Per-order detection is essentially tied.** That is a real finding, not a
disappointment, and it has a clean explanation: the de-biasing rescales each
order *and* the band refits around it, so the flagging decision --- a relative
comparison --- barely moves. De-biasing changes what the number **means**, and
therefore the cash figures and the interpretation, far more than it changes the
ranking.

Where Tier 4 actually wins:

- **the measurement is correct.** A 20%-participation order reported at −20 bps
  really lost 25 bps against the rest of the market. Tier 3 never says this.
- **the consistency test finds what the mean test cannot.** BRK_C is invisible
  to Tier 3 and unmissable in Tier 4.
- **`curve_drift` recall is ~10% in both tiers**, which is itself the finding:
  schedule deviation raises variance, so it has a low per-order detection
  ceiling *by construction*. For VWAP, the aggregate tests carry more of the
  load than the exception queue. Anyone promising high per-order recall on
  curve-following failures is promising something the mathematics does not allow.

## Limits

- **This is the proxy version.** True tracking error needs `SUM_t (w_t - v_t)²`
  from per-fill timestamps and a market volume profile. `session_coverage` and
  the auction share are observable stand-ins for the term that matters most.
  With fill data this becomes a measured quantity rather than a proxy.
- **`PR` must be participation over the ORDER'S INTERVAL**, not over the day. If
  your `PR` is a full-day figure and the order spanned two hours, the dilution
  correction is understated. Worth confirming with whoever produces the extract.
- **The de-biasing assumes `PR` is your share of the benchmark window's volume.**
  If the interval VWAP is computed over a different window than `PR`, the
  algebra no longer lines up exactly.

## Knobs

`tier4_vwap/config.py` --- `debias_benchmark`, `max_dilution_participation`,
`k_spread` / `k_track`, taus, `include_size`, `algo_effect`, plus the shared
scoring and cause-rule settings.
