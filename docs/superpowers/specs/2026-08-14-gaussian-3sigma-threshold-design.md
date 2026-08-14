# Tier 5 --- Gaussian 3-sigma threshold

**Date:** 2026-08-14
**Status:** approved, ready for planning

## The request

Add another way of producing a performance range for VWAP orders from pVWAP
slippage over one year, using the method the desk head asked for: *fit a normal
distribution and set the threshold at three standard deviations.*

This is the Shewhart / statistical-process-control control limit. In this
context:

1. Take pVWAP slippage for every VWAP order in the year: `x1 .. xn`.
2. Assume `x ~ Normal(mu, sigma^2)`; estimate `mu` and `sigma`.
3. The acceptable range is `mu - 3*sigma .. mu + 3*sigma`.
4. Outside the range is flagged and must be justified.

`k = 3` is not arbitrary. Under a normal distribution 99.73% of observations
fall within +/- 3 sigma, so the choice is a *promise about the flag rate*:
0.27% of orders, roughly 1 in 370. That is the method's appeal --- two numbers,
one formula, no model, checkable by hand.

That promise holds only if the data is actually normal. Execution slippage
generally is not: it is skewed with fat tails, and the ordinary standard
deviation is inflated by the very outliers the threshold is hunting. The
deliverable therefore includes the band **and the measurement of what the band
actually delivers**, so the result is "here is your 3-sigma range, and here is
what it does on our book" rather than an unverified number.

Note this is a distinct exercise from the argument already in the repository
against `mean +/- 2 sd`. The method is being built and measured, not
pre-judged.

## Scope

### Added

`tier5_gaussian/`, a self-contained tier folder in the same shape as the
others, driven by `python -m tier5_gaussian.run`. It consumes the shared
`tca.dataset.load_prepared`, so it scores the identical rows as
`tier3_model` and the comparison stays honest.

### Removed

| removed | reason |
|---|---|
| `tier1_fixed/` | superseded; the fixed-limit baseline is no longer wanted |
| `tier2_percentile/` | superseded |
| `tier4_vwap/` | superseded |
| `synthetic_vwap.py` | Tier 4's demo generator; already orphaned, nothing imports it |

`tier3_model/` is kept. It is the expected-cost estimation model, and it
generalises to strategies other than VWAP, which is why it survives the cull.

The dependency graph makes this safe: `tier3_model` imports nothing from
Tiers 1, 2 or 4 (the dependency runs the other way --- `tier4_vwap` imported
from `tier3_model`). `score_new.py` touches only `tier3_model` and is
unaffected. The only module that breaks is the top-level `run.py`, which
imports `tier1_fixed` and `tier2_percentile`.

### Naming

Folder names are **not** reflowed. `tier3_model/` keeps its name and its
`outputs/tier3/` paths, so `score_new.py`, the frozen
`outputs/tier3/model.json` and git history all stay valid. The new folder is
`tier5_gaussian/`. The resulting 3-then-5 gap is a scar from the deletion; the
README states in one line that the numbers are historical labels rather than a
ranking.

---

## Component design

```
tier5_gaussian/
  __init__.py
  config.py      Tier5Config: k, metric, estimator, level, min_group_n
  band.py        fit the band, classify orders, score a frame
  normality.py   coverage, shape statistics, required-k, QQ plot
  run.py         CLI driver
  README.md      method, settings, known weaknesses
```

Each module has one job and can be read without the others: `band.py` knows
nothing about normality testing, and `normality.py` produces evidence without
knowing how orders are scored.

### `config.py`

```python
@dataclass(frozen=True)
class Tier5Config:
    k_sigma: float = 3.0                    # the boss's 3
    metric: str = schema.SLIPPAGE_BPS       # raw pVWAP bps headline
    estimator: str = "classical"            # "classical" | "robust"
    score_level: str = "ALL"                # which fitted level scores orders
    min_group_n: int = 200                  # below this, a group is untrusted
    group_levels: tuple = ("ALL", "algo", "adv_bucket", "algo_x_adv_bucket")
    min_notional_review: float = 0.0        # materiality gate; 0 = off
    make_qq_plot: bool = True
```

`min_group_n = 200` matches the value Tier 2 used, for the same reason: a
sigma estimated from 44 orders is not a threshold.

### `band.py`

Two estimators, both computed for every group:

```
classical:  centre = mean(x)      scale = std(x, ddof=1)
robust:     centre = median(x)    scale = 1.4826 * median(|x - median(x)|)
band:       centre - k*scale  ..  centre + k*scale
```

The constant `1.4826` is `1 / Phi^-1(0.75)`, which makes the scaled MAD a
consistent estimator of sigma **under normality**. That is what makes the two
rows comparable: on truly normal data they agree, so any gap between them *is*
the non-normality, expressed in the band's own units.

**Four fitted levels**, matching how `tier3/threshold_table.csv` already
presents its bands:

| level | group key |
|---|---|
| `ALL` | every order --- the headline |
| `algo` | per strategy |
| `adv_bucket` | per %ADV bucket |
| `algo_x_adv_bucket` | the cross |

Every group appears in the output table. Groups with `n < min_group_n` are
marked `trusted = False` rather than dropped, so a thin cell is visible instead
of silently absent.

**Scoring** defaults to the `ALL` level, which is the request taken literally:
one range for the year. `--score-level algo_x_adv_bucket` scores against the
decomposition instead, falling back to `ALL` when the matched cell is
untrusted.

Output columns per order, reusing the existing vocabulary so downstream code
and the top-level comparison keep working:

| column | meaning |
|---|---|
| `band_centre`, `band_scale` | the fitted `mu` and `sigma` used |
| `band_lo`, `band_hi` | the threshold |
| `band_level` | which level supplied it |
| `zone` | `IN_RANGE` / `OUT_LOW` / `OUT_HIGH` / `NO_BAND` |
| `flagged` | zone in {OUT_LOW, OUT_HIGH} |
| `rank_stat` | `abs(x - centre) / (k * scale)` --- 1.0 means exactly at the limit |
| `material`, `review_required` | materiality gate, as in the other tiers |

`rank_stat` deliberately uses Tier 1's convention (1.0 at the limit) because
that is what lets `run.py` hold every method to a matched review budget.

Rows whose metric is missing or non-finite get `zone = NO_BAND` and are not
flagged.

**Known structural limit, stated in the tier README:** the band is symmetric
around the centre by construction. Slippage is not symmetric, so the two tails
will not be equally well served. This is a property of the method, not a bug,
and `normality.py` quantifies it per tail.

### `normality.py`

Three exhibits, ordered by how convincing they are.

**1. Tail coverage.** For `k` in 1, 2, 3, 4: what normal theory promises versus
what the book delivers. The `actual` columns below are **illustrative --- they
show the shape of the output, not measured values**; the promised column is
exact.

```
   k    promised inside   actual inside   promised outside   actual outside
  1.0       68.27%           76.4%             31.73%            23.6%
  2.0       95.45%           94.1%              4.55%             5.9%
  3.0       99.73%           98.44%             0.27%            1.56%   5.8x
  4.0       99.994%          99.61%             0.006%            0.39%   65x
```

Promised inside is `erf(k / sqrt(2))`; actual inside is
`mean(abs(x - centre) <= k * scale)`.

**2. Required k.** The single number that settles the discussion: the `k` at
which this book would actually flag 0.27%, computed as the 99.73rd percentile
of `abs(x - centre) / scale`. Reported alongside a per-tail decomposition ---
`k_lo = (centre - q_0.135) / scale` and `k_hi = (q_99.865 - centre) / scale`
--- which exposes the asymmetry the symmetric band cannot express.

**3. Shape statistics.** Skew, excess kurtosis, and D'Agostino K2
(`scipy.stats.normaltest`). Printed with an explicit caveat: at n around 12,000
**every** formal normality test rejects, so the p-value carries almost no
information. The effect sizes and the coverage table are the evidence; the test
is included because someone will ask for it.

**4. QQ plot** to `outputs/tier5/qq_plot.png`. A straight line means normal;
the curl at the ends is the fat tail. It is the one exhibit that needs no
statistics background to read.

Exhibits 1--3 are computed at the `ALL` level and additionally per `algo`. They
are not computed for the `adv_bucket` or `algo_x_adv_bucket` levels: the thin
cells there cannot support a tail estimate, and the table would be longer than
it is informative.

**Degradation is reported, never silent** --- the existing repository
convention. No `scipy`: skip K2, keep skew, kurtosis, coverage and required-k
(all computable with pandas/numpy). No `matplotlib`: skip the plot and print a
line saying so. Neither dependency sits in the scoring path.

### `run.py` (tier driver)

```bash
python -m tier5_gaussian.run --csv year.csv
python -m tier5_gaussian.run --csv year.csv --metric perf_norm
python -m tier5_gaussian.run --csv year.csv --k 2.5
python -m tier5_gaussian.run --csv year.csv --estimator robust
python -m tier5_gaussian.run --csv year.csv --score-level algo_x_adv_bucket
python -m tier5_gaussian.run --self-check
```

`--metric` accepts `slippage_bps` (default), `perf_in_spreads` and
`perf_norm`, so the identical method can be run on a normalized metric. The
default is raw bps because that is what "the range of performance" means to the
audience for this number, and it is a figure that goes on a slide.

Report sections, in order: cleaning, the band, the band table, the flag rate by
%ADV bucket, the coverage/required-k evidence, classical versus robust side by
side, and --- on synthetic data only --- detection against known truth.

---

## Data flow

```
CLI args
   |
   v
tca.dataset.load_prepared      (shared with tier3_model; identical rows)
   |
   v
band.fit(df, cfg)  ------------> band_table   (level x group, both estimators)
   |
   v
band.BandModel(table, cfg)
   |
   v
model.score_frame(df)  --------> scored_orders
   |
   +--> normality.evidence(df, cfg) --> coverage, required_k, shape stats
   |
   +--> normality.qq_plot(df, cfg)  --> qq_plot.png
```

The slippage sign is already normalized by `tca.pipeline` before this tier sees
it, so negative always means underperformance and no sign handling belongs
here.

## Outputs

```
outputs/tier5/band_table.csv       every level x group, both estimators, trusted flag
outputs/tier5/scored_orders.csv    per order: metric, band, zone, rank_stat
outputs/tier5/normality.csv        coverage + shape statistics, ALL and per algo
outputs/tier5/qq_plot.png          when matplotlib is available
```

`band_table.csv` carries both estimators as **columns on one row per group**,
so the classical and robust bands for a group are read side by side:

```
level  algo  adv_bucket  n  trusted
       centre_classical  scale_classical  lo_classical  hi_classical
       centre_robust     scale_robust     lo_robust     hi_robust
```

## Changes to existing files

| file | change |
|---|---|
| `run.py` | rewritten: compares `tier3_model` and `tier5_gaussian` instead of Tiers 1--3. Keeps both halves --- behaviour at each method's own threshold, and the matched-budget ranking comparison via `tca.evaluate.precision_at_budget` |
| `README.md` | substantial rewrite. Sections for Tiers 1, 2 and 4 removed; the tier table and the Evidence comparison become two-method; the Layout section updated; one line added explaining the historical numbering |
| `requirements.txt` | comments referencing Tiers 1 and 2 updated |
| `tca/`, `config.py`, `score_new.py`, `check_extract.py`, `distribution.py`, `synthetic_data.py` | unchanged |

`tca/evaluate.py` needs no change --- `precision_at_budget` is method-agnostic
and works with two methods exactly as it did with three.

## Verification

The repository has no test suite, and this design does not introduce one.
Correctness is proven where it can be proven, via a `--self-check` flag:

Illustrative output --- the shape of the check, not measured values:

```
$ python -m tier5_gaussian.run --self-check
  drew 200,000 samples from N(-8.7, 18.4)
  recovered   mean -8.698  sd 18.412             OK
  band        -63.93 .. +46.54                   OK
  flagged     0.269%   (theory 0.270%)           OK
  robust scale 18.39 vs classical 18.41          OK (ratio 1.00 on normal data)
```

The check draws from a normal distribution with a fixed seed and asserts four
things: the estimators recover the known parameters, the band matches the
closed form, the delivered flag rate matches 0.27% within sampling error, and
the classical and robust scales agree on normal data.

If all four hold, then any deviation observed on the real book is a property of
the data rather than a bug in the implementation --- which is precisely the
claim the entire report rests on.

Beyond the self-check, the acceptance criterion is that
`python run.py --csv <one year of orders>` runs end to end and produces the
head-to-head table, and that `python -m tier5_gaussian.run` reproduces the
headline range.

## Deliberately excluded

- **A third estimator** (trimmed or winsorized sigma). The robust MAD variant
  already makes the point that ordinary sigma is inflated by the tail; a third
  row adds length without adding an argument.
- **Cross-fitting.** The band is fitted and scored on the same book, exactly as
  the method was requested. This makes the in-sample flag rate partly
  circular, and the tier README says so --- `tier3_model` is where the
  out-of-sample number lives.
- **Freezing the band for future orders.** `score_new.py` applies the frozen
  Tier 3 surface. If the Gaussian band later needs the same treatment it is a
  small follow-on, but it is not part of this change.
- **Renumbering the surviving folders.** Cosmetic gain, real churn across
  `score_new.py`, output paths and the frozen `model.json` location.
