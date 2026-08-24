# Tier 5 — how to use it

The Gaussian band: take a metric, assume it is normal, accept everything within
`k` scales of the centre, flag the rest.

    RANGE = centre − k·scale  ..  centre + k·scale

`README.md` in this folder covers the method and where it breaks. This file
covers running it.

---

## Quick start

```bash
# 0. Prove the code works. Runs the method on 200k draws from a known normal
#    and checks it recovers the parameters and the 0.27% flag rate.
python -m tier5_gaussian.run --self-check

# 1. Inspect a new extract BEFORE anything else. It tells you what to set for
#    the two things a column name cannot tell you: SLIPPAGE_SIGN and the units.
python check_extract.py your_file.csv

# 2. Edit config.py -> COLUMN_MAP so it matches your extract's column names.

# 3. Run.
python -m tier5_gaussian.run --csv your_file.csv
```

With no `--csv` at all it generates a 12,000-order synthetic demo book, which is
the fastest way to confirm the install is intact.

---

## The flags

| Flag | Default | What it does |
|---|---|---|
| `--csv PATH` | — | Your extract. Omit to use synthetic demo data. |
| `--metric` | `slippage_bps` | What gets banded. Also `perf_in_spreads` (slippage/spread) or `perf_norm` (slippage/sigma_expected). |
| `--k` | `3.0` | Scales either side of the centre. |
| `--estimator` | `classical` | `classical` = mean ± k·sd. `robust` = median ± k·1.4826·MAD. |
| `--score-level` | `ALL` | Which fitted band each order is scored against: `ALL`, `algo`, `adv_bucket`, `algo_x_adv_bucket`. |
| `--n`, `--seed` | `12000`, `7` | Synthetic book size and seed. Ignored with `--csv`. |
| `--self-check` | — | Prove the implementation on data with a known closed-form answer. |

Anything not on the command line lives in `tier5_gaussian/config.py`:
`k_sigma`, `metric`, `estimator`, `score_level`, `min_group_n`, `group_levels`,
`min_notional_review`, `make_qq_plot`.

---

## Outputs

Written to `outputs/tier5/`:

| File | What's in it |
|---|---|
| `band_table.csv` | One row per (level, group): `n`, both estimators' centre/scale/lo/hi, and `trusted`. |
| `scored_orders.csv` | Every order with its band, `zone`, `rank_stat`, `flagged`, `review_required`. |
| `normality.csv` | Per group: promised vs delivered coverage, the `k` you would actually need, skew, kurtosis. |
| `qq_plot.png` | Straight line = normal. The curl at the ends is the fat tail. |

`rank_stat` is `|x − centre| / (k·scale)`, so `1.0` sits exactly on the limit.
That convention is shared with the other tiers, which is what lets them be held
to a common review budget.

Zones: `IN_RANGE`, `OUT_LOW` (underperformance), `OUT_HIGH` (suspiciously good,
usually a data problem), `NO_BAND` (no trusted group, or metric missing).

---

## Choosing `--score-level`

This is the setting that matters most, and the default is wrong for any book
that mixes execution strategies.

`ALL` fits one band for the whole book. If every order is the same strategy in
the same market, that is what you want. If not, wider strategies eat the entire
review queue — not because they are worse, but because they are wider.

Measured on a 30k-order synthetic book with five strategies:

| strategy | flag % with `--score-level ALL` | flag % with `--score-level algo` |
|---|---|---|
| VWAP | 0.40% | 1.45% |
| TWAP | 1.06% | 1.62% |
| POV | 1.18% | 1.65% |
| IS | 3.29% | 1.61% |
| CLOSE | 4.47% | 1.73% |

An 11× spread collapses to 1.2×. **On a mixed book, use `--score-level algo`.**

Watch for `trusted=False` rows in `band_table.csv`: any group below
`min_group_n` (200) falls back to the global band, which puts that group back in
the first column above. `--score-level algo_x_adv_bucket` conditions harder but
runs out of data faster — the cells that most need their own threshold are
exactly the ones too thin to fit one.

---

## Choosing `--k`

`k = 3` is a promise: under a normal distribution, 0.27% of orders fall outside,
about 1 in 370. Real execution books are leptokurtic — more concentrated in the
middle *and* fatter in the tails — so they do not keep that promise.

Every strategy measured above flags roughly **6× what k = 3 promises**, and each
needs `k ≈ 4.5–5.0` to actually deliver 0.27%. That is not a VWAP-specific
finding; it is what slippage looks like.

Two ways to proceed, both defensible:

- **Keep `k = 3`** and accept a ~1.5% queue. The number is textbook and needs no
  defending to anyone.
- **Set `k` from the data.** Read `k_symmetric` from `normality.csv` and use it.
  On the book above, a single `k = 4.67` gave 0.19–0.36% across all five
  strategies — one `k` does serve them all, *provided* each gets its own centre
  and scale via `--score-level algo`.

Do not tune `k` per strategy. The spread in required `k` across strategies is
small; the spread in required *centre and scale* is large. Fix the grouping
first, and one `k` will do.

---

## Choosing `--metric` and `--estimator`

**Metric.** Leave it on `slippage_bps`. `perf_norm` divides by an expected-noise
scale and produces tighter bands, but on a mixed-strategy book it *widens* the
cross-strategy spread (5.5× vs 1.9× at a 0.27% target). Raw bps is also what
"the range of performance" means to whoever asked for it.

**Estimator.** Leave it on `classical` for scoring. `robust` flags 3.8–5.7%,
which is a much larger queue than most teams want.

`robust` earns its place as a *diagnostic*. Both estimators are always computed
and reported. On genuinely normal data they agree exactly — `--self-check`
confirms they match to three decimals on 200k draws — so any gap between them is
the non-normality, measured in the band's own units. A `sd / robust` ratio of
1.41 means the standard deviation is 41% larger than the tail-resistant estimate
of the same quantity: the band is being inflated by the very outliers it exists
to catch.

---

## Files this needs

Tier 5 does not need the rest of the repo. To run it somewhere else, copy:

```
config.py               the only file you edit for real data
synthetic_data.py       required — tca/dataset.py imports it at module level,
                        even when you pass --csv
check_extract.py        strongly recommended (step 1 above)
requirements.txt

tca/
    __init__.py  schema.py  pipeline.py  dataset.py  report.py  evaluate.py

tier5_gaussian/
    __init__.py  config.py  band.py  normality.py  run.py  README.md  USAGE.md
```

Not needed: `tier3_model/`, the top-level `run.py` (that is the tier3-vs-tier5
comparison driver and would pull tier3 in), `distribution.py`, `score_new.py`,
`other/`, `outputs/`, `docs/`.

Dependencies: `pandas`, `numpy`, `scipy`. `matplotlib` only for the QQ plot;
`statsmodels` and `seaborn` are tier3/`distribution.py` only. Nothing in the
scoring path imports matplotlib — it degrades gracefully and says so.

---

## Known limits

Read `README.md` for the full account. In short:

- **The band is symmetric; slippage is not.** Skew of −0.96 means the two tails
  are not equally served. `normality.csv` gives `k_lo` and `k_hi` separately;
  when they differ, no single `k` serves both.
- **Fitted and scored on the same orders.** The flag rate is partly circular —
  the band is dragged toward the outliers it then counts. Freezing a band on one
  period and applying it to the next is what turns the flag rate into a
  measurement.
- **No expected cost, so no aggregation.** A broker who is consistently slightly
  worse is invisible to a per-order band.
