# Tier 5 standalone — freeze a band, score the next period

Fit a Gaussian acceptable range on a year of orders, freeze it, then apply it
unchanged to a later month to find the orders that fall outside — and explain
why.

    RANGE = centre − k·scale  ..  centre + k·scale

The metric is performance **normalised by the spread**, so the bounds come out
in **spreads**, not bps. A 12 bps miss is noise in a wide Indian small cap and a
serious miss in a tight Japanese large cap; dividing by the spread puts every
name and every region on one scale before the band is fitted. The extract
supplies that column already divided — see
[EXTRACT_COLUMNS.md](EXTRACT_COLUMNS.md).

One band per **region × strategy**. Region comes from the `Sym` suffix, strategy
from the `Strategy` column, and the period from the `Date` column, so nothing
has to be typed on the command line and no run can be mislabelled.

This folder is self-contained. Copy it anywhere; it imports nothing outside
itself.

---

## Why two steps instead of one

Fitting and scoring the same book tells you almost nothing. If a band flags 1.7%
of the orders it was fitted on, it flags 1.7% partly because 1.7% was *defined*
as flagged — the band was dragged toward the very outliers it then counts.

Freeze the band and apply it to orders it has never seen and the flag rate stops
being a definition and becomes a **measurement**. If July flags 4% against a
band that flagged 1.5% on the fit year, something real changed. That is the
whole point of `fit` then `score`.

---

## Install

```bash
pip install -r requirements.txt
```

`matplotlib` and `seaborn` are only used for the pictures. Without them
everything still fits and scores — the curve step prints that it was skipped.

## Prove it works before pointing it at real data

```bash
python -m tier5.run --self-check
```

Runs the method on 200,000 draws from a known normal and checks it recovers the
parameters, the closed-form band, the 0.27% flag rate, and that the two
estimators agree. If those pass, any oddity you see on a real book is the data,
not the code.

Then watch the whole workflow run on a synthetic book:

```bash
python -m tier5.fit --n 6000
```

---

## Pointing it at your data

**Which columns to export:** see [EXTRACT_COLUMNS.md](EXTRACT_COLUMNS.md) —
which seven are required, which spread-normalised column each strategy bands,
which are needed only in the scored period, and which are recorded at fit time
and cannot be backfilled later.

**1. Inspect the extract first.** This tells you the things a column name
cannot: which spread-normalised column each strategy will be banded on (section
**1b**), which sign convention the performance columns use, and what the units
are.

```bash
python check_extract.py your_file.csv
```

(It expects *vendor* column names. Run against a file written by the synthetic
demo it will report the essential columns as unresolved — that file already uses
the canonical names, so there is nothing to map. Not a fault, just don't use the
demo file to sanity-check this step.)

**2. Edit `config.py` → `COLUMN_MAP`** so it matches your column names. It ships
wired for `aggrTgtId` / `Strategy` / `Sym` / `Date` / `ePvwap/Sprd` / `eIS/Sprd`
/ `Pvwap` / `Sprd` / `%Adv` / `Vol` / `PR` / `Dur`.

Four columns carry more weight than the rest here:

| Column | Becomes | Why it matters |
|---|---|---|
| `Sym` | region, from the last two characters | `0700 HK` → `HK`. Drives the folder split. |
| `Strategy` | strategy | Drives the folder split, **and picks the metric column**. |
| `Date` | the period label | Names the output folders and powers the overlap check. |
| `ePvwap/Sprd` or `eIS/Sprd` | the banded metric | Which one is chosen per strategy by `METRIC_BY_STRATEGY`. |

`METRIC_BY_STRATEGY` maps VWAP to the interval-VWAP column and PART/POV to the
arrival column. Those two benchmarks must never share a band, which is why the
choice is made per strategy and printed on every run — a band on the wrong
benchmark still fits and still looks like a curve.

`Date` is optional. Without it, periods fall back to `--label` and the overlap
check is skipped — you lose the safety net, not the results.

Regions in `config.REGION_NAMES`: `AU`, `HK`, `JP`, `IN`. A suffix that isn't
listed still fits and scores normally; it's reported as unrecognised rather
than dropped, so a stray venue shows up instead of vanishing.

---

## The two-step workflow

```bash
# 1. Fit the year and freeze the bands
python -m tier5.fit --csv extracts/year.csv

# 2. Score a later period against them
python -m tier5.score --csv extracts/july.csv --label 2026-07
```

If the cells arrive as separate files, point the batch driver at the directory
instead — it walks it recursively and does all twelve:

```bash
python -m tier5.batch fit   --dir extracts/year/
python -m tier5.batch score --dir extracts/july/ --label 2026-07
```

Both work the same whether one file holds every region and strategy or twelve
files hold one each. The cells come from the data, not the filenames.

---

## What you get

```
bands/
  HK/VWAP.json                    the frozen range — this is the artefact
  HK/TWAP.json
  JP/VWAP.json                    ...

outputs/
  fit/2025-06_2026-05/HK/VWAP/
      curve.png                   the band drawn over the year's distribution
      normality.csv               does 3σ mean what it says on this book?
  score/2026-07/HK/VWAP/
      outliers.csv                <- open this one first
      scored.csv                  every order, with its zone
      curve.png                   July's shape against the frozen band
  score/2026-07/_summary.csv      all twelve cells side by side
```

### Reading `outliers.csv`

One row per order outside the band, **worst first**:

| Column | What it is |
|---|---|
| `n_sigma_outside` | How far past the bound, in scales. Always positive, whichever tail. Because each cell divides by its own scale, this is comparable across regions and strategies. |
| `zone` | `OUT_LOW` = underperformance. `OUT_HIGH` = suspiciously good, usually a data problem. |
| `band_lo` / `band_hi` | The frozen bounds this order broke. |
| everything else | `spread_bps`, `pct_adv`, `participation`, `duration_min`, `passive_fill_pct`, `auction_pct`, `reversion_bps` … |

Those diagnostic columns are there to support **your** explanation. This tool
deliberately does not attribute cause — it tells you which orders need an
explanation, not what the explanation is.

### The curve

`curve.png` puts two shapes on one axis: a KDE of what the orders actually did,
and the dashed normal the band assumes they did. **The gap between them is the
non-normality.** A real execution book is more peaked in the middle and fatter
at the ends than a normal — that is what makes `k = 3` flag more than it
promises. The view is framed on the band plus the near tails so a handful of
extreme orders can't squash it; anything beyond the frame is noted in the
caption and still counted in every number.

---

## The flags

**`tier5.fit`**

| Flag | Default | What it does |
|---|---|---|
| `--csv PATH` | — | The extract. Omit for a synthetic demo book. |
| `--k` | `3.0` | Scales either side of the centre. |
| `--target-review-count N` | off | Solve for `k` per cell so about `N` orders **per month** fall outside. Overrides `--k`, which becomes a floor. Say it in orders when the answer is "a couple a month". |
| `--target-flag-rate PCT` | off | Solve for `k` per cell so exactly `PCT`% of the fit book falls outside. Overrides `--k`. |
| `--metric` | `perf_in_spreads` | Also `slippage_bps` (bps), `perf_norm` (sigma). |
| `--estimator` | `classical` | `classical` = mean ± k·sd. `robust` = median ± k·1.4826·MAD. |
| `--bands-dir` | `bands` | Where band JSON is written. |
| `--out-dir` | `outputs` | Where curves and evidence go. |
| `--force` | off | Fit cells below `min_group_n` (200) anyway. |
| `--n`, `--seed` | `12000`, `7` | Synthetic book size and seed. |

**`tier5.score`**

| Flag | Default | What it does |
|---|---|---|
| `--csv PATH` | — | The later period's extract. |
| `--bands-dir` | `bands` | Where to look up frozen bands. |
| `--label` | from `Date` | Names the period folder. |
| `--min-notional-review AMOUNT` | frozen with the band | Only queue flagged orders at least this large. The band does not move; every flagged order still appears in `outliers.csv`, marked `review_required`. |
| `--out-dir` | `outputs` | |

**`tier5.batch`** takes `fit` or `score`, plus `--dir` and all of the above.

**`tier5.run`** is the original single-run driver: fits and scores one book in
one go, useful for exploring. Its flag rate is in-sample and therefore partly
circular — use `fit` + `score` for a number you can defend.

---

## Choosing `k` — or not choosing it at all

`k = 3` is a promise: under a normal distribution 99.73% of orders fall inside,
so you flag 0.27% — about 1 in 370. **Real execution books do not keep that
promise.** They are leptokurtic: sharply concentrated in the middle *and* far
fatter in the tails than a normal — exactly the shape a two-parameter Gaussian
cannot represent. Open any `curve.png` and the gap between the filled shape and
the dashed line is that failure, drawn.

On a real HK VWAP book of 47k orders (`sd = 2.67`), `k = 3` put **2.42%**
outside — nine times what it advertised, about 95 orders a month to explain.

There are exactly two honest responses, and the choice between them is a
resourcing decision, not a statistical one.

**Keep `k = 3` and accept the real rate.** The number is textbook and needs no
defending. You just cannot also claim it flags 0.27%.

**State the load you want and report the `k` it took.** Two flags do this, and
the difference between them matters more than it looks.

#### `--target-review-count N` — say it in orders (recommended)

The question a desk head actually asks is *"how many of these will land on my
team each month?"* — a number of orders, not a percentage. This flag takes that
number directly, per cell, per month:

```bash
python -m tier5.fit --csv year.csv --target-review-count 2
```

Each cell converts your budget into its own rate using its own volume and its
own fit window —

```
rate = orders_per_month × months_in_fit_window ÷ n
```

— then solves for the `k` that delivers it. On the 47k HK VWAP book, `2` gives:

```
RANGE      -14.37 .. 13.66 spreads
k            5.26      <- what 2/month cost on this cell
an order must miss by 14.0 spreads to be flagged
cut on the 24 most extreme order(s) of the fit book
in-sample flagged: 0.05%
```

That is **99.95% coverage** — almost every order inside — with roughly two a
month to explain. Simulating 600 fresh months against that frozen band: mean
2.0 a month, 90% of months between 0 and 5, 16% of months completely clean.

**Why a count and not a rate.** A rate is only a workload once you know the
volume. `0.5%` is twenty orders a month on a 47k book and one a quarter on a
thin one, so a single rate across twelve cells hands the busy desks all the
work. A count gives every desk the same load, which is what "we can explain two
a month" actually means.

**Two guards, both load-bearing:**

- **`--k` becomes a floor.** On a thin cell the budget can imply a rate *wider*
  than nominal — 2 a month out of 400 orders a year is 6% — which would give the
  small desk a **tighter** band than the large one, exactly backwards. The
  budget may only ever widen a band. When the floor catches a cell it says so:
  `k  3.00  <- HELD AT THE FLOOR`, with the reason.
- **Tail evidence is reported.** A bound at the 99.95th percentile is cut on the
  handful of orders beyond it. `cut on the 24 most extreme order(s)` is enough
  to freeze; below ten the fit prints `THIN TAIL` and tells you to fit a longer
  window or raise the budget.

The band file records the whole derivation, so `k = 5.26` is never somebody's
guess six months from now:

```json
"k_sigma": 5.26, "k_source": "target_review_count", "target_review_count": 2.0,
"review_budget": {"rate": 0.000511, "n_tail": 24, "floored": false,
                  "thin_tail": false}
```

#### `--target-flag-rate PCT` — say it as a share of the book

Use this when the standard genuinely is a percentage, or when the cells are
similar enough in volume that a rate and a count say the same thing. It
solves for `k` per cell the same way:

```bash
python -m tier5.fit --csv year.csv --target-flag-rate 0.5
```

What that costs on a book shaped like the one above:

| Target | `k` needed | Band | Per year | Per month |
|---|---|---|---|---|
| 2.42% (`k = 3`) | 3.00 | −8.26 .. 7.76 | 1,136 | 95 |
| 1.00% | 3.61 | −9.88 .. 9.38 | 470 | 39 |
| 0.50% | 4.03 | −11.02 .. 10.52 | 235 | 20 |
| 0.27% | 4.39 | −11.96 .. 11.46 | 127 | 11 |

Note how little the band has to widen. Because the tail is heavy, going from
`k = 3` to `k ≈ 4` cuts the queue five-fold — the orders being dropped are the
dense inner tail, not the genuinely extreme ones.

What is **not** honest is calling a band "3 sigma" while its tails mean
something entirely different from what that phrase implies.

The two flags are mutually exclusive — setting both is refused rather than
silently resolved, because they choose the same bound from different directions.

### The catch with per-cell `k`

Both flags give every cell the same *load*, which means a cell with fatter tails
gets a wider band. Read that twice: **a region with
genuinely worse execution is handed a more forgiving bound.** That is right if
you are allocating scarce review capacity evenly, and wrong if you want one
absolute standard across regions. If you want the latter, fit once with
`--target-flag-rate`, read the solved `k` values, and then refit everything with
a single `--k`. The band file records which you did:

```json
"k_sigma": 4.03, "k_source": "target_flag_rate", "target_flag_rate": 0.5
```

Every band file also records what `k` *would* have delivered 0.27%, whichever
route you took:

```json
"reference": { "k_required": 4.61, "k_required_lo": 4.67, "k_required_hi": 4.41 }
```

### Do not reach for `--estimator robust` here

It moves the band the wrong way. On a book with a sharp central spike the MAD is
*small*, so `median ± k·1.4826·MAD` is **narrower** than the classical band and
flags **more**, not fewer. Robust estimators resist outliers dragging the scale
out; they do not widen a band. It remains the right choice when a handful of
data errors are inflating `sd`, which is a different problem.

### The second lever: materiality

Widening the band is not the only way to shrink the queue. Being outside the
band and being worth an analyst's hour are different questions — a 3-spread miss
on a $40k order costs almost nothing to fix.

```bash
python -m tier5.score --csv july.csv --min-notional-review 5000000
```

On the same book: 95 flagged a month at `k = 3`; 20 with the band calibrated to
0.5%; **7** once a $5m materiality gate is applied. Nothing is hidden — every
flagged order stays in `outliers.csv` with a `review_required` column, so the
record is complete and the queue is short.

The gate is frozen with the band so a rescore reproduces the same queue, but the
flag above overrides it without refitting: the band is a measurement and must not
move, while how much of the queue you work is a policy that can change.

---

## Guard rails, and why they're there

**Scoring a period the band already saw is refused.** If the `Date` range of the
scoring file overlaps the band's fit window, `score` stops and says so. That
overlap is the one mistake that makes the whole exercise meaningless, and it is
easy to make by exporting the wrong window. In batch mode it's reported once as
a systemic problem rather than twelve times as separate failures.

**A cell with no band file is skipped, never substituted.** Scoring Australian
IS orders against the Japanese TWAP band would produce plausible-looking numbers
with no warning. Skipping is worse than nothing; substituting is worse than
skipping.

**Cells under 200 orders are not fitted** unless you pass `--force`. A sigma
estimated from 40 orders is not a threshold. Skipped cells are listed, not
silently dropped.

**Drift is reported at score time.** If a feature median has moved more than 25%
against the fit book, `score` says so. A frozen band decays silently otherwise,
and this is what separates "the market changed, recalibrate" from "execution
changed, that's a finding."

---

## Known limits

- **The band is symmetric; slippage is not.** `mu ± k·sigma` is the same distance
  either side of centre, but execution slippage is skewed. `normality.csv` gives
  `k_lo` and `k_hi` separately — when they differ, no single `k` serves both
  tails equally.
- **Two parameters, no difficulty adjustment.** A band on raw bps does not know
  that a 30%-of-ADV order is harder than a 0.5% one. Within a single region and
  strategy that matters less than it would across a whole book, but it doesn't
  vanish.
- **No expected cost, so no aggregation.** A broker who is consistently
  *slightly* worse never breaks a per-order band and is invisible here.
- **Bands go stale.** Refit when the drift report says the feature medians moved.

---

## Tests

```bash
python -m pytest tests/ -v
```

69 tests. The ones that matter most: `fit` → `score` round-trips without the
band moving, a wider later period flags more *without* widening the band, and
scoring the fit book against its own band is refused.
