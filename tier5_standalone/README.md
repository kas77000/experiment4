# Tier 5 standalone — freeze a band, score the next period

Fit a Gaussian acceptable range on a year of orders, freeze it, then apply it
unchanged to a later month to find the orders that fall outside — and explain
why.

    RANGE = centre − k·scale  ..  centre + k·scale

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

**1. Inspect the extract first.** This tells you the two things a column name
cannot: which sign convention `Pvwap` uses and what the units are.

```bash
python check_extract.py your_file.csv
```

(It expects *vendor* column names. Run against a file written by the synthetic
demo it will report the essential columns as unresolved — that file already uses
the canonical names, so there is nothing to map. Not a fault, just don't use the
demo file to sanity-check this step.)

**2. Edit `config.py` → `COLUMN_MAP`** so it matches your column names. It ships
wired for `aggrTgtId` / `Strategy` / `Sym` / `Pvwap` / `Sprd` / `%Adv` / `Vol` /
`PR` / `Dur` / `Date`.

Three columns carry more weight than the rest here:

| Column | Becomes | Why it matters |
|---|---|---|
| `Sym` | region, from the last two characters | `0700 HK` → `HK`. Drives the folder split. |
| `Strategy` | strategy | Drives the folder split. |
| `Date` | the period label | Names the output folders and powers the overlap check. |

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
| `--metric` | `slippage_bps` | Also `perf_in_spreads`, `perf_norm`. |
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
| `--out-dir` | `outputs` | |

**`tier5.batch`** takes `fit` or `score`, plus `--dir` and all of the above.

**`tier5.run`** is the original single-run driver: fits and scores one book in
one go, useful for exploring. Its flag rate is in-sample and therefore partly
circular — use `fit` + `score` for a number you can defend.

---

## Choosing `k`

`k = 3` is a promise: under a normal distribution 99.73% of orders fall inside,
so you flag 0.27% — about 1 in 370. Real execution books don't keep that
promise. They are leptokurtic: more concentrated in the middle *and* fatter in
the tails than a normal, which is exactly the shape a two-parameter Gaussian
cannot represent.

Measured across five strategies on a synthetic book, every one of them flagged
roughly **6× what `k = 3` promised**, and each needed `k ≈ 4.5–5.0` to actually
deliver 0.27%.

Every band file records what would have worked:

```json
"reference": { "k_required": 4.61, "k_required_lo": 4.67, "k_required_hi": 4.41 }
```

Two defensible choices:

- **Keep `k = 3`** and accept a queue nearer 1.5%. The number is textbook and
  needs no defending.
- **Set `k` from `k_required`** across your twelve cells. One `k` serves all of
  them well, *provided* each cell gets its own centre and scale — which this
  tool does by construction.

Don't tune `k` per cell. The spread in required `k` between strategies is small;
the spread in required *centre and scale* is large. The per-cell bands already
handle that.

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
