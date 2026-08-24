# Which columns the extract needs

The band is fitted on performance **already divided by the spread**, so `lo` and
`hi` come out in **spreads**, not bps. That column comes from the extract
pre-divided — this folder never does the division itself.

Seven columns are required to fit a band. Everything else is optional at fit
time but earns its place at score time.

---

## Strictly required — the year extract

| Your column | Maps to | Why |
|---|---|---|
| `aggrTgtId` | `order_id` | essential — row dropped if missing |
| `Strategy` | `algo` | essential; splits the cells **and** picks the metric column |
| `Sym` | `market` (last two characters) | essential; gives the region folder |
| `Pvwap` | `slippage_bps` | essential (see the wart below) |
| `Sprd` | `spread_bps` | essential (see the wart below) |
| `Date` | `order_date` | not essential, but effectively required — see below |
| **`ePvwap/Sprd`** *or* **`eIS/Sprd`** | `perf_in_spreads` | **the banded metric.** Which one depends on the strategy |

### The metric column is chosen per strategy

Strategies are benchmarked differently, and the two benchmarks must never share
a band — arrival slippage carries the market's drift over the order's life,
interval VWAP does not.

| Strategy | Column | Benchmark |
|---|---|---|
| `VWAP` | `ePvwap/Sprd` | interval VWAP |
| `PART`, `POV` | `eIS/Sprd` | arrival price |

That map lives in `config.py` → `METRIC_BY_STRATEGY`, matched
case-insensitively against the `Strategy` value. A strategy that isn't listed
falls back to `METRIC_COLUMN_DEFAULT` (`ePvwap/Sprd`) — and every entry point
prints a **fallback** warning when that happens, because a band fitted on the
wrong benchmark looks completely normal.

Adding a strategy is one line:

```python
METRIC_BY_STRATEGY = {
    "VWAP": "ePvwap/Sprd",
    "PART": "eIS/Sprd",
    "POV":  "eIS/Sprd",
    "TWAP": "ePvwap/Sprd",   # <- new
}
```

Nothing else changes: region, strategy and period all come from the data.

### The wart: `Pvwap` and `Sprd` are required but not banded

Both sit in `schema.ESSENTIAL`, so a row with a good `ePvwap/Sprd` but a blank
`Sprd` is still discarded — even though the band no longer touches either
column. They're kept because `Sprd` drives the drift report and `Pvwap` rides
into `outliers.csv` as the bps figure people expect to see next to the ratio.

This is **not silent**: the cleaning report counts every such row under
`missing essential`. Read that number. If it's large, say so and the requirement
can be made conditional on the chosen metric.

### `Date` — technically optional, practically not

It never enters the metric or the band. Its job is to label folders and stamp
the fit window into the band file. But without it:

- output folders become `unknown-period` instead of `2025-06_2026-05`;
- the fit window is never recorded, so **the leakage guard in `tier5/score.py`
  cannot fire** — `cells.windows_overlap` returns `False` the moment any bound
  is `None`, and scoring July against a band that already saw July would go
  through quietly and produce a circular flag rate.

That guard is the main reason the workflow is split in two. Include `Date`.

---

## Not required to fit, but recorded only at fit time

`%Adv`, `Vol`, `PR`, `Dur` — `persist.REFERENCE_FEATURES`.

Their medians are written into the band JSON **when the band is fitted**, and
that stamp is the only baseline the July drift report has. It separates the two
reasons a flag rate can move:

> the flag rate doubled because **the book got harder** (bigger `%Adv`, wider
> `Sprd`, higher `Vol`)

versus

> the flag rate doubled because **execution got worse**

Leave them out of the year extract and July still scores — you just can't tell
those apart, and you can't fix it later without refitting the year. A median
that moves more than 25% raises a warning saying the band was not fitted on
orders like these.

---

## Only matter for the scored period, not for the year

`$Mln`, `#Shares`, `Side`, `%POST`, `%OPEN` / `%CLOSE`, `Rev30min`.

These ride into `outliers.csv` as the diagnostic columns — the raw material for
explaining each flagged order. Harmless in the year file, wanted in the July
one. Missing ones are simply left out; nothing errors.

---

## Minimum viable header

For a VWAP book:

```
aggrTgtId,Strategy,Sym,Date,ePvwap/Sprd,Pvwap,Sprd
```

For a PART/POV book, swap in the arrival column:

```
aggrTgtId,Strategy,Sym,Date,eIS/Sprd,Pvwap,Sprd
```

The version actually worth exporting — one header that covers both, so a mixed
file works too:

```
aggrTgtId,Strategy,Sym,Date,ePvwap/Sprd,eIS/Sprd,Pvwap,Sprd,%Adv,Vol,PR,Dur
```

Then run

```bash
python check_extract.py year.csv
```

before anything else. Section **1b — BANDED METRIC** prints one row per
strategy naming the column its band will be built from, how many rows have no
value for it, and whether the strategy fell back to the default. That section is
the one to read closely: choosing the wrong source column is invisible
afterwards, because the band still fits and the curve still looks like a curve.

The other thing `check_extract` infers is the settings a header cannot express:
`SLIPPAGE_SIGN`, and whether `Vol`, `PR` and `%Adv` arrive as percent or
fraction. The sign is already settled for this extract — `Pvwap` is
`positive_is_good`, and `config.py` ships that way. The same convention is
applied to the spread-normalised column, since both come out of the same system.

---

## If you change `--metric`

| Metric | Units | Needs |
|---|---|---|
| `perf_in_spreads` *(default)* | spreads | the per-strategy column above |
| `slippage_bps` | bps | `Pvwap` only |
| `perf_norm` | sigma | `Sprd` **and** `Vol` **and** `Dur` |

`perf_norm` is the only one where the `Vol` / `PR` / `%Adv` unit settings move
the band itself; on the other two they affect the drift table only. Note also
that `DataConfig.vol_horizon_weight` is tuned for interval-VWAP benchmarks — if
you ever band `perf_norm` on an arrival benchmark, raise it toward 1.0.

**Bands are not interchangeable across metrics.** A band frozen on `slippage_bps`
is in bps; scoring it in the same run as a band frozen in spreads would put two
different units in one summary table. `tier5.score` refuses that — such a cell
is skipped with a reason naming both metrics, and the band file records its
units in `metric_units` so the artefact says which it is.
