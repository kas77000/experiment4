# TCA Performance Thresholds

**The question:** out of tens of thousands of executed orders, which ones went
badly enough to deserve a human's attention --- and why?

Every order gets a performance **range**. Inside the range is acceptable.
Outside it is flagged and must be justified. The range is two-sided on purpose,
so both unusually *bad* orders and suspiciously *good* ones surface --- a result
that looks too good is usually a data or benchmark error, and you want to know
about those too.

The same problem is solved three ways, in increasing order of what the market
actually does. Each is self-contained in its own folder, and all three run on the
identical set of orders so the comparison is honest.

| | question it answers | folder |
|---|---|---|
| **Tier 1** | "did this order break the policy limit?" | `tier1_fixed/` |
| **Tier 2** | "was this order in the worst 2% of similar orders?" | `tier2_percentile/` |
| **Tier 3** | "did this order cost more than expected *for an order like it* --- and what went wrong?" | `tier3_model/` |

---

## Running it

```bash
pip install -r requirements.txt
```

**Step 1 --- check the file before trusting any number it produces.**

```bash
python check_extract.py your_file.csv
```

This reads your extract, confirms every column mapped correctly, and verifies
the settings that a column *name* cannot tell you (whether `Pvwap` is a gain or a
cost, and whether `Vol`/`%Adv`/`PR` are percentages or fractions). It prints the
exact settings to use. Accepts `.csv`, `.xlsx` and `.parquet`.

**Step 2 --- run it.**

```bash
python run.py --csv your_file.csv               # all three tiers, side by side
python -m tier3_model.run --csv your_file.csv   # the full Tier 3 report
```

The other two individually, if you want their detail:

```bash
python -m tier1_fixed.run --csv your_file.csv
python -m tier2_percentile.run --csv your_file.csv
```

Useful variants:

```bash
python run.py --csv your_file.csv --budget 2          # size the review queue at 2%
python -m tier3_model.run --csv your_file.csv --examples 10   # narrate 10 worst orders
python -m tier3_model.run --csv your_file.csv --algo-effect expose   # compare algos
```

**Step 3 --- apply the threshold to future orders.**

```bash
python score_new.py next_quarter.csv
```

Step 2 *fits* the threshold and freezes it to `outputs/tier3/model.json`. Step 3
applies that frozen surface to orders it has never seen, with no refitting. This
is the point of the whole exercise --- see "Using it on future data" below.

Running with no `--csv` uses a built-in synthetic book, which is how every number
in this README was produced.

---

## Reading the results

Everything lands in `outputs/tier1/`, `outputs/tier2/`, `outputs/tier3/`.

| file | what it is | when to open it |
|---|---|---|
| `tier3/review_queue.csv` | the orders to look at, worst first, each with a cause and a cash cost | **every run** |
| `tier3/slice_findings.csv` | systematic problems by algo, market, size | **every run** |
| `tier3/threshold_table.csv` | the fitted band per group, in bps and in spreads | when someone asks "what is the threshold?" |
| `tier3/calibration.csv` | promised vs delivered coverage | before trusting anything |
| `tier3/scored_orders.csv` | every order with its own band, z, zone, cause | drilling into specifics |
| `tier3/model.json` | the frozen threshold | consumed by `score_new.py` |
| `tier3/model_coefficients.csv` | the fitted formula | handing the method to someone |

### The columns that matter

For any order:

| column | meaning |
|---|---|
| `expected_bps` | what an order like this *should* have cost |
| `band_lo_bps` / `band_hi_bps` | **its threshold**, in bps. Outside this is flagged |
| `band_lo_spreads` / `band_hi_spreads` | the same threshold in **spread multiples** |
| `residual_bps` | actual minus expected --- the miss |
| `shortfall_ccy` | that miss in **actual USD** |
| `z` | the miss in units of the dispersion expected for *this* order |
| `zone` | `IN_RANGE` / `OUT_LOW` / `OUT_HIGH` |
| `severity` | `OK` / `MONITOR` / `ESCALATE` |
| `likely_cause` | what went wrong |
| `remedy` | what to change |

**Where is the threshold?** There is no single one. Every order gets its own,
which is exactly what Tiers 1 and 2 cannot do. Per order it is
`band_lo_bps`..`band_hi_bps`; summarised it is `threshold_table.csv`:

```
algo  adv_bucket     n  expected_bps  band_lo_bps  band_hi_bps  band_lo_p10  band_lo_p90  band_lo_spreads
VWAP        <1%  2476         -5.56       -61.18        41.97      -112.17       -33.42            -5.72
VWAP       1-5%  3430        -10.94       -66.89        35.55      -122.21       -36.39            -6.18
VWAP      5-10%   470        -19.68       -78.16        26.92      -130.27       -43.23            -7.24
VWAP     10-20%   143        -25.86       -79.43        16.33      -140.54       -47.26            -7.51
VWAP       >20%    20        -45.17      -105.01         2.51      -188.34       -65.48           -10.61
```

Read it as: *a VWAP order under 1% ADV is expected to lose about 5.6 bps, and is
acceptable anywhere between −61 and +42 bps. The same algo above 20% ADV is
expected to lose 45 bps and gets −105 to +2.5.*

`band_lo_p10` and `band_lo_p90` are the important pair: within the `<1%` cell
alone the lower threshold ranges from **−112 to −33 bps** depending on each
order's spread, volatility and duration. Tier 2 would put a single number there
for all 2,476 orders. That range **is** the resolution Tier 3 buys.

The band is fitted in units of `sigma_expected`, not of spread --- that is the
volatility argument --- so `band_lo_spreads` is a presentation of the fitted
band, not a second threshold. It varies per order too: the same band is a
different number of spreads on a fast order than on a slow one.

### The one number to check first

```
                  bound       promised   delivered
   below the lower band          1.0%       1.08%
   above the upper band          0.5%       0.58%
```

Out-of-sample. Delivered close to promised means the threshold will hold on next
quarter's orders. Far *above* means the model is missing a driver; far *below*
means it is overfitted and will stop flagging. Read this before the queue.

### Reading a single order

```
Order HK30005594  |  VWAP_Passive  |  0103 HK  buy
  size 1.04% ADV   POV 7.5%   58 min   spread 12.2bps   vol 236bps/day
  actual -165.5 bps   vs expected -2.2 bps   (band -69.9 .. +52.9 bps)
  in spreads: actual -13.57  vs expected -0.18  (band -5.73 .. +4.34 spreads)
  shortfall -163.3 bps   z = -6.52   -> OUT_LOW / ESCALATE
  expected cost driven by: algo=VWAP_Passive +6.1bps, sqrt_adv +4.2bps
  evidence: reversion 1.9x sigma (p94); passive fill 31% (p8 in algo)
  LIKELY CAUSE: over_aggressive
  ACTION: Cap participation / lengthen the horizon on this profile.
```

An easy order that should have cost 2 bps cost 165. The price snapped back after
the last fill and passive fill was in the bottom decile for this algo --- it
pushed, the market moved, and it gave the money away. That is a queue entry
someone can act on, not a number to argue about.

### The management view

```
                   orders   mean_z   mean_shortfall_bps    total_usd
over_aggressive        36    -3.47                -78.8   -2,173,571
unexplained            38    -4.01                -81.7   -1,515,295
missed_close           35    -3.52                -78.1   -1,477,404
spread_bleed           20    -3.19                -81.1     -889,639
```

Ranked by money, not order count. `unexplained` sitting high is normal and
usually means marks or benchmark windows to verify, not execution to fix.

---

## Using it on future data

The threshold is only worth having if it can be **frozen and reused**. Fitting
and scoring the same book is close to circular: 1.5% flags because 1.5% was
*defined* as flagging. Applied to unseen orders, that number becomes a
measurement.

```bash
python -m tier3_model.run --csv 2025_h1.csv     # fit once -> outputs/tier3/model.json
python score_new.py 2025_q3.csv                 # apply it, no refitting
python score_new.py 2025_q4.csv                 # and again
```

`model.json` holds everything needed to reproduce the surface exactly --- not
just the coefficients, but the standardization statistics, the imputation
medians and the dummy levels. Coefficients alone would silently rescale every
feature on the next run, and nothing would error.

### The drift block

A frozen threshold decays quietly. Nothing breaks; the numbers just stop meaning
what they did. So `score_new.py` reports whether the new book still resembles the
one the threshold was fitted on:

```
                  check  training  new_data  nominal  change_pct
0           flag rate %     1.700     1.650      1.5         NaN
1        median pct_adv     1.358     1.395      NaN         2.7
2     median volatility   179.600   181.300      NaN         0.9
3     median spread_bps    10.050     9.890      NaN        -1.6

  No drift worth acting on. The threshold still fits this book.
```

This separates the two reasons a flag rate can move, which call for **opposite**
responses:

**Execution changed** --- features stable, flag rate jumped. A real finding: act
on it.

```
  flag rate %   1.700 -> 7.810     (features all within 3%)
  WARNING: Flag rate 7.8% is well above the 1.5% the gate was set to.
```

**The market changed** --- features moved. Recalibrate; the threshold was not
fitted on orders like these.

```
  median volatility   179.6 -> 326.3   (+82%)
  WARNING: volatility median moved +82% vs the training book.
  WARNING: Flag rate 0.5% is far below the 1.5% nominal. The threshold has
           gone slack and is no longer catching much.
```

**A new algo or market appears** --- called out separately, because these fall
into the model's baseline category and get scored against the wrong intercept.
That produces plausible-looking numbers with no warning at all, so it is checked
explicitly:

```
  WARNING: 800 orders have a algo not present when the model was fitted
           (['VWAP_Dark']). They are scored against the baseline category,
           so their thresholds are unreliable -- refit to include them.
```

Refit when the market moves or the book changes shape; otherwise keep the
threshold fixed, so a change in the flag rate means something.

---

## Why this is harder than it looks

The obvious approach --- flag any order worse than, say, 50 bps --- fails for a
reason that shows up immediately in real data. Compare two orders:

| | order A | order B |
|---|---|---|
| slippage vs VWAP | **-40 bps** | **-12 bps** |
| size | 22% of daily volume | 0.3% of daily volume |
| spread | 45 bps | 4 bps |
| worked over | 5 hours | 8 minutes |

Order A looks five times worse. In reality A is a *good* execution of a very hard
order and B is a *poor* execution of an easy one. A fixed limit flags A and
ignores B --- exactly backwards.

Two separate corrections are needed, and they are independent:

**1. Put every order on a common scale.** Raw bps are not comparable across
names. The natural size of a slippage number depends on the spread you have to
cross and on how much the price can move while you work the order:

```
sigma_expected = sqrt( (0.5 x spread)^2  +  (0.20 x volatility x sqrt(T))^2 )
```

where `T` is the order's duration as a fraction of the trading session. For a
small fast order the spread term dominates. For an order worked over hours the
volatility term dominates. Adding them in quadrature lets each take over where it
should, with nothing to hand-tune. Dividing slippage by this gives a
dimensionless number where 1.0 means "one typical move for an order like this".

**2. Adjust for how hard the order was.** Even on a common scale, a 22%-ADV
order is *supposed* to cost more than a 0.3% one. Tier 2 does this by grouping
into size buckets; Tier 3 models it directly. This is the difference between
measuring difficulty and measuring quality.

---

## Tier 1 --- Fixed thresholds

**One limit, applied to every order.**

```
flag if  |slippage|  >  50 bps
```

Three variants are available: an absolute bps limit, a multiple of the spread, or
a multiple of `sigma_expected`. Plus a materiality gate, so no one writes a memo
about a tiny order.

**What it gets right.** It is trivially auditable, a compliance officer can check
it by hand, and it needs no model to defend to a regulator. Most best-execution
policies still say something like this in writing, and that is a legitimate
choice.

**Where it breaks.** The limit does not scale with difficulty, so it measures
order size rather than execution quality:

| order size (%ADV) | orders | % flagged |
|---|---|---|
| under 1% | 4,566 | 4.2% |
| 1-5% | 6,258 | 5.4% |
| 5-10% | 863 | 8.9% |
| 10-20% | 245 | **17.6%** |
| over 20% | 44 | **27.3%** |

One order in four above 20% ADV is flagged; one in twenty-five below 1% ADV.
Traders learn this within a quarter, and the report stops being read --- every
flag has the same answer ready ("it was a big order").

**Cheapest improvement:** switch the rule to `sigma_multiple`. Still one fixed
number, but the yardstick now adapts to each order. Costs nothing, recovers most
of the gap to Tier 2.

---

## Tier 2 --- Percentile bands within peer groups

**Rank each order against comparable orders; flag the tails.**

1. Group comparable orders: same algo, same market, same size bucket.
2. Within each group, take the 2nd and 98th percentiles of performance. Those two
   numbers are the band.
3. If a group has too few orders to be reliable (under 200), fall back to a wider
   grouping: `algo x market`, then all orders.

This is the vendor approach --- Abel Noser, Virtu/ITG and the other TCA universes
sell exactly this, with the peer group drawn from a cross-industry database
rather than your own history. Pass `--peer-csv` and this code does the same thing
against a street extract, so you are ranked against the market instead of against
your own past.

**Why percentiles rather than "mean plus 2 standard deviations".** Execution cost
distributions are skewed with fat tails. The mean and the standard deviation are
both dragged around by the very outliers you are hunting --- a handful of
disasters widens the band until it stops catching anything. Percentiles are
robust: a few extreme values cannot move the 2nd percentile. They also produce a
naturally lopsided band, which matches the real shape of the data, whereas
"mean +/- 2 sd" forces a symmetry that is not there.

**What it gets right.** The threshold now adapts to the peer group. The size
gradient collapses from 23 percentage points to 3.

**Where it breaks.**

- **The band is a staircase.** Every order in the 1-5% bucket shares one
  threshold, so a 1.1% order and a 4.9% order are held to the same standard, and
  the threshold jumps at the bucket edge.
- **You run out of data fast.** Each additional conditioning variable multiplies
  the number of groups. With a 200-order minimum you can afford two or three
  before the groups are too thin to trust. Tier 3 conditions on six things at
  once.
- **It grades on a curve.** The bands are fitted on the same orders being scored,
  so 4% is flagged because 4% was *defined* as flagged. The number is guaranteed,
  not earned.
- **No expected cost, so no aggregation.** You cannot average a percentile rank
  into "this broker is consistently worse". Systematic problems stay invisible.

**One number worth knowing:** the default percentile choice determines the
workload before you see any data. p10/p90 flags 20% of the book *by construction*
--- unworkable if a human justifies each one. This is set to p2/p98 (~4%), which
is where production exception reports actually run.

---

## Tier 3 --- Expected-cost model and residual scoring

**Predict what each order should have cost, then measure the gap.**

This is what serious TCA does --- the approach behind Kissell's I-Star model,
Bloomberg's BTCA, BestX and the Barra/MSCI impact models. Instead of asking "was
this order in the worst 2% of its bucket?", it asks "what should *this specific
order* have cost, and how far off was it?"

### Step 1 --- Model the expected cost

Fit a formula predicting performance from the order's characteristics:

| input | why it is there |
|---|---|
| **sqrt(size as %ADV)** | the square-root law: the most reproducible finding in execution research is that cost grows with the *square root* of order size, not linearly |
| **participation rate** | urgency. Taking 30% of the volume costs more per share than taking 3% |
| **duration** | how long you were exposed |
| **spread** | liquidity tier |
| **volatility** | how much the price could move against you |
| **size x urgency** | a big order traded *fast* is worse than the two effects added separately |
| **algo, market, side** | each strategy and venue gets its own baseline |

The technique is **quantile regression**. Ordinary regression predicts an
average; quantile regression predicts a *percentile*. Fitting it three times ---
at the 2nd, 50th and 99th percentiles --- produces, for every individual order, a
predicted median (the expected cost) and a predicted band around it. It is the
Tier 2 idea with the bucket staircase replaced by a smooth surface, and with six
conditioning variables instead of two.

### Step 2 --- Score the residual

```
z  =  (actual performance  -  expected performance)  /  expected dispersion
```

A **z-score** says how far the order landed from what was expected *of it*,
measured in units of the variation expected *for it*. This is the number that
makes order A and order B from the earlier example finally comparable: a -40 bps
fill on a huge illiquid order and a -12 bps fill on a small liquid one can now
produce the same z, or very different ones, and either way the comparison means
something.

Flags are then tiered, because a flag costs analyst time:

| | condition | action |
|---|---|---|
| **OK** | inside the band | nothing |
| **MONITOR** | outside the band | logged and trended |
| **ESCALATE** | z beyond 3 | written justification |

The default band is deliberately lopsided --- 1% on the downside, 0.5% on the
upside --- because underperformance is what you act on, while the upper gate
mainly catches data errors. Total queue: **~1.7% of the book**.

**The flag rate is a dial, not a finding.** It is set by the percentile choice
and you get back what you dial in, near-exactly:

| setting | nominal | actually flagged |
|---|---|---|
| 0.02 / 0.99 | 3.00% | 3.12% |
| **0.01 / 0.995** *(current)* | **1.50%** | **1.77%** |
| 0.005 / 0.9975 | 0.75% | 0.89% |
| 0.0025 / 0.999 | 0.35% | 0.62% |

So "only 1.7% of orders fell outside the range" states where the threshold was
set, not how well anyone traded. Set it to 0.35% and it looks better; set it to
p10/p90 and 20% flags, with identical underlying executions. The number that
carries information is the calibration table in Step 3. Change it in
`tier3_model/config.py`; below about 0.0025 the far tail has too few
observations to fit and calibration starts to drift.

There is also an optional **materiality gate** (`min_notional_review`): `flagged`
is everything outside the band, and an order additionally needs a minimum
notional to become `review_required`. It ships **off**, so the two are equal and
everything outside the threshold goes in the queue.

### Step 3 --- Prove the threshold is honest

Two safeguards, both of which matter more than they sound.

**Cross-fitting.** Every order is scored by a model that was fitted *without* it
(the book is split into 5 parts; each part is scored by a model trained on the
other 4). Without this, a band fitted and tested on the same orders always looks
near-perfect. Tier 2 has this problem and cannot fix it.

**Calibration.** On real data you never find out which orders were "really" bad,
so you cannot measure accuracy directly. But you *can* check that a threshold set
to catch 2% actually catches about 2% of orders it has never seen:

```
                 bound       promised   delivered
  below the lower band          2.0%       2.03%
  above the upper band          1.0%       1.09%
  below the middle             50.0%      50.03%
```

Delivered far *above* promised means the model is missing something. Far *below*
means it is overfitted and will stop flagging next quarter. **This is the single
check to look at before trusting any output**, and it is the only validation
available on live data.

### Step 4 --- Find the systematic problems

Here is the part that changes what the report is *for*.

A single order's z-score is mostly market noise --- the price moved while you
worked, and you got what you got. But average it over hundreds of orders and the
noise cancels:

```
t  =  average z  /  (standard deviation  /  sqrt(number of orders))
```

The **t-statistic** measures whether a pattern is real or luck. Anything beyond
about 3 is very unlikely to be chance.

In the demo book, one broker is worse by 0.18 --- less than a fifth of a typical
move per order. **No single-order threshold can ever see that**; it barely shifts
the tail rate. Across 2,616 orders it comes out at **t = -9.3**, which is
overwhelming. The same test independently finds a strategy that degrades on large
orders.

Because dozens of slices are tested at once (roughly two will look significant by
pure chance), p-values are corrected using the Benjamini-Hochberg procedure, and
only findings surviving that correction are reported.

**Tiers 1 and 2 cannot produce this table at all.** Without an expected cost
there is no residual to average, so a broker who is consistently worse is
indistinguishable from one who was simply handed the harder orders.

### Step 5 --- Say what actually went wrong

A z-score tells you an order was bad. It does not tell you what to change. Where
the extract carries the right columns, each flagged order is attributed to a
cause:

| cause | evidence | what to do |
|---|---|---|
| **traded too aggressively** | price reverted after you finished, and participation was high | cap participation, work it longer |
| **spread bleed** | crossed the spread far more than this algo normally does, **and no reversion** | post more, cross less |
| **missed the close** | almost no auction participation where peers do real volume | route a closing slice |
| **adverse market drift** | the market moved hard during the interval | execution was fine; the timing decision is the issue |
| **unexplained** | no fingerprint | check the marks and the benchmark window before blaming execution |

**The reversion test is the important one.** If the price snaps back after you
finish, you moved it yourself --- trade slower. If it does not, you either paid
the spread or were adversely selected --- a completely different fix. Without
post-trade marks these two look identical and have opposite remedies.

That last point is enforced rather than just documented. When no reversion column
is present, low passive fill is genuinely ambiguous between "we crossed the
spread all day" and "we pushed so hard the price ran away", because both leave
the same footprint. The label emitted in that case is
**`low_passive_unverified`**, which names both remedies and says which column
would separate them --- rather than confidently prescribing "post more" for an
order whose real fix is "trade slower".

Rules trigger at percentiles of *your own book*, so nothing needs retuning for a
different market or year. On the demo book, **92% of flagged real failures get
the correct diagnosis**.

---

## Evidence

12,000 synthetic orders, of which 444 were *genuinely* broken --- the generator
records which ones and why, so the tiers can be scored rather than argued about.
Nothing is tuned to favour a tier; the failure types and hidden broker/algo
effects are written down in `synthetic_data.py`.

**Is the threshold calibrated?** A good threshold flags roughly the same share of
easy and hard orders. Anything else means it is measuring difficulty.

| order size | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| under 1% ADV | 4.2% | 4.1% | 3.3% |
| 1-5% | 5.4% | 4.1% | 2.9% |
| 5-10% | 8.9% | 4.5% | 3.1% |
| 10-20% | 17.6% | 7.4% | 3.7% |
| over 20% | 27.3% | 6.8% | 2.3% |
| **spread (lower is better)** | **23.1 pts** | **3.3 pts** | **1.4 pts** |

**Which one ranks orders best?** Each tier gets the *same* 3% review queue and
fills it with its own worst orders. This removes the size of the queue as a
variable --- otherwise a tier that flags more always looks better at catching
things.

| | queue | real problems caught | precision | recall |
|---|---|---|---|---|
| Tier 1 fixed | 359 | 188 | 52.4% | 42.3% |
| Tier 2 percentile | 359 | 116 | 32.3% | 26.1% |
| **Tier 3 model** | 359 | **242** | **67.4%** | **54.5%** |

For the same amount of analyst time, Tier 3 puts **29% more real problems** in
front of them than the fixed limit, and more than double what the percentile
bands manage.

**Diagnosis quality**, Tier 3 only:

| the real problem was | flagged | diagnosed correctly |
|---|---|---|
| traded too aggressively | 56 | 100% |
| missed the close | 45 | 95.6% |
| spread bleed | 42 | 90.5% |
| benchmark/data error | 84 | 85.7% |

### An honest result worth reading twice

Tier 2 scores *worse* than Tier 1 above, which is not what the ordering implies.
The cause is the denominator, not the method:

```bash
python -m tier2_percentile.run                      # F1 27.3%  (divide by spread)
python -m tier2_percentile.run --metric perf_norm   # F1 54.1%  (divide by sigma_expected)
```

Identical logic, identical percentiles --- **double the score**, purely from
normalizing by the right thing. Choosing what to divide by matters at least as
much as choosing how to set the band, and it is much the cheaper of the two
fixes.

---

## Your data

Wired for this extract:

| your column | used as | handling |
|---|---|---|
| `aggrTgtId` | order id | direct |
| `Strategy` | algo | direct |
| `Sym` | market + symbol | last 2 characters (`"0700 HK"` -> `HK`) |
| `Pvwap` | slippage vs interval VWAP | bps |
| `Sprd` | spread | bps |
| `%Adv` | order size | percent of ADV |
| `Vol` | volatility | Parkinson daily vol, percent -> bps |
| `PR` | participation rate | percent -> fraction |
| `Dur` | duration | minutes |
| `$Mln` | notional | x 1,000,000 |
| `#Shares` | quantity | x 1,000 |
| `%POST` | passive fill share | percent -> fraction |
| `%OPEN` + `%CLOSE` | auction share | summed, percent -> fraction |
| `Side` | direction | `BUY` / `SELL` / `SSH` -> buy/sell |
| `Rev30min` | post-trade reversion | sign-corrected using `Side` |

Notional is **USD** (`notional_currency` in `config.py`), which makes the bps
shortfall convertible into actual cash:

```
shortfall = residual_bps / 10,000  x  notional
```

Every scored order carries a `shortfall_ccy` column, and the cause table is
ranked by total money rather than order count --- ten orders costing $400k
matter more than sixty costing $30k, and only the cash column says so:

```
                        orders   mean_z   mean_shortfall_bps    total_usd
unexplained                 48    -4.02                -81.1   -2,713,857
low_passive_unverified      45    -3.33                -80.4   -2,241,231
missed_close                36    -3.52                -77.8   -1,471,786
```

**All seven cost-model features are live** (side included), and **three of four cause rules** work:
`spread_bleed` (from `%POST`) and `missed_close` (from `%OPEN` + `%CLOSE`). Both
attribute at 100% accuracy on labelled data.

`%POST`, `%OPEN` and `%CLOSE` are wired as **diagnostics, not model features**,
on purpose. They describe how the order was executed, not how hard it was --- and
if passive fill went into the expected-cost model, an algo that crosses the
spread all day would lower its own expectation and stop flagging, absorbing the
exact behaviour the report exists to catch.

### The reversion column, and why its sign needs checking

`Rev30min` is what makes cause attribution work. Reversion is the test that
separates "we moved the price ourselves" (it snaps back --- trade slower) from
"we paid the spread or got picked off" (it doesn't --- post more). Without it
those two collapse into one ambiguous bucket.

But reversion is **signed by direction**, and a raw post-trade-minus-fill
difference points opposite ways for buys and sells: you push a buy up so it
falls back, you push a sell down so it rises back. An unsigned column averaged
over a mixed book cancels to nothing, and on any single order a large negative
value means "reverted nicely" on a buy and "kept running against me" on a sell
--- the exact two states the column exists to separate.

There is a second trap on top of that. The house rule "+ is good, - is bad" is
unambiguous for `Pvwap`, but **not** for a reversion column. A buy whose price
falls back afterwards has "bad" post-trade movement, yet that is exactly the
impact signature worth detecting. A buy whose price keeps rising has "good"
movement and shows no impact at all. Same sign, opposite diagnosis. Reversion is
diagnostic, not evaluative --- a large positive reversion under the internal
convention means *you moved the market*, which is a problem, not a win.

So `check_extract.py` resolves it from the data rather than from anyone's
recollection. It scores all four possible conventions on a physical fact ---
bigger and faster orders push harder and give back more --- so the correct one
is where reversion **rises** with size and participation:

```
     convention  corr_%ADV  corr_POV   score
            raw     -0.013    -0.002  -0.007
   raw_inverted     +0.013    +0.002  +0.007
         signed     -0.592    -0.181  -0.386
signed_inverted     +0.592    +0.181  +0.386   <- correct
```

The near-zero rows are the buy/sell cancellation, and they are what tells you
which *family* you are in: if the column were raw rather than pre-signed, the
zeros and the strong scores would swap places. So one table settles both the
sign and whether `Side` needs applying. Set `REVERSION_SIGN` in `config.py` to
whatever it names.

`SSH` (sell short) signs as a sell, which is all the reversion maths needs. The
raw code is preserved in `_side_raw` because HK's tick rule prevents a short
sale executing below the best bid, so shorts are forced into more passive
behaviour than long sales --- a genuinely different execution problem worth
slicing separately.

**What is still missing.**

*`broker`* --- no broker slice test, and this is now the highest-value column
outstanding. On the demo book that test finds the single largest systematic
effect.

*`momentum_bps`* --- the adverse-drift rule cannot fire. Minor here: interval
VWAP is largely immune to market drift, since the benchmark moves with it.

Everything the code needs to consume these is already in place --- add the column
to the extract and it lights up automatically.

### Configuration

`config.py` is the only shared file to edit:

- `PRE_TRANSFORM` --- anything a rename cannot express: unit scaling, deriving
  market from the ticker suffix. Runs before the rename.
- `COLUMN_MAP` --- canonical name to your column name. Names starting with `_`
  are produced by `PRE_TRANSFORM`.
- `SLIPPAGE_SIGN` --- whether positive means "beat the benchmark" or "cost".
- `DataConfig` --- source units, cleaning filters, the `sigma_expected`
  constants, session length, %ADV buckets.

Each tier's thresholds live in its own folder's `config.py`.

**Only five columns are strictly required** (`order_id, market, algo,
slippage_bps, spread_bps`). Everything else degrades gracefully, and every
degradation is *reported* rather than silent --- including falling back to
bucketed percentiles if `statsmodels` is not installed.

---

## Layout

```
check_extract.py       preflight: validate a file and infer its settings
score_new.py           apply a frozen threshold to future orders
config.py              shared data contract --- the only file you edit for real data
synthetic_data.py      demo book with a documented generator + ground-truth labels
run.py                 all three tiers, side by side
tca/                   shared infrastructure, no thresholds of its own
  schema.py              canonical column names
  pipeline.py            load -> units -> clean -> metrics -> buckets
  dataset.py             CLI to prepared frame (so every tier scores the same rows)
  evaluate.py            precision/recall, including the matched-budget comparison
  report.py              shared formatting
tier1_fixed/           Tier 1: fixed limits
tier2_percentile/      Tier 2: percentile bands, own history or a peer universe
tier3_model/           Tier 3: cost model, z-scores, slice tests, cause attribution
  features.py            the design matrix (square-root law)
  cost_model.py          quantile regression, cross-fitting, calibration
  scoring.py             residual z, zones, severity tiers, review gate
  persist.py             freeze/load the threshold + drift detection
  aggregate.py           slice t-tests with false-discovery control
  diagnostics.py         cause attribution + single-order narratives
```

Each tier folder has its own README with the method, the settings and its known
weaknesses.

## Known limits

- **The slice tests assume orders are independent.** Many orders in the same name
  on the same day are not, so borderline results are optimistic. Strong findings
  (t beyond 5) are unaffected.
- **Cause attribution is a rule engine, not a model.** The rules trigger on
  percentiles of your own book, so they travel, but they encode judgement about
  what matters for interval VWAP.
- **Algo effects are absorbed by default**, so a structurally worse algo hides in
  its own baseline and never flags at order level. That is intentional --- the
  slice tests are what catch it. Use `--algo-effect expose` when the question is
  which algo to keep.
- **Session length is set to HK's 330 minutes.** If you trade several venues this
  is a mild misspecification for the others; it enters inside a square root and
  the duration feature absorbs a constant proportional error.
