"""The banded metric now arrives from the extract already divided by spread.

Different strategies are benchmarked differently, so each names its own source
column: VWAP against interval VWAP, PART/POV against arrival. These tests pin
the resolution down, because picking the wrong source column produces a
perfectly plausible band on the wrong benchmark.
"""

import numpy as np
import pandas as pd
import pytest

import config
from tca import pipeline, schema
from tier5 import config as t5cfg


def raw(**over):
    """A raw extract frame, using the vendor column names."""
    n = over.pop("n", 6)
    base = {
        "aggrTgtId": [f"O{i}" for i in range(n)],
        "Strategy": ["VWAP"] * n,
        "Sym": ["0700 HK"] * n,
        "Date": ["2025-06-02"] * n,
        "Pvwap": np.linspace(-20.0, 20.0, n),
        "Sprd": np.full(n, 10.0),
        "ePvwap/Sprd": np.linspace(-2.0, 2.0, n),
        "eIS/Sprd": np.linspace(-5.0, 5.0, n),
    }
    base.update(over)
    return pd.DataFrame(base)


# --------------------------------------------------------------------------
# resolving the source column
# --------------------------------------------------------------------------

def test_vwap_rows_take_the_pvwap_spread_column():
    out = config.PRE_TRANSFORM(raw())
    assert list(out["_perf_spreads"]) == pytest.approx(list(raw()["ePvwap/Sprd"]))


def test_part_rows_take_the_arrival_column():
    df = raw(n=4, Strategy=["PART"] * 4)
    out = config.PRE_TRANSFORM(df)
    assert list(out["_perf_spreads"]) == pytest.approx(list(df["eIS/Sprd"]))


def test_a_mixed_book_resolves_each_strategy_separately():
    """One year file can hold both. Each cell must band its own benchmark."""
    df = raw(n=4, Strategy=["VWAP", "PART", "VWAP", "PART"])
    out = config.PRE_TRANSFORM(df)
    got = list(out["_perf_spreads"])
    assert got[0] == pytest.approx(df["ePvwap/Sprd"][0])
    assert got[1] == pytest.approx(df["eIS/Sprd"][1])
    assert got[2] == pytest.approx(df["ePvwap/Sprd"][2])
    assert got[3] == pytest.approx(df["eIS/Sprd"][3])


def test_strategy_names_match_case_insensitively():
    df = raw(n=3, Strategy=["vwap", "Vwap", " VWAP "])
    out = config.PRE_TRANSFORM(df)
    assert out["_perf_spreads"].notna().all()


def test_an_unlisted_strategy_falls_back_to_the_default_column():
    df = raw(n=3, Strategy=["SNIPER"] * 3)
    out = config.PRE_TRANSFORM(df)
    assert list(out["_perf_spreads"]) == pytest.approx(list(df[config.METRIC_COLUMN_DEFAULT]))


def test_a_missing_source_column_yields_nan_rather_than_raising():
    """A strategy whose column was left out of the export must not crash the run."""
    df = raw(n=3, Strategy=["PART"] * 3).drop(columns=["eIS/Sprd"])
    out = config.PRE_TRANSFORM(df)
    assert out["_perf_spreads"].isna().all()


def test_synthetic_data_without_vendor_columns_is_left_alone():
    """The demo uses canonical names and has no ePvwap/Sprd; it must still run."""
    df = pd.DataFrame({"slippage_bps": [1.0, 2.0], "spread_bps": [10.0, 10.0]})
    out = config.PRE_TRANSFORM(df)
    assert "_perf_spreads" not in out.columns


# --------------------------------------------------------------------------
# reporting which column each strategy used
# --------------------------------------------------------------------------

def test_metric_sources_reports_the_column_used_per_strategy():
    """A silent fallback is the dangerous case, so it has to be printable."""
    df = raw(n=4, Strategy=["VWAP", "PART", "VWAP", "SNIPER"])
    table = config.metric_sources(df)
    got = dict(zip(table["strategy"], table["column"]))
    assert got["VWAP"] == "ePvwap/Sprd"
    assert got["PART"] == "eIS/Sprd"
    assert got["SNIPER"] == config.METRIC_COLUMN_DEFAULT


def test_metric_sources_flags_a_strategy_using_the_fallback():
    df = raw(n=2, Strategy=["SNIPER"] * 2)
    table = config.metric_sources(df)
    assert bool(table.loc[table["strategy"] == "SNIPER", "fallback"].iloc[0])


def test_metric_sources_counts_rows_with_no_value():
    df = raw(n=3, Strategy=["PART"] * 3).drop(columns=["eIS/Sprd"])
    table = config.metric_sources(df)
    row = table[table["strategy"] == "PART"].iloc[0]
    assert int(row["n_rows"]) == 3
    assert int(row["n_missing"]) == 3


# --------------------------------------------------------------------------
# the mapped column reaches the pipeline intact
# --------------------------------------------------------------------------

def test_column_map_points_perf_in_spreads_at_the_derived_column():
    assert config.COLUMN_MAP[schema.PERF_IN_SPREADS] == "_perf_spreads"


def test_a_supplied_metric_is_never_divided_by_the_spread_again():
    """The one mistake here that produces plausible-looking output."""
    df = pd.DataFrame({
        schema.SLIPPAGE_BPS: [20.0, -20.0],
        schema.SPREAD_BPS: [10.0, 10.0],
        schema.PERF_IN_SPREADS: [1.5, -1.5],
    })
    out = pipeline.add_metric(df)
    assert list(out[schema.PERF_IN_SPREADS]) == pytest.approx([1.5, -1.5])


def test_the_metric_is_still_derived_when_the_extract_lacks_it():
    """The synthetic demo has no pre-normalised column and must keep working."""
    df = pd.DataFrame({schema.SLIPPAGE_BPS: [20.0], schema.SPREAD_BPS: [10.0]})
    out = pipeline.add_metric(df)
    assert out[schema.PERF_IN_SPREADS].iloc[0] == pytest.approx(2.0)


def test_a_supplied_metric_follows_the_slippage_sign_convention():
    """Both columns come from one system, so one flip has to move both."""
    df = pd.DataFrame({
        schema.ORDER_ID: ["A"], schema.MARKET: ["HK"], schema.ALGO: ["VWAP"],
        schema.SLIPPAGE_BPS: [20.0], schema.SPREAD_BPS: [10.0],
        schema.PERF_IN_SPREADS: [1.5],
    })
    out, _ = pipeline.clean(df, config.DATA, "cost")
    assert out[schema.PERF_IN_SPREADS].iloc[0] == pytest.approx(-1.5)


def test_rows_with_no_supplied_metric_are_dropped_and_counted():
    """A NaN metric cannot be banded; vanishing without a count is the bad case."""
    df = pd.DataFrame({
        schema.ORDER_ID: ["A", "B"], schema.MARKET: ["HK", "HK"],
        schema.ALGO: ["VWAP", "VWAP"],
        schema.SLIPPAGE_BPS: [20.0, 20.0], schema.SPREAD_BPS: [10.0, 10.0],
        schema.PERF_IN_SPREADS: [1.5, np.nan],
    })
    out, rep = pipeline.clean(df, config.DATA, "positive_is_good")
    assert len(out) == 1
    assert rep.dropped_no_metric == 1
    assert "no metric" in rep.as_text().lower()


# --------------------------------------------------------------------------
# the band's units
# --------------------------------------------------------------------------

def test_tier5_bands_the_spread_normalised_metric_by_default():
    assert t5cfg.CONFIG.metric == schema.PERF_IN_SPREADS


def test_units_are_named_per_metric():
    assert t5cfg.units_of(schema.PERF_IN_SPREADS) == "spreads"
    assert t5cfg.units_of(schema.SLIPPAGE_BPS) == "bps"


# --------------------------------------------------------------------------
# the band file, and stale bands in the other unit
# --------------------------------------------------------------------------

def _cell(n=800, seed=3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        schema.ORDER_ID: [f"O{i}" for i in range(n)],
        schema.MARKET: "HK", schema.ALGO: "VWAP",
        schema.SLIPPAGE_BPS: rng.normal(-10.0, 20.0, n),
        schema.SPREAD_BPS: rng.uniform(5.0, 15.0, n),
        schema.PCT_ADV: rng.uniform(0.1, 5.0, n),
        schema.VOLATILITY: rng.uniform(100.0, 250.0, n),
        schema.DURATION_MIN: rng.uniform(10.0, 300.0, n),
        schema.ORDER_DATE: pd.bdate_range("2025-06-02", periods=n).astype(str),
    })
    return pipeline.add_metric(df)


def test_band_file_records_the_units_of_its_bounds(tmp_path):
    """"-2.31 .. 1.87" is unreadable, and quietly wrong if read as bps."""
    from tier5 import band, fit
    fit.fit_frame(_cell(), t5cfg.CONFIG, bands_dir=str(tmp_path / "bands"),
                  out_dir=str(tmp_path / "out"), source_csv="y.csv")
    import json
    with open(tmp_path / "bands" / "HK" / "VWAP.json") as fh:
        saved = json.load(fh)
    assert saved["metric"] == schema.PERF_IN_SPREADS
    assert saved["metric_units"] == "spreads"


def test_a_band_frozen_on_a_different_metric_is_skipped_not_scored(tmp_path):
    """A leftover bps band must not silently score alongside spread bands."""
    import dataclasses
    from tier5 import fit, score
    bps_cfg = dataclasses.replace(t5cfg.CONFIG, metric=schema.SLIPPAGE_BPS)
    fit.fit_frame(_cell(), bps_cfg, bands_dir=str(tmp_path / "bands"),
                  out_dir=str(tmp_path / "out"), source_csv="y.csv")

    later = _cell(n=300, seed=9)
    later[schema.ORDER_DATE] = pd.bdate_range("2030-07-01", periods=300).astype(str)
    res = score.score_frame(later, t5cfg.CONFIG,
                            bands_dir=str(tmp_path / "bands"),
                            out_dir=str(tmp_path / "out"))
    assert res[0]["skipped"]
    assert schema.SLIPPAGE_BPS in res[0]["reason"]
    assert schema.PERF_IN_SPREADS in res[0]["reason"]


def test_a_cell_with_no_usable_metric_says_so(tmp_path):
    """Not 'n=0 below min_group_n', which points at the wrong problem."""
    from tier5 import fit
    df = _cell(n=400)
    df[t5cfg.CONFIG.metric] = np.nan
    res = fit.fit_frame(df, t5cfg.CONFIG, bands_dir=str(tmp_path / "bands"),
                        out_dir=str(tmp_path / "out"), source_csv="y.csv")
    assert res[0]["skipped"]
    assert t5cfg.CONFIG.metric in res[0]["reason"]
    assert "min_group_n" not in res[0]["reason"]


# --------------------------------------------------------------------------
# the real CLI path, on an extract shaped like the production one
# --------------------------------------------------------------------------

def _extract(path, strategy, n, seed, start="2025-06-02"):
    import os
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    spread = rng.uniform(5.0, 15.0, n)
    in_spreads = rng.normal(-0.8, 1.6, n)
    # Pvwap is deliberately NOT ePvwap/Sprd x Sprd. They are different columns
    # measuring different things in the real extract, and keeping them distinct
    # here is what lets these tests tell a re-derived metric apart from the
    # supplied one -- if Pvwap/Sprd happened to equal ePvwap/Sprd, a double
    # normalisation would be invisible.
    pd.DataFrame({
        "aggrTgtId": [f"{strategy}{i}" for i in range(n)],
        "Sym": "0700 HK",
        "Strategy": strategy,
        "Date": pd.bdate_range(start, periods=n).astype(str),
        "Pvwap": (in_spreads + 0.6) * spread,
        "Sprd": spread,
        "ePvwap/Sprd": in_spreads,
        "eIS/Sprd": in_spreads + rng.normal(0.0, 0.4, n),
        "%Adv": rng.uniform(0.1, 5.0, n),
        "Vol": rng.uniform(1.0, 2.5, n),
        "PR": rng.uniform(1.0, 20.0, n),
        "Dur": rng.uniform(10.0, 300.0, n),
    }).to_csv(path, index=False)
    return path


def _run_fit(tmp_path, csv, k=None):
    """`k` pins the multiple. These tests are about which COLUMN the band came
    from and what UNITS it is in, so they must not also depend on the coverage
    standard in tier5/config.py -- that would make an unrelated policy change
    read as a units bug."""
    import sys
    from tier5 import fit
    argv = sys.argv
    sys.argv = ["fit", "--csv", csv,
                "--bands-dir", str(tmp_path / "bands"),
                "--out-dir", str(tmp_path / "outputs")] + (
                    ["--k", str(k)] if k is not None else [])
    try:
        fit.main()
    finally:
        sys.argv = argv


def test_bounds_come_out_in_spreads_not_bps(tmp_path, capsys):
    """The end-to-end guard against dividing by the spread a second time."""
    import json
    csv = _extract(str(tmp_path / "year.csv"), "VWAP", 800, 11)
    _run_fit(tmp_path, csv)
    with open(tmp_path / "bands" / "HK" / "VWAP.json") as fh:
        saved = json.load(fh)
    # Fitted on a N(-0.8, 1.6) book in spreads: 3 sigma lands near +/- 5.
    # A double normalisation would divide by ~10 again and land near +/- 0.5;
    # forgetting to normalise at all would land near +/- 50.
    assert 3.0 < abs(saved["lo"]) < 9.0
    assert saved["metric_units"] == "spreads"


def test_fit_prints_the_units_next_to_the_bounds(tmp_path, capsys):
    """On the RANGE line itself -- that is the number people copy out."""
    csv = _extract(str(tmp_path / "year.csv"), "VWAP", 800, 12)
    _run_fit(tmp_path, csv)
    line = [l for l in capsys.readouterr().out.splitlines() if "RANGE" in l]
    assert line and "spreads" in line[0]


def test_fit_prints_which_column_fed_each_strategy(tmp_path, capsys):
    """The choice that is invisible in the output has to be stated in it."""
    csv = _extract(str(tmp_path / "year.csv"), "PART", 800, 13)
    _run_fit(tmp_path, csv)
    out = capsys.readouterr().out
    assert "eIS/Sprd" in out
    assert "PART" in out


def test_fit_warns_when_a_strategy_falls_back_to_the_default_column(tmp_path, capsys):
    csv = _extract(str(tmp_path / "year.csv"), "SNIPER", 800, 14)
    _run_fit(tmp_path, csv)
    out = capsys.readouterr().out
    assert "SNIPER" in out
    assert "fallback" in out.lower()


def test_score_prints_the_units_next_to_the_frozen_band(tmp_path, capsys):
    import sys
    from tier5 import score
    _run_fit(tmp_path, _extract(str(tmp_path / "year.csv"), "VWAP", 800, 21))
    july = _extract(str(tmp_path / "july.csv"), "VWAP", 300, 22, start="2030-07-01")
    argv = sys.argv
    sys.argv = ["score", "--csv", july,
                "--bands-dir", str(tmp_path / "bands"),
                "--out-dir", str(tmp_path / "outputs")]
    try:
        score.main()
    finally:
        sys.argv = argv
    line = [l for l in capsys.readouterr().out.splitlines()
            if l.strip().startswith("band ")]
    assert line and "spreads" in line[0]


def test_batch_prints_which_column_fed_each_strategy(tmp_path, capsys):
    import sys
    from tier5 import batch
    _extract(str(tmp_path / "year" / "HK" / "PART.csv"), "PART", 600, 23)
    argv = sys.argv
    sys.argv = ["batch", "fit", "--dir", str(tmp_path / "year"),
                "--bands-dir", str(tmp_path / "bands"),
                "--out-dir", str(tmp_path / "outputs")]
    try:
        batch.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "eIS/Sprd" in out


def test_check_extract_names_the_metric_column_per_strategy(tmp_path, capsys):
    """The first thing run on a new extract must confirm what will be banded."""
    import sys
    import check_extract
    csv = _extract(str(tmp_path / "year.csv"), "PART", 300, 31)
    argv = sys.argv
    sys.argv = ["check_extract", csv]
    try:
        check_extract.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "BANDED METRIC" in out.upper()
    assert "eIS/Sprd" in out
    assert "PART" in out


def test_check_extract_reports_a_strategy_with_no_metric_column(tmp_path, capsys):
    """Rows that cannot be banded must be visible before the fit, not after."""
    import sys
    import check_extract
    csv = str(tmp_path / "year.csv")
    _extract(csv, "PART", 300, 32)
    pd.read_csv(csv).drop(columns=["eIS/Sprd"]).to_csv(csv, index=False)
    argv = sys.argv
    sys.argv = ["check_extract", csv]
    try:
        check_extract.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "300" in out and "eIS/Sprd" in out


# --------------------------------------------------------------------------
# the bounds are in spreads: proved against the raw column, not asserted
# --------------------------------------------------------------------------

def test_band_bounds_equal_mean_plus_k_sd_of_the_raw_spread_column(tmp_path):
    """lo/hi must be a pure function of ePvwap/Sprd, in its own units.

    Computed here straight from the CSV with pandas, bypassing the whole
    pipeline. If anything between the file and the band file rescales the
    metric -- a unit conversion, a sign flip, a second division by the spread --
    these two numbers stop matching.
    """
    import json
    csv = _extract(str(tmp_path / "year.csv"), "VWAP", 900, 41)
    _run_fit(tmp_path, csv, k=3.0)

    x = pd.read_csv(csv)["ePvwap/Sprd"]
    expected_lo = x.mean() - 3.0 * x.std(ddof=1)
    expected_hi = x.mean() + 3.0 * x.std(ddof=1)

    with open(tmp_path / "bands" / "HK" / "VWAP.json") as fh:
        saved = json.load(fh)
    assert saved["lo"] == pytest.approx(expected_lo, rel=1e-12)
    assert saved["hi"] == pytest.approx(expected_hi, rel=1e-12)
    assert saved["centre"] == pytest.approx(x.mean(), rel=1e-12)
    assert saved["scale"] == pytest.approx(x.std(ddof=1), rel=1e-12)


def test_the_bps_column_does_not_influence_the_band(tmp_path):
    """Change Pvwap wildly, leave ePvwap/Sprd alone: the band must not move."""
    import json
    csv = _extract(str(tmp_path / "a.csv"), "VWAP", 900, 42)
    _run_fit(tmp_path, csv)
    with open(tmp_path / "bands" / "HK" / "VWAP.json") as fh:
        before = json.load(fh)

    df = pd.read_csv(csv)
    df["Pvwap"] = df["Pvwap"] * 7.5 + 100.0
    df.to_csv(csv, index=False)
    _run_fit(tmp_path, csv)
    with open(tmp_path / "bands" / "HK" / "VWAP.json") as fh:
        after = json.load(fh)

    assert after["lo"] == pytest.approx(before["lo"], rel=1e-12)
    assert after["hi"] == pytest.approx(before["hi"], rel=1e-12)


def test_the_spread_column_does_not_influence_the_band(tmp_path):
    """The division already happened at source; Sprd must not divide again."""
    import json
    csv = _extract(str(tmp_path / "b.csv"), "VWAP", 900, 43)
    _run_fit(tmp_path, csv)
    with open(tmp_path / "bands" / "HK" / "VWAP.json") as fh:
        before = json.load(fh)

    df = pd.read_csv(csv)
    df["Sprd"] = df["Sprd"] * 4.0
    df.to_csv(csv, index=False)
    _run_fit(tmp_path, csv)
    with open(tmp_path / "bands" / "HK" / "VWAP.json") as fh:
        after = json.load(fh)

    assert after["lo"] == pytest.approx(before["lo"], rel=1e-12)
    assert after["hi"] == pytest.approx(before["hi"], rel=1e-12)


def test_unit_settings_do_not_rescale_the_banded_metric(tmp_path):
    """volatility_unit / pct_adv_unit / participation_unit must leave it alone."""
    import dataclasses
    df = pd.DataFrame({
        schema.PERF_IN_SPREADS: [1.5, -2.25],
        schema.VOLATILITY: [1.8, 1.8], schema.PCT_ADV: [3.5, 3.5],
        schema.PARTICIPATION: [15.0, 15.0],
    })
    out = pipeline.normalize_units(df, config.DATA)
    assert list(out[schema.PERF_IN_SPREADS]) == pytest.approx([1.5, -2.25])
