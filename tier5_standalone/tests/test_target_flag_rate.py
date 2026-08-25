"""Calibrate k to a review load instead of asserting k = 3.

On a leptokurtic book -- which every real execution book is -- k = 3 does not
deliver the 0.27% it promises. Rather than pretend otherwise, the band can be
fitted to the flag rate actually wanted, with the data supplying the k.
"""

import dataclasses
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from tca import pipeline, schema
from tier5 import cells, config as t5cfg, fit, normality


def _heavy(n=20000, seed=5, df=2.5, scale=1.0, centre=-0.25,
           start="2025-06-02", days=261):
    """A book shaped like the real one: sharp middle, fat tails.

    Dates are drawn from a fixed window rather than one row per business day --
    n rows of consecutive business days would span decades and trip the
    leakage guard for reasons that have nothing to do with what is being tested.
    """
    rng = np.random.default_rng(seed)
    x = centre + scale * rng.standard_t(df, n)
    span = pd.bdate_range(start, periods=days)
    frame = pd.DataFrame({
        schema.ORDER_ID: [f"O{i}" for i in range(n)],
        schema.MARKET: "HK", schema.ALGO: "VWAP",
        schema.SLIPPAGE_BPS: x * 10.0,
        schema.SPREAD_BPS: 10.0,
        schema.PERF_IN_SPREADS: x,
        schema.PCT_ADV: 1.0, schema.VOLATILITY: 180.0,
        schema.DURATION_MIN: 60.0,
        schema.ORDER_DATE: span[rng.integers(0, days, n)].astype(str),
    })
    return frame


def _fit(tmp_path, cfg, df=None):
    return fit.fit_frame(df if df is not None else _heavy(), cfg,
                         bands_dir=str(tmp_path / "bands"),
                         out_dir=str(tmp_path / "out"), source_csv="y.csv")


def _band(tmp_path):
    with open(tmp_path / "bands" / "HK" / "VWAP.json") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# the setting
# --------------------------------------------------------------------------

# CONFIG now carries the coverage standard (COVERAGE_PCT), so it is no longer
# the plain fixed-k baseline these tests compare against. Name that explicitly
# rather than letting "default" quietly mean two different things.
# k = 3 explicitly, not "whatever CONFIG happens to hold". These tests are
# about what a FIXED multiple does to a heavy book, so the multiple has to be
# pinned or the test silently starts measuring the current policy instead.
FIXED_K = dataclasses.replace(t5cfg.CONFIG, band_percentile=None,
                              target_flag_rate=None, k_sigma=3.0)


def test_the_shipped_default_is_the_per_side_rule_not_this_one():
    """Coverage is an opt-in. The default is MAX(mean+k*sigma, P), so a
    target_flag_rate left unset must NOT quietly become the standard."""
    assert t5cfg.CONFIG.target_flag_rate is None
    assert t5cfg.CONFIG.band_percentile == t5cfg.PERCENTILE_PCT
    assert t5cfg.CONFIG.k_sigma == t5cfg.K_SIGMA


def test_k_three_on_a_heavy_book_overshoots_badly(tmp_path):
    """The problem this option exists to solve, stated as a test."""
    res = _fit(tmp_path, FIXED_K)
    assert res[0]["flag_rate_pct"] > 4 * 100 * normality.NOMINAL_OUTSIDE


# --------------------------------------------------------------------------
# calibrating to a rate
# --------------------------------------------------------------------------

def test_target_rate_is_delivered_on_the_fit_book(tmp_path):
    cfg = dataclasses.replace(t5cfg.CONFIG, band_percentile=None,
                              target_flag_rate=0.5)
    res = _fit(tmp_path, cfg)
    assert res[0]["flag_rate_pct"] == pytest.approx(0.5, abs=0.05)


def test_a_tighter_target_is_also_delivered(tmp_path):
    cfg = dataclasses.replace(t5cfg.CONFIG, band_percentile=None,
                              target_flag_rate=0.27)
    res = _fit(tmp_path, cfg)
    assert res[0]["flag_rate_pct"] == pytest.approx(0.27, abs=0.05)


def test_the_band_widens_rather_than_narrows(tmp_path):
    """The boss's actual request: fewer orders outside."""
    wide = _fit(tmp_path / "a", dataclasses.replace(t5cfg.CONFIG,
                                                    target_flag_rate=0.5))
    plain = _fit(tmp_path / "b", FIXED_K)
    assert wide[0]["lo"] < plain[0]["lo"]
    assert wide[0]["hi"] > plain[0]["hi"]


def test_the_solved_k_is_reported(tmp_path):
    cfg = dataclasses.replace(t5cfg.CONFIG, band_percentile=None,
                              target_flag_rate=0.5)
    res = _fit(tmp_path, cfg)
    assert res[0]["k_used"] > 3.0


def test_the_band_file_records_the_k_actually_used(tmp_path):
    """Scoring must apply the same k that was fitted, not the config default."""
    cfg = dataclasses.replace(t5cfg.CONFIG, band_percentile=None,
                              target_flag_rate=0.5)
    res = _fit(tmp_path, cfg)
    saved = _band(tmp_path)
    assert saved["k_sigma"] == pytest.approx(res[0]["k_used"])
    assert saved["k_sigma"] != 3.0


def test_the_band_file_says_how_k_was_chosen(tmp_path):
    """A band at k=5.9 must not look like somebody's arbitrary guess."""
    cfg = dataclasses.replace(t5cfg.CONFIG, band_percentile=None,
                              target_flag_rate=0.5)
    _fit(tmp_path, cfg)
    saved = _band(tmp_path)
    assert saved["k_source"] == "target_flag_rate"
    assert saved["target_flag_rate"] == 0.5


def test_a_fixed_k_says_so_too(tmp_path):
    _fit(tmp_path, FIXED_K)
    saved = _band(tmp_path)
    assert saved["k_source"] == "fixed"
    assert saved["target_flag_rate"] is None


def test_bounds_stay_centre_plus_minus_k_scale(tmp_path):
    """Still the same method -- only k moved."""
    cfg = dataclasses.replace(t5cfg.CONFIG, band_percentile=None,
                              target_flag_rate=0.5)
    res = _fit(tmp_path, cfg)
    r = res[0]
    assert r["lo"] == pytest.approx(r["centre"] - r["k_used"] * r["scale"])
    assert r["hi"] == pytest.approx(r["centre"] + r["k_used"] * r["scale"])


def test_each_cell_is_calibrated_separately(tmp_path):
    """Two books with different tails need different k for the same load."""
    a = _heavy(n=8000, seed=1, df=2.0)
    b = _heavy(n=8000, seed=2, df=30.0)
    b[schema.ALGO] = "TWAP"
    # Floor lowered out of the way: this test is about the COVERAGE rule
    # calibrating per cell, and a floor that catches both cells would flatten
    # exactly the difference being measured.
    cfg = dataclasses.replace(t5cfg.CONFIG, band_percentile=None,
                              target_flag_rate=0.5, k_sigma=1.0)
    res = _fit(tmp_path, cfg, df=pd.concat([a, b], ignore_index=True))
    ks = {r["strategy"]: r["k_used"] for r in res}
    assert ks["VWAP"] > ks["TWAP"]


# --------------------------------------------------------------------------
# refusing nonsense
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0.0, -1.0, 100.0, 150.0])
def test_an_impossible_target_is_refused(tmp_path, bad):
    cfg = dataclasses.replace(t5cfg.CONFIG, band_percentile=None,
                              target_flag_rate=bad)
    with pytest.raises(ValueError, match="target_flag_rate"):
        _fit(tmp_path, cfg)


# --------------------------------------------------------------------------
# the CLI
# --------------------------------------------------------------------------

def test_cli_flag_calibrates_and_says_so(tmp_path, capsys):
    csv = str(tmp_path / "year.csv")
    raw = _heavy(n=6000)
    pd.DataFrame({
        "aggrTgtId": raw[schema.ORDER_ID], "Sym": "0700 HK",
        "Strategy": "VWAP", "Date": raw[schema.ORDER_DATE],
        "ePvwap/Sprd": raw[schema.PERF_IN_SPREADS],
        "Pvwap": raw[schema.SLIPPAGE_BPS], "Sprd": raw[schema.SPREAD_BPS],
    }).to_csv(csv, index=False)

    argv = sys.argv
    sys.argv = ["fit", "--csv", csv, "--target-flag-rate", "0.5",
                "--bands-dir", str(tmp_path / "bands"),
                "--out-dir", str(tmp_path / "outputs")]
    try:
        fit.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "0.5" in out
    assert "k=" in out.lower() or "k =" in out.lower()


# --------------------------------------------------------------------------
# the second lever: don't review a flagged order that cannot matter
# --------------------------------------------------------------------------

def _scored(tmp_path, gate=0.0, notionals=None):
    from tier5 import score
    year = _heavy(n=4000, seed=11)
    fit.fit_frame(year, t5cfg.CONFIG, bands_dir=str(tmp_path / "bands"),
                  out_dir=str(tmp_path / "out"), source_csv="y.csv")
    later = _heavy(n=2000, seed=12, start="2030-07-01", days=23)
    later[schema.NOTIONAL] = (notionals if notionals is not None
                              else np.linspace(1e3, 5e7, len(later)))
    res = score.score_frame(later, t5cfg.CONFIG,
                            bands_dir=str(tmp_path / "bands"),
                            out_dir=str(tmp_path / "out"), label="2030-07",
                            min_notional_review=gate or None)
    return res[0]


def test_review_required_equals_flagged_when_the_gate_is_off(tmp_path):
    r = _scored(tmp_path, gate=0.0)
    assert r["n_review"] == r["n_flagged"]


def test_the_gate_shrinks_the_review_queue(tmp_path):
    r = _scored(tmp_path, gate=1e7)
    assert 0 < r["n_review"] < r["n_flagged"]


def test_outliers_csv_still_lists_every_flagged_order(tmp_path):
    """The gate must shrink the queue, not hide orders from the record."""
    r = _scored(tmp_path, gate=1e7)
    out = pd.read_csv(os.path.join(r["out_dir"], "outliers.csv"))
    assert len(out) == r["n_flagged"]
    assert "review_required" in out.columns
    assert int(out["review_required"].sum()) == r["n_review"]


def test_the_gate_travels_with_the_band_file(tmp_path):
    """Frozen alongside the band, so a scored queue is reproducible."""
    from tier5 import score
    year = _heavy(n=4000, seed=13)
    cfg = dataclasses.replace(t5cfg.CONFIG, min_notional_review=2.5e7)
    fit.fit_frame(year, cfg, bands_dir=str(tmp_path / "bands"),
                  out_dir=str(tmp_path / "out"), source_csv="y.csv")
    with open(tmp_path / "bands" / "HK" / "VWAP.json") as fh:
        assert json.load(fh)["scoring_config"]["min_notional_review"] == 2.5e7


def test_the_frozen_gate_applies_when_no_override_is_given(tmp_path):
    """Reproducibility: rescoring the same band gives the same queue."""
    from tier5 import score
    year = _heavy(n=4000, seed=14)
    cfg = dataclasses.replace(t5cfg.CONFIG, min_notional_review=1e7)
    fit.fit_frame(year, cfg, bands_dir=str(tmp_path / "bands"),
                  out_dir=str(tmp_path / "out"), source_csv="y.csv")
    later = _heavy(n=2000, seed=15, start="2030-07-01", days=23)
    later[schema.NOTIONAL] = np.linspace(1e3, 5e7, len(later))
    r = score.score_frame(later, t5cfg.CONFIG,
                          bands_dir=str(tmp_path / "bands"),
                          out_dir=str(tmp_path / "out"), label="2030-07")[0]
    assert r["min_notional_review"] == 1e7
    assert r["n_review"] < r["n_flagged"]


def test_an_override_beats_the_frozen_gate(tmp_path):
    """Review capacity is a policy, not a property of the band."""
    r = _scored(tmp_path, gate=1e7)
    assert r["min_notional_review"] == 1e7
    assert 0 < r["n_review"] < r["n_flagged"]
