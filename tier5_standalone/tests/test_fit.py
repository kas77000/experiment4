import dataclasses
import json
import os

import numpy as np
import pandas as pd

from tca import pipeline, schema
from tier5 import cells, config as t5cfg, fit


def _book(n_per_cell=600, seed=4):
    rng = np.random.default_rng(seed)
    frames = []
    for region, strategy, mu, sd in [("HK", "VWAP", -10.0, 20.0),
                                     ("HK", "TWAP", -14.0, 24.0),
                                     ("JP", "VWAP", -8.0, 18.0)]:
        frames.append(pipeline.add_metric(pd.DataFrame({
            schema.MARKET: region,
            schema.ALGO: strategy,
            schema.SLIPPAGE_BPS: rng.normal(mu, sd, n_per_cell),
            schema.SPREAD_BPS: rng.uniform(5.0, 15.0, n_per_cell),
            schema.PCT_ADV: rng.uniform(0.1, 5.0, n_per_cell),
            schema.VOLATILITY: rng.uniform(100.0, 250.0, n_per_cell),
            schema.DURATION_MIN: rng.uniform(10.0, 300.0, n_per_cell),
            schema.ORDER_DATE: pd.bdate_range("2025-06-02",
                                              periods=n_per_cell).astype(str),
        })))
    return pd.concat(frames, ignore_index=True)


def _thin(n=40, seed=9):
    rng = np.random.default_rng(seed)
    return pipeline.add_metric(pd.DataFrame({
        schema.MARKET: "AU", schema.ALGO: "IS",
        schema.SLIPPAGE_BPS: rng.normal(-9.0, 19.0, n),
        schema.SPREAD_BPS: 10.0, schema.PCT_ADV: 1.0,
        schema.VOLATILITY: 180.0, schema.DURATION_MIN: 60.0,
        schema.ORDER_DATE: "2025-07-01",
    }))


def test_writes_one_band_per_cell(tmp_path):
    res = fit.fit_frame(_book(), t5cfg.CONFIG,
                        bands_dir=str(tmp_path / "bands"),
                        out_dir=str(tmp_path / "outputs"),
                        source_csv="year.csv")
    written = {(r["region"], r["strategy"]) for r in res if not r["skipped"]}
    assert written == {("HK", "VWAP"), ("HK", "TWAP"), ("JP", "VWAP")}
    for region, strategy in written:
        assert os.path.exists(cells.band_path(str(tmp_path / "bands"),
                                              region, strategy))


def test_an_absolute_band_is_the_same_for_every_cell(tmp_path):
    """The shipped band is stated, so of course it does not vary by cell.

    Worth a test rather than a shrug: a per-cell band was the previous
    behaviour, and the property that replaced it is exactly the one somebody
    will question. What must STILL vary is the measurement underneath.
    """
    res = fit.fit_frame(_book(), t5cfg.CONFIG,
                        bands_dir=str(tmp_path / "bands"),
                        out_dir=str(tmp_path / "outputs"),
                        source_csv="year.csv")
    los = {(r["region"], r["strategy"]): r["lo"] for r in res}
    assert los[("HK", "TWAP")] == los[("HK", "VWAP")]

    centres = {(r["region"], r["strategy"]): r["centre"] for r in res}
    scales = {(r["region"], r["strategy"]): r["scale"] for r in res}
    assert centres[("HK", "TWAP")] != centres[("HK", "VWAP")]
    assert scales[("HK", "TWAP")] != scales[("HK", "VWAP")]


def test_a_fitted_band_still_differs_between_cells(tmp_path):
    """Switch the absolute band off and per-cell fitting comes back."""
    cfg = dataclasses.replace(t5cfg.CONFIG, band_abs=None)
    res = fit.fit_frame(_book(), cfg, bands_dir=str(tmp_path / "bands"),
                        out_dir=str(tmp_path / "outputs"),
                        source_csv="year.csv")
    los = {(r["region"], r["strategy"]): r["lo"] for r in res}
    assert los[("HK", "TWAP")] != los[("HK", "VWAP")]


def test_band_file_records_region_strategy_and_window(tmp_path):
    fit.fit_frame(_book(), t5cfg.CONFIG, bands_dir=str(tmp_path / "bands"),
                  out_dir=str(tmp_path / "outputs"), source_csv="year.csv")
    with open(cells.band_path(str(tmp_path / "bands"), "HK", "VWAP")) as fh:
        p = json.load(fh)
    assert p["region"] == "HK"
    assert p["strategy"] == "VWAP"
    assert p["fit_date_min"] == "2025-06-02"


def test_thin_cell_is_skipped_not_written(tmp_path):
    book = pd.concat([_book(), _thin()], ignore_index=True)
    res = fit.fit_frame(book, t5cfg.CONFIG, bands_dir=str(tmp_path / "bands"),
                        out_dir=str(tmp_path / "outputs"), source_csv="y.csv")
    thin = [r for r in res if r["region"] == "AU"][0]
    assert thin["skipped"] is True
    assert "min_group_n" in thin["reason"]
    assert not os.path.exists(cells.band_path(str(tmp_path / "bands"), "AU", "IS"))


def test_force_writes_thin_cell(tmp_path):
    res = fit.fit_frame(_thin(), t5cfg.CONFIG, bands_dir=str(tmp_path / "bands"),
                        out_dir=str(tmp_path / "outputs"), source_csv="y.csv",
                        force=True)
    assert res[0]["skipped"] is False
    assert os.path.exists(cells.band_path(str(tmp_path / "bands"), "AU", "IS"))


def test_flag_rate_is_recorded_and_sane(tmp_path):
    res = fit.fit_frame(_book(), t5cfg.CONFIG, bands_dir=str(tmp_path / "bands"),
                        out_dir=str(tmp_path / "outputs"), source_csv="y.csv")
    for r in res:
        assert 0.0 <= r["flag_rate_pct"] <= 5.0


def test_curve_written_per_cell(tmp_path):
    fit.fit_frame(_book(), t5cfg.CONFIG, bands_dir=str(tmp_path / "bands"),
                  out_dir=str(tmp_path / "outputs"), source_csv="y.csv")
    hits = []
    for _, _, files in os.walk(str(tmp_path / "outputs" / "fit")):
        hits += [f for f in files if f == "curve.png"]
    assert len(hits) == 3


def test_missing_metric_column_raises(tmp_path):
    """Whatever metric is configured, its absence must stop the run."""
    book = _book().drop(columns=[t5cfg.CONFIG.metric])
    import pytest
    with pytest.raises(ValueError, match=t5cfg.CONFIG.metric):
        fit.fit_frame(book, t5cfg.CONFIG, bands_dir=str(tmp_path / "bands"),
                      out_dir=str(tmp_path / "outputs"), source_csv="y.csv")
