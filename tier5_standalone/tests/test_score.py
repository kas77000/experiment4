import json
import os

import numpy as np
import pandas as pd
import pytest

from tca import schema
from tier5 import cells, config as t5cfg, fit, score


def _cell(region, strategy, n, mu, sd, start, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        schema.ORDER_ID: [f"{region}{strategy}{i}" for i in range(n)],
        schema.MARKET: region,
        schema.ALGO: strategy,
        schema.SLIPPAGE_BPS: rng.normal(mu, sd, n),
        schema.SPREAD_BPS: rng.uniform(5.0, 15.0, n),
        schema.PCT_ADV: rng.uniform(0.1, 5.0, n),
        schema.VOLATILITY: rng.uniform(100.0, 250.0, n),
        schema.DURATION_MIN: rng.uniform(10.0, 300.0, n),
        schema.ORDER_DATE: pd.bdate_range(start, periods=n).astype(str),
    })


@pytest.fixture
def frozen(tmp_path):
    year = pd.concat([_cell("HK", "VWAP", 800, -10.0, 20.0, "2025-06-02", 1),
                      _cell("JP", "VWAP", 800, -8.0, 18.0, "2025-06-02", 2)],
                     ignore_index=True)
    fit.fit_frame(year, t5cfg.CONFIG, bands_dir=str(tmp_path / "bands"),
                  out_dir=str(tmp_path / "outputs"), source_csv="year.csv")
    return tmp_path


def test_scores_against_frozen_band(frozen):
    july = _cell("HK", "VWAP", 300, -10.0, 20.0, "2030-07-01", 7)
    res = score.score_frame(july, t5cfg.CONFIG,
                            bands_dir=str(frozen / "bands"),
                            out_dir=str(frozen / "outputs"))
    r = res[0]
    assert r["skipped"] is False
    assert r["region"] == "HK" and r["strategy"] == "VWAP"
    assert 0 <= r["flag_rate_pct"] <= 100


def test_never_refits_bounds(frozen):
    """A far wider new period must NOT widen the band."""
    with open(cells.band_path(str(frozen / "bands"), "HK", "VWAP")) as fh:
        frozen_lo = json.load(fh)["lo"]
    july = _cell("HK", "VWAP", 300, -10.0, 90.0, "2030-07-01", 8)
    res = score.score_frame(july, t5cfg.CONFIG,
                            bands_dir=str(frozen / "bands"),
                            out_dir=str(frozen / "outputs"))
    assert res[0]["lo"] == frozen_lo
    assert res[0]["flag_rate_pct"] > 5.0   # wider data, same band -> more flags


def test_missing_band_is_skipped_not_borrowed(frozen):
    july = _cell("AU", "IS", 300, -10.0, 20.0, "2030-07-01", 9)
    res = score.score_frame(july, t5cfg.CONFIG,
                            bands_dir=str(frozen / "bands"),
                            out_dir=str(frozen / "outputs"))
    assert res[0]["skipped"] is True
    assert "no band" in res[0]["reason"].lower()


def test_overlapping_window_is_refused(frozen):
    overlapping = _cell("HK", "VWAP", 300, -10.0, 20.0, "2025-06-02", 10)
    with pytest.raises(score.LeakageError, match="overlap"):
        score.score_frame(overlapping, t5cfg.CONFIG,
                          bands_dir=str(frozen / "bands"),
                          out_dir=str(frozen / "outputs"))


def test_outliers_csv_written_and_sorted(frozen):
    july = _cell("HK", "VWAP", 400, -10.0, 45.0, "2030-07-01", 11)
    res = score.score_frame(july, t5cfg.CONFIG,
                            bands_dir=str(frozen / "bands"),
                            out_dir=str(frozen / "outputs"))
    path = os.path.join(res[0]["out_dir"], "outliers.csv")
    assert os.path.exists(path)
    out = pd.read_csv(path)
    assert len(out) == res[0]["n_flagged"]
    assert (out["n_sigma_outside"] > 0).all()
    assert out["n_sigma_outside"].is_monotonic_decreasing


def test_outliers_carry_diagnostic_columns(frozen):
    july = _cell("HK", "VWAP", 400, -10.0, 45.0, "2030-07-01", 14)
    res = score.score_frame(july, t5cfg.CONFIG,
                            bands_dir=str(frozen / "bands"),
                            out_dir=str(frozen / "outputs"))
    out = pd.read_csv(os.path.join(res[0]["out_dir"], "outliers.csv"))
    for col in (schema.ORDER_ID, schema.SPREAD_BPS, schema.PCT_ADV,
                schema.DURATION_MIN, schema.ORDER_DATE):
        assert col in out.columns


def test_scored_csv_has_every_row(frozen):
    july = _cell("HK", "VWAP", 300, -10.0, 20.0, "2030-07-01", 12)
    res = score.score_frame(july, t5cfg.CONFIG,
                            bands_dir=str(frozen / "bands"),
                            out_dir=str(frozen / "outputs"))
    scored = pd.read_csv(os.path.join(res[0]["out_dir"], "scored.csv"))
    assert len(scored) == 300


def test_label_overrides_period_folder(frozen):
    july = _cell("HK", "VWAP", 300, -10.0, 20.0, "2030-07-01", 13)
    res = score.score_frame(july, t5cfg.CONFIG,
                            bands_dir=str(frozen / "bands"),
                            out_dir=str(frozen / "outputs"),
                            label="my-label")
    assert "my-label" in res[0]["out_dir"]


def test_n_sigma_outside_is_zero_free_and_positive(frozen):
    july = _cell("HK", "VWAP", 500, -10.0, 60.0, "2030-07-01", 15)
    res = score.score_frame(july, t5cfg.CONFIG,
                            bands_dir=str(frozen / "bands"),
                            out_dir=str(frozen / "outputs"))
    scored = pd.read_csv(os.path.join(res[0]["out_dir"], "scored.csv"))
    inside = scored[scored["zone"] == "IN_RANGE"]
    assert (inside["n_sigma_outside"] == 0).all()


def test_score_curve_is_labelled_frozen_not_fitted(frozen):
    """The dashed curve on a score chart is the frozen band, not a refit.

    Mislabelling it would suggest the band adapts to each period, which is the
    opposite of what this whole workflow does.
    """
    import inspect
    src = inspect.getsource(score.score_frame)
    assert "normal_label=\"frozen band's normal\"" in src
    assert "NOT a refit" in src
