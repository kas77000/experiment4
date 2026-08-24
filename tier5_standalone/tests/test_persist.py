import json

import numpy as np
import pandas as pd
import pytest

from tca import schema
from tier5 import band, config as t5cfg, persist


def _fixture(tmp_path, n=2000, seed=5):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        schema.SLIPPAGE_BPS: rng.normal(-10.0, 20.0, n),
        schema.SPREAD_BPS: rng.uniform(5.0, 15.0, n),
        schema.PCT_ADV: rng.uniform(0.1, 5.0, n),
        schema.VOLATILITY: rng.uniform(100.0, 250.0, n),
        schema.DURATION_MIN: rng.uniform(10.0, 300.0, n),
        schema.ORDER_DATE: pd.bdate_range("2025-06-02", periods=n).astype(str),
    })
    cfg = t5cfg.CONFIG
    est = band.estimates(df[cfg.metric].to_numpy(), cfg.k_sigma)
    path = str(tmp_path / "HK" / "VWAP.json")
    persist.save(est, cfg, path, region="HK", strategy="VWAP",
                 source_csv="year.csv", period="2025-06_2033-01",
                 df=df, flag_rate_pct=1.45)
    return path, est, cfg, df


def test_save_writes_nested_file(tmp_path):
    path, _, _, _ = _fixture(tmp_path)
    assert path.endswith("VWAP.json")
    with open(path) as fh:
        assert json.load(fh)["region"] == "HK"


def test_roundtrip_preserves_bounds_exactly(tmp_path):
    path, est, cfg, _ = _fixture(tmp_path)
    loaded, _, _ = persist.load(path, cfg)
    assert loaded["lo"] == est["lo_classical"]
    assert loaded["hi"] == est["hi_classical"]
    assert loaded["centre"] == est["centre_classical"]
    assert loaded["scale"] == est["scale_classical"]


def test_both_estimators_stored(tmp_path):
    path, est, cfg, _ = _fixture(tmp_path)
    loaded, _, _ = persist.load(path, cfg)
    assert loaded["lo_robust"] == est["lo_robust"]
    assert loaded["hi_robust"] == est["hi_robust"]


def test_reference_carries_shape_and_required_k(tmp_path):
    path, _, cfg, _ = _fixture(tmp_path)
    _, _, ref = persist.load(path, cfg)
    assert ref["flag_rate_pct"] == 1.45
    assert np.isfinite(ref["k_required"])
    assert "spread_bps" in ref["feature_medians"]


def test_fit_window_stamped(tmp_path):
    path, _, cfg, _ = _fixture(tmp_path)
    loaded, _, _ = persist.load(path, cfg)
    assert loaded["fit_date_min"] == "2025-06-02"
    assert loaded["fit_date_max"] is not None


def test_scoring_config_travels(tmp_path):
    path, _, cfg, _ = _fixture(tmp_path)
    _, loaded_cfg, _ = persist.load(path, cfg)
    assert loaded_cfg.k_sigma == cfg.k_sigma
    assert loaded_cfg.metric == cfg.metric
    assert loaded_cfg.estimator == cfg.estimator


def test_format_version_mismatch_raises(tmp_path):
    path, _, cfg, _ = _fixture(tmp_path)
    with open(path) as fh:
        payload = json.load(fh)
    payload["format_version"] = 999
    with open(path, "w") as fh:
        json.dump(payload, fh)
    with pytest.raises(ValueError, match="format version"):
        persist.load(path, cfg)


def test_drift_report_flags_moved_median(tmp_path):
    path, _, cfg, df = _fixture(tmp_path)
    _, _, ref = persist.load(path, cfg)
    moved = df.copy()
    moved[schema.SPREAD_BPS] = moved[schema.SPREAD_BPS] * 2.0
    scored = moved.assign(flagged=False)
    table, warnings = persist.drift_report(moved, scored, ref, cfg)
    assert len(table)
    assert any("spread_bps" in w for w in warnings)


def test_drift_report_quiet_when_nothing_moved(tmp_path):
    path, _, cfg, df = _fixture(tmp_path)
    _, _, ref = persist.load(path, cfg)
    flagged = np.zeros(len(df), dtype=bool)
    flagged[:int(0.0145 * len(df))] = True
    scored = df.assign(flagged=flagged)
    _, warnings = persist.drift_report(df, scored, ref, cfg)
    assert warnings == []
