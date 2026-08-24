import numpy as np
import pandas as pd

import config
import synthetic_data
from tca import pipeline, schema
from tier5 import band, normality


def test_order_date_in_schema():
    assert schema.ORDER_DATE == "order_date"
    assert schema.ORDER_DATE not in schema.NUMERIC
    assert schema.ORDER_DATE not in schema.ESSENTIAL


def test_date_is_mapped():
    assert config.COLUMN_MAP[schema.ORDER_DATE] == "Date"


def test_region_names():
    assert set(config.REGION_NAMES) == {"AU", "HK", "JP", "IN"}


def test_synthetic_data_emits_dates():
    df = synthetic_data.generate(n=500, seed=1)
    assert "Date" in df.columns
    d = pd.to_datetime(df["Date"])
    assert d.min().year == 2025
    assert d.max().year == 2026


def test_date_survives_prepare():
    raw = synthetic_data.generate(n=500, seed=1)
    df, _ = pipeline.prepare(raw, config.COLUMN_MAP, config.DATA,
                             config.SLIPPAGE_SIGN,
                             pre_transform=config.PRE_TRANSFORM)
    assert schema.ORDER_DATE in df.columns
    assert df[schema.ORDER_DATE].notna().all()


def test_no_evaluate_imported():
    import tier5.run
    assert not hasattr(tier5.run, "evaluate")


def test_band_self_check_still_passes():
    x = np.random.default_rng(11).normal(-8.7, 18.4, 200_000)
    e = band.estimates(x, 3.0)
    assert abs(e["centre_classical"] - (-8.7)) < 0.20
    assert abs(e["scale_classical"] - 18.4) < 0.20
    outside = float(np.mean((x < e["lo_classical"]) | (x > e["hi_classical"])))
    assert abs(outside - (1 - normality.promised_inside(3.0))) < 0.0006
