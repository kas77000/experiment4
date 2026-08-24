import os

import pandas as pd
import pytest

from tca import schema
from tier5 import cells


def _df(rows):
    return pd.DataFrame(rows)


def test_cells_splits_by_region_and_strategy():
    df = _df([
        {schema.MARKET: "HK", schema.ALGO: "VWAP"},
        {schema.MARKET: "HK", schema.ALGO: "VWAP"},
        {schema.MARKET: "HK", schema.ALGO: "TWAP"},
        {schema.MARKET: "JP", schema.ALGO: "VWAP"},
    ])
    got = [(r, s, len(g)) for r, s, g in cells.cells(df)]
    assert got == [("HK", "TWAP", 1), ("HK", "VWAP", 2), ("JP", "VWAP", 1)]


def test_cells_is_sorted():
    df = _df([
        {schema.MARKET: "JP", schema.ALGO: "VWAP"},
        {schema.MARKET: "AU", schema.ALGO: "TWAP"},
    ])
    assert [(r, s) for r, s, _ in cells.cells(df)] == [("AU", "TWAP"), ("JP", "VWAP")]


def test_cells_uppercases_and_strips_region():
    df = _df([{schema.MARKET: " hk ", schema.ALGO: " VWAP "}])
    r, s, _ = cells.cells(df)[0]
    assert (r, s) == ("HK", "VWAP")


def test_period_label_single_month():
    df = _df([{schema.ORDER_DATE: "2026-07-03"}, {schema.ORDER_DATE: "2026-07-28"}])
    assert cells.period_label(df) == "2026-07"


def test_period_label_range():
    df = _df([{schema.ORDER_DATE: "2025-06-02"}, {schema.ORDER_DATE: "2026-05-29"}])
    assert cells.period_label(df) == "2025-06_2026-05"


def test_period_label_none_without_dates():
    assert cells.period_label(_df([{schema.MARKET: "HK"}])) is None


def test_period_label_none_when_all_dates_unparseable():
    df = _df([{schema.ORDER_DATE: "not a date"}])
    assert cells.period_label(df) is None


def test_safe_sanitises():
    assert cells.safe("VWAP/Passive") == "VWAP_Passive"
    assert cells.safe("") == "UNKNOWN"


def test_band_path_nests_by_region_then_strategy():
    p = cells.band_path("bands", "HK", "VWAP")
    assert p == os.path.join("bands", "HK", "VWAP.json")


def test_out_dir_nests():
    p = cells.out_dir("outputs", "score", "2026-07", "HK", "VWAP")
    assert p == os.path.join("outputs", "score", "2026-07", "HK", "VWAP")


@pytest.mark.parametrize("a_lo,a_hi,b_lo,b_hi,expected", [
    ("2025-06-01", "2026-05-31", "2026-07-01", "2026-07-31", False),
    ("2025-06-01", "2026-05-31", "2026-05-15", "2026-07-31", True),
    ("2025-06-01", "2026-05-31", "2025-06-01", "2026-05-31", True),
])
def test_windows_overlap(a_lo, a_hi, b_lo, b_hi, expected):
    ts = pd.Timestamp
    assert cells.windows_overlap(ts(a_lo), ts(a_hi), ts(b_lo), ts(b_hi)) is expected


def test_windows_overlap_false_when_any_bound_missing():
    assert cells.windows_overlap(None, None, pd.Timestamp("2026-07-01"),
                                 pd.Timestamp("2026-07-31")) is False
