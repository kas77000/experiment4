"""Prove the banded metric came from the extract, don't claim it.

If the strategy's source column is missing, the pipeline falls back to deriving
slippage_bps / spread_bps. That fallback produces a number of the right
magnitude against the WRONG benchmark, so it cannot be caught by eye -- and a
run that prints "VWAP -> ePvwap/Sprd" while quietly having derived it is worse
than one that says nothing.
"""

import sys

import numpy as np
import pandas as pd

import config
from tca import pipeline, schema


def _raw(n=400, with_metric=True, strategy="VWAP"):
    d = {
        "aggrTgtId": [f"O{i}" for i in range(n)],
        "Sym": "0700 HK", "Strategy": strategy,
        "Date": pd.bdate_range("2025-06-02", periods=n).astype(str),
        "Pvwap": np.linspace(-40.0, 40.0, n),
        "Sprd": np.full(n, 10.0),
    }
    if with_metric:
        d["ePvwap/Sprd"] = np.linspace(-2.0, 2.0, n)
    return pd.DataFrame(d)


def _prep(raw):
    return pipeline.prepare(raw, config.COLUMN_MAP, config.DATA,
                            config.SLIPPAGE_SIGN,
                            pre_transform=config.PRE_TRANSFORM)


# --------------------------------------------------------------------------
# the pipeline knows which happened, so it must say
# --------------------------------------------------------------------------

def test_supplied_metric_is_recorded_as_supplied():
    _, rep = _prep(_raw(with_metric=True))
    assert rep.metric_supplied is True


def test_derived_metric_is_recorded_as_derived():
    _, rep = _prep(_raw(with_metric=False))
    assert rep.metric_supplied is False


def test_the_cleaning_report_says_which():
    _, rep = _prep(_raw(with_metric=False))
    assert "derived" in rep.as_text().lower()


def test_the_cleaning_report_says_so_when_supplied():
    _, rep = _prep(_raw(with_metric=True))
    assert "supplied" in rep.as_text().lower()


def test_the_two_paths_really_do_differ():
    """Guards the premise: a silent fallback changes the numbers."""
    a, _ = _prep(_raw(with_metric=True))
    b, _ = _prep(_raw(with_metric=False))
    assert a[schema.PERF_IN_SPREADS].max() != b[schema.PERF_IN_SPREADS].max()


# --------------------------------------------------------------------------
# the metric-source block must stop claiming what it has not checked
# --------------------------------------------------------------------------

def test_source_lines_warn_loudly_when_the_metric_was_derived():
    lines = "\n".join(config.metric_source_lines(["VWAP"], supplied=False))
    assert "NOT" in lines or "not found" in lines.lower()
    assert "slippage_bps" in lines


def test_source_lines_name_the_column_when_it_was_supplied():
    lines = "\n".join(config.metric_source_lines(["VWAP"], supplied=True))
    assert "ePvwap/Sprd" in lines
    assert "not found" not in lines.lower()


def test_source_lines_default_to_claiming_nothing_unverified():
    """An unknown provenance must not print as a confirmation."""
    lines = "\n".join(config.metric_source_lines(["VWAP"]))
    assert "ePvwap/Sprd" in lines


# --------------------------------------------------------------------------
# end to end: the run itself has to tell you
# --------------------------------------------------------------------------

def _run_fit(tmp_path, raw, capsys):
    from tier5 import fit
    csv = str(tmp_path / "year.csv")
    raw.to_csv(csv, index=False)
    argv = sys.argv
    sys.argv = ["fit", "--csv", csv, "--bands-dir", str(tmp_path / "bands"),
                "--out-dir", str(tmp_path / "outputs")]
    try:
        fit.main()
    finally:
        sys.argv = argv
    return capsys.readouterr().out


def test_fit_confirms_the_column_when_it_is_there(tmp_path, capsys):
    out = _run_fit(tmp_path, _raw(n=800, with_metric=True), capsys)
    assert "ePvwap/Sprd" in out
    assert "not found" not in out.lower()


def test_fit_refuses_to_claim_a_column_it_never_read(tmp_path, capsys):
    out = _run_fit(tmp_path, _raw(n=800, with_metric=False), capsys)
    assert "not found" in out.lower()
    assert "slippage_bps" in out
