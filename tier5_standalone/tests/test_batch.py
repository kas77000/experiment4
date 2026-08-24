import os

import numpy as np
import pandas as pd

from tier5 import batch, config as t5cfg


def _write(path, region, strategy, n, mu, sd, start, seed):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(seed)
    pd.DataFrame({
        "aggrTgtId": [f"{region}{strategy}{i}" for i in range(n)],
        "Sym": f"0001 {region}",
        "Strategy": strategy,
        "Pvwap": rng.normal(mu, sd, n),
        "Sprd": rng.uniform(5.0, 15.0, n),
        "%Adv": rng.uniform(0.1, 5.0, n),
        "Vol": rng.uniform(1.0, 2.5, n),
        "PR": rng.uniform(1.0, 20.0, n),
        "Dur": rng.uniform(10.0, 300.0, n),
        "Date": pd.bdate_range(start, periods=n).astype(str),
    }).to_csv(path, index=False)


def test_fit_then_score_across_a_directory(tmp_path):
    for region in ("HK", "JP"):
        _write(str(tmp_path / "year" / region / "VWAP.csv"),
               region, "VWAP", 600, -10.0, 20.0, "2025-06-02", 1)
    results, failures = batch.run("fit", str(tmp_path / "year"),
                                  bands_dir=str(tmp_path / "bands"),
                                  out_dir=str(tmp_path / "outputs"),
                                  cfg=t5cfg.CONFIG)
    assert failures == []
    assert len({(r["region"], r["strategy"]) for r in results}) == 2

    for region in ("HK", "JP"):
        _write(str(tmp_path / "july" / region / "VWAP.csv"),
               region, "VWAP", 200, -10.0, 20.0, "2030-07-01", 2)
    results, failures = batch.run("score", str(tmp_path / "july"),
                                  bands_dir=str(tmp_path / "bands"),
                                  out_dir=str(tmp_path / "outputs"),
                                  cfg=t5cfg.CONFIG)
    assert failures == []
    assert all(not r["skipped"] for r in results)


def test_one_broken_file_does_not_abort_the_rest(tmp_path):
    _write(str(tmp_path / "year" / "HK" / "VWAP.csv"),
           "HK", "VWAP", 600, -10.0, 20.0, "2025-06-02", 1)
    broken = tmp_path / "year" / "JP" / "VWAP.csv"
    os.makedirs(os.path.dirname(str(broken)), exist_ok=True)
    broken.write_text("this,is,not\na,valid,extract\n", encoding="utf-8")

    results, failures = batch.run("fit", str(tmp_path / "year"),
                                  bands_dir=str(tmp_path / "bands"),
                                  out_dir=str(tmp_path / "outputs"),
                                  cfg=t5cfg.CONFIG)
    assert len(failures) == 1
    assert "VWAP.csv" in failures[0]["file"]
    assert any(r["region"] == "HK" for r in results)


def test_summary_frame_has_a_row_per_cell(tmp_path):
    for region in ("HK", "JP"):
        _write(str(tmp_path / "year" / region / "VWAP.csv"),
               region, "VWAP", 600, -10.0, 20.0, "2025-06-02", 1)
    results, _ = batch.run("fit", str(tmp_path / "year"),
                           bands_dir=str(tmp_path / "bands"),
                           out_dir=str(tmp_path / "outputs"), cfg=t5cfg.CONFIG)
    summary = batch.summary_frame(results, "fit")
    assert len(summary) == 2
    assert {"region", "strategy", "n", "lo", "hi"} <= set(summary.columns)


def test_score_summary_has_vs_fit(tmp_path):
    _write(str(tmp_path / "year" / "HK" / "VWAP.csv"),
           "HK", "VWAP", 600, -10.0, 20.0, "2025-06-02", 1)
    batch.run("fit", str(tmp_path / "year"), bands_dir=str(tmp_path / "bands"),
              out_dir=str(tmp_path / "outputs"), cfg=t5cfg.CONFIG)
    _write(str(tmp_path / "july" / "HK" / "VWAP.csv"),
           "HK", "VWAP", 200, -10.0, 20.0, "2030-07-01", 2)
    results, _ = batch.run("score", str(tmp_path / "july"),
                           bands_dir=str(tmp_path / "bands"),
                           out_dir=str(tmp_path / "outputs"),
                           cfg=t5cfg.CONFIG, label="2030-07")
    summary = batch.summary_frame(results, "score")
    assert "vs_fit" in summary.columns
    assert "fit_flag_pct" in summary.columns


def test_nested_directories_are_walked_recursively(tmp_path):
    _write(str(tmp_path / "year" / "asia" / "HK" / "VWAP.csv"),
           "HK", "VWAP", 600, -10.0, 20.0, "2025-06-02", 1)
    results, failures = batch.run("fit", str(tmp_path / "year"),
                                  bands_dir=str(tmp_path / "bands"),
                                  out_dir=str(tmp_path / "outputs"),
                                  cfg=t5cfg.CONFIG)
    assert failures == []
    assert results[0]["region"] == "HK"


def test_empty_directory_is_not_an_error(tmp_path):
    os.makedirs(str(tmp_path / "empty"))
    results, failures = batch.run("fit", str(tmp_path / "empty"),
                                  bands_dir=str(tmp_path / "bands"),
                                  out_dir=str(tmp_path / "outputs"),
                                  cfg=t5cfg.CONFIG)
    assert results == []
    assert failures == []


def test_leakage_is_tagged_separately_from_errors(tmp_path):
    """A refusal is one systemic mistake, not a broken file."""
    _write(str(tmp_path / "year" / "HK" / "VWAP.csv"),
           "HK", "VWAP", 400, -10.0, 20.0, "2025-06-02", 1)
    batch.run("fit", str(tmp_path / "year"), bands_dir=str(tmp_path / "bands"),
              out_dir=str(tmp_path / "outputs"), cfg=t5cfg.CONFIG)
    # Same window again -> overlaps the fit window.
    _write(str(tmp_path / "again" / "HK" / "VWAP.csv"),
           "HK", "VWAP", 400, -10.0, 20.0, "2025-06-02", 2)
    results, failures = batch.run("score", str(tmp_path / "again"),
                                  bands_dir=str(tmp_path / "bands"),
                                  out_dir=str(tmp_path / "outputs"),
                                  cfg=t5cfg.CONFIG)
    assert results == []
    assert len(failures) == 1
    assert failures[0]["kind"] == "leakage"
    assert "overlap" in failures[0]["error"]


def test_broken_file_is_tagged_as_error_not_leakage(tmp_path):
    broken = tmp_path / "year" / "JP" / "VWAP.csv"
    os.makedirs(os.path.dirname(str(broken)), exist_ok=True)
    broken.write_text("nonsense\n1\n", encoding="utf-8")
    _, failures = batch.run("fit", str(tmp_path / "year"),
                            bands_dir=str(tmp_path / "bands"),
                            out_dir=str(tmp_path / "outputs"), cfg=t5cfg.CONFIG)
    assert failures[0]["kind"] == "error"
