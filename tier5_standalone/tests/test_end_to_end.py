"""The round trip: freeze on one book, score another, prove nothing leaked."""

import json

import numpy as np
import pytest

import config
import synthetic_data
from tca import pipeline, schema
from tier5 import cells, config as t5cfg, fit, score


def _prep(raw):
    df, _ = pipeline.prepare(raw, config.COLUMN_MAP, config.DATA,
                             config.SLIPPAGE_SIGN,
                             pre_transform=config.PRE_TRANSFORM)
    return df


def test_full_round_trip(tmp_path):
    year = _prep(synthetic_data.generate(n=6000, seed=7,
                                         start_date="2025-06-02",
                                         end_date="2026-05-29"))
    later = _prep(synthetic_data.generate(n=1200, seed=8,
                                          start_date="2026-07-01",
                                          end_date="2026-07-31"))

    fit_res = fit.fit_frame(year, t5cfg.CONFIG,
                            bands_dir=str(tmp_path / "bands"),
                            out_dir=str(tmp_path / "outputs"),
                            source_csv="year.csv")
    assert any(not r["skipped"] for r in fit_res)

    score_res = score.score_frame(later, t5cfg.CONFIG,
                                  bands_dir=str(tmp_path / "bands"),
                                  out_dir=str(tmp_path / "outputs"))
    scored_cells = [r for r in score_res if not r["skipped"]]
    assert scored_cells

    # The band that scored July is byte-identical to the one fit froze.
    for r in scored_cells:
        with open(cells.band_path(str(tmp_path / "bands"),
                                  r["region"], r["strategy"])) as fh:
            frozen = json.load(fh)
        assert r["lo"] == frozen["lo"]
        assert r["hi"] == frozen["hi"]


def test_scoring_the_fit_book_itself_is_refused(tmp_path):
    year = _prep(synthetic_data.generate(n=3000, seed=7))
    fit.fit_frame(year, t5cfg.CONFIG, bands_dir=str(tmp_path / "bands"),
                  out_dir=str(tmp_path / "outputs"), source_csv="year.csv")
    with pytest.raises(score.LeakageError):
        score.score_frame(year, t5cfg.CONFIG,
                          bands_dir=str(tmp_path / "bands"),
                          out_dir=str(tmp_path / "outputs"))


def test_fits_recorded_flag_rate_matches_a_manual_classify(tmp_path):
    """fit's in-sample rate is the real thing, not an approximation."""
    year = _prep(synthetic_data.generate(n=4000, seed=7))
    res = fit.fit_frame(year, t5cfg.CONFIG, bands_dir=str(tmp_path / "bands"),
                        out_dir=str(tmp_path / "outputs"), source_csv="y.csv")
    for r in res:
        if r["skipped"]:
            continue
        g = year[(year[schema.MARKET] == r["region"])
                 & (year[schema.ALGO] == r["strategy"])]
        x = g[t5cfg.CONFIG.metric].to_numpy()
        manual = 100.0 * float(np.mean((x < r["lo"]) | (x > r["hi"])))
        assert abs(manual - r["flag_rate_pct"]) < 1e-9


def test_a_wider_later_period_flags_more_without_moving_the_band(tmp_path):
    """The headline claim: the band measures, it does not adapt."""
    year = _prep(synthetic_data.generate(n=6000, seed=7,
                                         start_date="2025-06-02",
                                         end_date="2026-05-29"))
    fit.fit_frame(year, t5cfg.CONFIG, bands_dir=str(tmp_path / "bands"),
                  out_dir=str(tmp_path / "outputs"), source_csv="y.csv")

    later = _prep(synthetic_data.generate(n=2000, seed=8,
                                          start_date="2026-07-01",
                                          end_date="2026-07-31"))
    # Blow out the BANDED metric so the later period is genuinely worse.
    later[t5cfg.CONFIG.metric] = later[t5cfg.CONFIG.metric] * 3.0

    res = score.score_frame(later, t5cfg.CONFIG,
                            bands_dir=str(tmp_path / "bands"),
                            out_dir=str(tmp_path / "outputs"))
    scored = [r for r in res if not r["skipped"]]
    assert scored
    for r in scored:
        with open(cells.band_path(str(tmp_path / "bands"),
                                  r["region"], r["strategy"])) as fh:
            frozen = json.load(fh)
        assert r["lo"] == frozen["lo"]          # band did not move
        assert r["flag_rate_pct"] > r["fit_flag_rate_pct"]   # but the rate did


def test_folder_has_no_reference_to_anything_outside_it():
    """The folder must stand alone: no tier3, no old package name, no evaluate."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    here = pathlib.Path(__file__).resolve()
    needles = ("tier3_model", "tier5_gaussian", "tca.evaluate",
               "from tca import evaluate")
    offenders = []
    for path in root.rglob("*.py"):
        if "__pycache__" in str(path) or path.resolve() == here:
            continue   # this file names the needles in order to look for them
        src = path.read_text(encoding="utf-8")
        offenders += [f"{path.relative_to(root)}: {n}" for n in needles if n in src]
    assert offenders == [], offenders
