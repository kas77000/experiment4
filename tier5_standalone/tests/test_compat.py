"""A half-copied folder must say so, not crash 40 seconds in.

tier5/ and tca/ are two halves of one release. Copying only one of them leaves
a folder that imports cleanly, cleans the book, prints a plausible header --
and then dies on an AttributeError deep in the reporting, AFTER the bands have
already been written to disk. The run looks failed and the output looks fitted.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from tca import pipeline
from tier5 import compat


@dataclass
class _Old:
    """A CleanReport from before the provenance commit."""
    rows_in: int = 10
    dropped_no_metric: int = 0


@dataclass
class _Ancient:
    rows_in: int = 10


class TestCheckReport:
    def test_a_current_report_passes(self):
        rep = pipeline.CleanReport(rows_in=1, dropped_missing=0,
                                   dropped_bad_spread=0, dropped_dust=0,
                                   dropped_data_error=0, rows_out=1)
        compat.check_report(rep)          # must not raise

    def test_a_stale_tca_is_refused(self):
        with pytest.raises(compat.VersionSkewError):
            compat.check_report(_Old())

    def test_it_names_the_missing_field(self):
        with pytest.raises(compat.VersionSkewError, match="metric_supplied"):
            compat.check_report(_Old())

    def test_it_names_every_missing_field_not_just_the_first(self):
        with pytest.raises(compat.VersionSkewError) as exc:
            compat.check_report(_Ancient())
        assert "metric_supplied" in str(exc.value)
        assert "dropped_no_metric" in str(exc.value)

    def test_it_says_which_half_is_stale(self):
        with pytest.raises(compat.VersionSkewError, match="tca"):
            compat.check_report(_Old())

    def test_it_says_what_to_do_about_it(self):
        with pytest.raises(compat.VersionSkewError, match=r"copy|recopy|pull"):
            compat.check_report(_Old())


class TestWiredIntoTheEntryPoints:
    """The check is worthless if a caller can skip it."""

    @pytest.mark.parametrize("mod", ["tier5.fit", "tier5.score", "tier5.batch"])
    def test_each_entry_point_checks(self, mod):
        import importlib
        src = importlib.import_module(mod)
        assert "compat" in src.__dict__, f"{mod} does not import compat"


class TestItActuallyFires:
    """Importing the module proves nothing; the call has to happen."""

    def _stale(self, monkeypatch, mod):
        """Make load_prepared hand back a pre-provenance CleanReport."""
        import tca.dataset
        import pandas as pd
        df = pd.DataFrame({"perf_in_spreads": [0.1, 0.2]})
        monkeypatch.setattr(tca.dataset, "load_prepared",
                            lambda args, quiet=False: (df, _Old()))

    def test_fit_refuses_before_it_fits_anything(self, monkeypatch, tmp_path):
        from tier5 import fit
        self._stale(monkeypatch, fit)
        monkeypatch.setattr("sys.argv",
                            ["fit", "--bands-dir", str(tmp_path / "bands")])
        with pytest.raises(compat.VersionSkewError):
            fit.main()
        assert not (tmp_path / "bands").exists(), "wrote bands before refusing"

    def test_score_refuses_too(self, monkeypatch, tmp_path):
        from tier5 import score
        self._stale(monkeypatch, score)
        monkeypatch.setattr("sys.argv",
                            ["score", "--bands-dir", str(tmp_path / "bands")])
        with pytest.raises(compat.VersionSkewError):
            score.main()
