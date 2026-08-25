"""One complete report of what is missing, not one AttributeError at a time.

The first version of this guard checked a single CleanReport field, which
caught exactly the breakage that prompted it and then let the next one through
-- a stale tca/report.py surfaced as `module 'tca.report' has no attribute
'header'` a day later. A guard that only knows about yesterday's failure is a
guard you rewrite after every incident.
"""
from __future__ import annotations

import pytest

from tier5 import compat


class TestItChecksEverythingTier5Uses:
    def test_a_healthy_folder_passes(self):
        compat.check_environment()          # must not raise

    def test_every_declared_module_is_real(self):
        """The manifest must not drift into naming things that do not exist."""
        import importlib
        for mod in compat.REQUIRED_SURFACE:
            importlib.import_module(mod)

    def test_the_manifest_covers_what_the_code_actually_calls(self):
        """Guard against the manifest going stale as tier5 grows.

        Every `report.x` / `dataset.x` / `pipeline.x` reference in tier5/ must
        appear in the manifest, or the guard has a hole exactly where the next
        half-copy will land.
        """
        import pathlib, re
        used = {}
        for py in pathlib.Path("tier5").glob("*.py"):
            for mod, attr in re.findall(r"\b(report|dataset|pipeline)\.([a-zA-Z_]\w*)",
                                        py.read_text(encoding="utf-8")):
                if attr == "py":       # "pipeline.py" in prose
                    continue
                used.setdefault(f"tca.{mod}", set()).add(attr)
        for mod, attrs in used.items():
            declared = set(compat.REQUIRED_SURFACE.get(mod, ()))
            assert attrs <= declared, f"{mod}: {attrs - declared} not in manifest"


class TestItReportsAllOfThemAtOnce:
    def test_a_missing_attribute_is_named(self, monkeypatch):
        import tca.report
        monkeypatch.delattr(tca.report, "header")
        with pytest.raises(compat.VersionSkewError, match="header"):
            compat.check_environment()

    def test_two_missing_attributes_are_both_named(self, monkeypatch):
        import tca.report, tca.dataset
        monkeypatch.delattr(tca.report, "header")
        monkeypatch.delattr(tca.dataset, "out_path")
        with pytest.raises(compat.VersionSkewError) as exc:
            compat.check_environment()
        msg = str(exc.value)
        assert "header" in msg and "out_path" in msg, \
            "stopped at the first problem instead of listing them all"

    def test_a_missing_schema_constant_is_caught(self, monkeypatch):
        import tca.schema
        monkeypatch.delattr(tca.schema, "PERF_IN_SPREADS")
        with pytest.raises(compat.VersionSkewError, match="PERF_IN_SPREADS"):
            compat.check_environment()

    def test_a_stale_cleanreport_is_still_caught(self, monkeypatch):
        import tca.pipeline, dataclasses
        stale = [f for f in dataclasses.fields(tca.pipeline.CleanReport)
                 if f.name != "metric_supplied"]
        fake = dataclasses.make_dataclass(
            "CleanReport", [(f.name, f.type) for f in stale])
        monkeypatch.setattr(tca.pipeline, "CleanReport", fake)
        with pytest.raises(compat.VersionSkewError, match="metric_supplied"):
            compat.check_environment()


class TestTheMessageIsActionable:
    def test_it_shows_where_the_module_was_loaded_from(self, monkeypatch):
        """A foreign `tca` on sys.path looks identical to a stale copy until
        you can see the path it came from."""
        import tca.report
        monkeypatch.delattr(tca.report, "header")
        with pytest.raises(compat.VersionSkewError) as exc:
            compat.check_environment()
        assert "report.py" in str(exc.value)

    def test_it_says_what_to_do(self, monkeypatch):
        import tca.report
        monkeypatch.delattr(tca.report, "header")
        with pytest.raises(compat.VersionSkewError, match=r"copy|pull"):
            compat.check_environment()


class TestItFiresBeforeAnyWork:
    """A guard that runs after the CSV is read has already wasted the minute."""

    @pytest.mark.parametrize("entry", ["tier5.fit", "tier5.score", "tier5.batch"])
    def test_the_entry_point_refuses_without_reading_the_extract(
            self, entry, monkeypatch, tmp_path):
        import importlib, sys
        import tca.report, tca.dataset
        mod = importlib.import_module(entry)

        def _boom(*a, **k):
            raise AssertionError("read the extract before checking the folder")

        monkeypatch.setattr(tca.dataset, "load_prepared", _boom)
        monkeypatch.setattr(tca.report, "header", _boom)
        monkeypatch.delattr(tca.report, "zone_summary")
        monkeypatch.setattr(sys, "argv", [entry, "--dir", str(tmp_path)]
                            if entry == "tier5.batch" else [entry])
        with pytest.raises(compat.VersionSkewError):
            mod.main()
