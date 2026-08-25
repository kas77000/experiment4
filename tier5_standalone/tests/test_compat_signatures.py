"""Presence is not compatibility: the parameter has to be there too.

The guard grew one incident at a time -- first a CleanReport field, then the
tca module surface -- and each version was scoped to the thing that had just
broken. The third failure got past both:

    TypeError: metric_source_lines() got an unexpected keyword argument
    'supplied'

Two holes at once. config.py sits at the top level, so a manifest that only
knew about tca/ never looked at it; and the name WAS present, so hasattr would
have passed it anyway. A folder is compatible when the calls tier5 makes
actually bind, not when the names exist.
"""
from __future__ import annotations

import pytest

from tier5 import compat


class TestTheRootConfigIsPartOfTheRelease:
    def test_config_is_in_the_manifest(self):
        assert "config" in compat.REQUIRED_SURFACE

    def test_a_healthy_folder_passes(self):
        compat.check_environment()

    def test_a_stale_root_config_is_caught(self, monkeypatch):
        import config
        monkeypatch.delattr(config, "PRE_TRANSFORM")
        with pytest.raises(compat.VersionSkewError, match="PRE_TRANSFORM"):
            compat.check_environment()

    def test_the_manifest_covers_what_tier5_calls(self):
        """`config.x` in tier5/ must be declared, or the hole reopens."""
        import pathlib, re
        used = set()
        for py in pathlib.Path("tier5").glob("*.py"):
            src = py.read_text(encoding="utf-8")
            for line in src.splitlines():
                if "t5cfg" in line:
                    continue
                for attr in re.findall(r"(?<!\w)config\.([a-zA-Z_]\w*)", line):
                    if attr != "py":            # "config.py" in prose
                        used.add(attr)
        declared = set(compat.REQUIRED_SURFACE.get("config", ()))
        assert used <= declared, f"not in manifest: {used - declared}"


class TestSignaturesNotJustNames:
    def test_the_exact_call_that_broke_is_covered(self):
        assert "supplied" in compat.REQUIRED_PARAMS["config.metric_source_lines"]

    def test_a_missing_keyword_argument_is_caught(self, monkeypatch):
        """The old signature: the name is present, the parameter is not."""
        import config
        monkeypatch.setattr(config, "metric_source_lines",
                            lambda strategies: [])
        with pytest.raises(compat.VersionSkewError, match="supplied"):
            compat.check_environment()

    def test_hasattr_alone_would_have_missed_it(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "metric_source_lines",
                            lambda strategies: [])
        assert hasattr(config, "metric_source_lines"), \
            "premise: the name is present, so only a signature check catches it"
        with pytest.raises(compat.VersionSkewError):
            compat.check_environment()

    def test_a_kwargs_signature_is_accepted(self, monkeypatch):
        """**kwargs absorbs anything, so it is not a mismatch."""
        import config
        monkeypatch.setattr(config, "metric_source_lines",
                            lambda strategies, **kw: [])
        compat.check_environment()

    def test_a_non_callable_where_a_function_belongs_is_caught(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "metric_source_lines", 42)
        with pytest.raises(compat.VersionSkewError):
            compat.check_environment()


class TestTheMessage:
    def test_it_shows_the_root_config_path(self, monkeypatch):
        """root config.py and tier5/config.py are different files with the
        same name; the path is what tells a reader which one is stale."""
        import config
        monkeypatch.delattr(config, "REGION_NAMES")
        with pytest.raises(compat.VersionSkewError) as exc:
            compat.check_environment()
        assert "config.py" in str(exc.value)

    def test_it_names_a_signature_problem_as_such(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "metric_source_lines", lambda s: [])
        with pytest.raises(compat.VersionSkewError,
                           match=r"older signature|does not accept"):
            compat.check_environment()


class TestThePreflightCommand:
    """`python -m tier5.compat` must be usable before anything else is run."""

    def test_it_returns_zero_on_a_healthy_folder(self, capsys):
        from tier5 import compat as c
        assert c._main() == 0
        assert "OK" in capsys.readouterr().out

    def test_it_returns_one_and_explains_on_a_bad_one(self, monkeypatch, capsys):
        import config
        from tier5 import compat as c
        monkeypatch.delattr(config, "COLUMN_MAP")
        assert c._main() == 1
        assert "COLUMN_MAP" in capsys.readouterr().out

    def test_it_prints_rather_than_raising(self, monkeypatch):
        """A traceback reads as a crash; this is a refusal with an answer."""
        import config
        from tier5 import compat as c
        monkeypatch.delattr(config, "COLUMN_MAP")
        c._main()          # must not raise
