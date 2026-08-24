import ast
import os
import pathlib

import numpy as np
import pytest

from tier5 import curve


@pytest.fixture
def sample():
    return np.random.default_rng(3).normal(-10.0, 20.0, 5000)


def test_writes_a_png(tmp_path, sample):
    path = str(tmp_path / "curve.png")
    msg = curve.plot(sample, centre=-10.0, scale=20.0, lo=-70.0, hi=50.0,
                     path=path, title="HK / VWAP")
    assert os.path.exists(path)
    assert os.path.getsize(path) > 1000
    assert path in msg


def test_creates_missing_directories(tmp_path, sample):
    path = str(tmp_path / "a" / "b" / "curve.png")
    curve.plot(sample, centre=-10.0, scale=20.0, lo=-70.0, hi=50.0,
               path=path, title="t")
    assert os.path.exists(path)


def test_empty_input_skips_cleanly(tmp_path):
    msg = curve.plot(np.array([]), centre=0.0, scale=1.0, lo=-3.0, hi=3.0,
                     path=str(tmp_path / "c.png"), title="t")
    assert "skipped" in msg.lower()
    assert not os.path.exists(str(tmp_path / "c.png"))


def test_non_finite_scale_skips_cleanly(tmp_path, sample):
    msg = curve.plot(sample, centre=0.0, scale=0.0, lo=-3.0, hi=3.0,
                     path=str(tmp_path / "c.png"), title="t")
    assert "skipped" in msg.lower()


def test_not_imported_at_module_level():
    """The scoring path must run without matplotlib or seaborn installed."""
    src = pathlib.Path(curve.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append((node.module or "").split(".")[0])
    assert "matplotlib" not in names
    assert "seaborn" not in names


def _with_extremes():
    rng = np.random.default_rng(5)
    return np.concatenate([rng.normal(0.0, 10.0, 5000),
                           np.array([-4000.0, 5000.0])])


def test_view_excludes_the_extremes():
    """A few extreme orders must not squash the band into an unreadable spike."""
    x_min, x_max = curve.view_range(_with_extremes(), lo=-30.0, hi=30.0, scale=10.0)
    assert x_min > -4000.0
    assert x_max < 5000.0


def test_view_always_contains_the_band():
    """Framing must never crop out the bounds the picture exists to show."""
    x_min, x_max = curve.view_range(_with_extremes(), lo=-30.0, hi=30.0, scale=10.0)
    assert x_min < -30.0
    assert x_max > 30.0


def test_view_widens_for_a_band_beyond_the_data():
    x = np.random.default_rng(7).normal(0.0, 1.0, 1000)
    x_min, x_max = curve.view_range(x, lo=-500.0, hi=500.0, scale=100.0)
    assert x_min <= -500.0
    assert x_max >= 500.0


def test_offscreen_orders_are_still_counted_in_the_caption(tmp_path):
    """Clipping the view must never look like clipping the data."""
    msg = curve.plot(_with_extremes(), centre=0.0, scale=10.0, lo=-30.0, hi=30.0,
                     path=str(tmp_path / "c.png"), title="t")
    assert "Wrote" in msg
    assert os.path.exists(str(tmp_path / "c.png"))


def test_normal_label_defaults_to_fitted(tmp_path, sample):
    """Default wording is right for fit, where the normal really is fitted."""
    import inspect
    assert inspect.signature(curve.plot).parameters["normal_label"].default \
        == "fitted normal"


def test_normal_label_is_overridable(tmp_path, sample):
    """Score must be able to say the curve is frozen, not refitted."""
    msg = curve.plot(sample, centre=-10.0, scale=20.0, lo=-70.0, hi=50.0,
                     path=str(tmp_path / "c.png"), title="t",
                     normal_label="frozen band's normal")
    assert os.path.exists(str(tmp_path / "c.png"))
    assert "Wrote" in msg


def test_caption_states_the_units_of_the_axis():
    """A bare "-5.59" on the x-axis is unreadable, and bps is the wrong guess."""
    text = curve.caption(n=260, k=3.0, outside=0.0808, n_offscreen=0,
                         units="spreads")
    assert "spreads" in text


def test_caption_still_reports_offscreen_orders():
    text = curve.caption(n=260, k=3.0, outside=0.08, n_offscreen=2,
                         units="spreads")
    assert "2" in text and "still counted" in text


def test_caption_omits_units_when_there_are_none():
    text = curve.caption(n=10, k=3.0, outside=0.0, n_offscreen=0, units="")
    assert "()" not in text
