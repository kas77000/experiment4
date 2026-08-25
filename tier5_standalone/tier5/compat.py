"""Is the tca/ half of this folder the same vintage as the tier5/ half?

tier5_standalone is copied between machines as a folder, not installed as a
package, and the failure mode of a folder copy is that some of it lands and
some of it does not. The result imports cleanly and runs: it loads the extract,
cleans the book, prints a header nobody would question -- and then dies on an
AttributeError inside the reporting, several seconds AFTER fit_frame has
already written the band files. The run reads as failed and the bands directory
reads as fitted, which is the worst pairing available.

The check has to live on the tier5 side. A guard inside tca/ cannot fire when
tca/ is the stale half, because the guard would be stale too.

This is deliberately not a version number. Nobody bumps a version string in a
folder they copy with Explorer, so it would be wrong within a week. Asking each
module what it can actually do is true by construction.
"""

from __future__ import annotations

import dataclasses
import importlib

# Attributes tier5 reads off a CleanReport. Each arrived alongside a tier5
# feature that fails without it, so absence means the two halves disagree.
REQUIRED_REPORT_FIELDS = (
    "dropped_no_metric",   # rows whose supplied metric column was null
    "metric_supplied",     # whether the metric came from the extract
)

# Everything tier5 reaches for across the tca boundary.
#
# The first version of this module checked one CleanReport field. It caught the
# breakage that prompted it and then let the next one straight through, a day
# later, as
#
#     AttributeError: module 'tca.report' has no attribute 'header'
#
# A guard that only knows about yesterday's failure is one you rewrite after
# every incident. So this is the whole surface, and a test walks tier5/*.py and
# fails if the code starts calling something this manifest does not list --
# otherwise the hole reopens quietly the next time tier5 grows.
REQUIRED_TCA_SURFACE = {
    "tca.report":   ("header", "frame", "zone_summary"),
    "tca.dataset":  ("load_prepared", "add_common_args", "out_path"),
    "tca.pipeline": ("prepare", "CleanReport"),
    "tca.schema":   ("ADV_BUCKET", "ALGO", "AUCTION_PCT", "DURATION_MIN",
                     "MARKET", "MOMENTUM_BPS", "NOTIONAL", "ORDER_DATE",
                     "ORDER_ID", "PARTICIPATION", "PASSIVE_FILL_PCT",
                     "PCT_ADV", "PERF_IN_SPREADS", "PERF_NORM",
                     "REVERSION_BPS", "SIDE", "SLIPPAGE_BPS", "SPREAD_BPS",
                     "SYMBOL", "VOLATILITY"),
}

_FIX = """
  Nothing was fitted. Copy the WHOLE tier5_standalone folder across again --
  tca/ and tier5/ are two halves of one release and cannot be updated
  separately -- then delete every __pycache__ directory and re-run.

  If this folder is a git clone, `git pull` instead.

  If the paths above point somewhere unexpected, a different `tca` package is
  shadowing this one on sys.path."""


class VersionSkewError(RuntimeError):
    """tier5/ and tca/ came from different copies of the folder."""


def check_environment() -> None:
    """Verify the whole tca surface before any work starts.

    Collects EVERY missing name rather than stopping at the first, because a
    folder missing `report.header` is usually missing three other things too,
    and finding them one run at a time is how an afternoon disappears.

    Each module's file path goes in the message. A foreign `tca` package
    earlier on sys.path raises exactly the same AttributeError as a stale copy,
    and the path is the only thing that tells them apart.
    """
    problems, paths = [], []
    for mod_name, attrs in REQUIRED_TCA_SURFACE.items():
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:                        # noqa: BLE001
            problems.append(f"    {mod_name} -- will not import: {exc}")
            continue

        paths.append(f"    {mod_name:<14} {getattr(mod, '__file__', '?')}")
        missing = [a for a in attrs if not hasattr(mod, a)]
        if missing:
            problems.append(f"    {mod_name} is missing: {', '.join(missing)}")

        if mod_name == "tca.pipeline" and hasattr(mod, "CleanReport"):
            have = {f.name for f in dataclasses.fields(mod.CleanReport)}
            gone = [f for f in REQUIRED_REPORT_FIELDS if f not in have]
            if gone:
                problems.append("    tca.pipeline.CleanReport is missing: "
                                + ", ".join(gone))

    if not problems:
        return
    raise VersionSkewError(
        "This folder is half-copied: tier5/ and tca/ are not the same release."
        "\n\n" + "\n".join(problems)
        + "\n\n  Loaded from:\n" + "\n".join(paths)
        + "\n" + _FIX)


def check_report(report) -> None:
    """The same check against a CleanReport INSTANCE, once one exists.

    check_environment inspects the class and runs first. This catches the
    narrower case of an object built by an older code path, and is what the
    batch loader calls per file.
    """
    missing = [f for f in REQUIRED_REPORT_FIELDS if not hasattr(report, f)]
    if not missing:
        return
    raise VersionSkewError(
        "This folder is half-copied: tier5/ is newer than tca/.\n\n"
        f"    tca.pipeline.CleanReport is missing: {', '.join(missing)}\n"
        + _FIX)
