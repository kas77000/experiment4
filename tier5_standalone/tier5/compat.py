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
folder they copy with Explorer, so it would be wrong within a week. Asking the
object what it can actually do is true by construction.
"""

from __future__ import annotations

# Attributes tier5 reads off a CleanReport. Each was added alongside a tier5
# feature that fails without it, so absence means the two halves disagree.
REQUIRED_REPORT_FIELDS = (
    "dropped_no_metric",   # rows whose supplied metric column was null
    "metric_supplied",     # whether the metric came from the extract
)


class VersionSkewError(RuntimeError):
    """tier5/ and tca/ came from different copies of the folder."""


def check_report(report) -> None:
    """Refuse before any fitting if tca/ is older than tier5/.

    Raised early and loudly, because the alternative is a traceback that names
    an attribute rather than the mistake, after the damage is on disk.
    """
    missing = [f for f in REQUIRED_REPORT_FIELDS if not hasattr(report, f)]
    if not missing:
        return
    raise VersionSkewError(
        "This folder is half-copied: tier5/ is newer than tca/.\n\n"
        f"  tier5 needs CleanReport.{', CleanReport.'.join(missing)}, which "
        f"this tca/pipeline.py\n  does not define.\n\n"
        "  Nothing was fitted. Copy the WHOLE tier5_standalone folder across "
        "again --\n"
        "  tca/ and tier5/ are two halves of one release and cannot be "
        "updated\n  separately -- then delete every __pycache__ directory and "
        "re-run.\n\n"
        "  If this folder is a git clone, `git pull` instead."
    )
