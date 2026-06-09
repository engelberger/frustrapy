"""One-time non-commercial license notice for PyRosetta-backed code paths.

PyRosetta is distributed by RosettaCommons and is licensed for academic and
non-commercial use only. FrustraPy uses it (when installed and selected) in two
optional places: the ``atomic`` REF2015 energy backend and the ``pyrosetta``
side-chain mutation backend. Neither is a core dependency; the default
lammps/native/tmol energy paths and the threading/modeller mutation paths never
import PyRosetta.

This leaf module emits a single, clear notice the first time any PyRosetta-backed
path actually runs in a process, so a user is never silently relying on a
non-commercially-licensed component. It imports nothing from the rest of the
package (no import cycles) and uses the standard ``logging`` machinery rather than
``print`` so it is captured and routed like every other diagnostic. The notice is
ON by default and visible even when the application configures no logging handler
(Python's last-resort handler emits WARNING and above to stderr).

The helper is wired at the one PyRosetta init boundary (``_ensure_pyrosetta`` in
``frustrapy.analysis.mutation_backends``); it is not scattered across call sites.
"""

import logging
import os

logger = logging.getLogger("frustrapy")

# Once-per-process guard. Set on the first emit (or first suppressed call) so the
# notice never repeats within a process, regardless of how many PyRosetta-backed
# operations run.
_NOTICE_EMITTED = False

# Set this environment variable (to anything other than empty/"0"/"false") to
# silence the one-time notice. It is ON by default.
SUPPRESS_ENV_VAR = "FRUSTRAPY_SUPPRESS_PYROSETTA_NOTICE"

_NOTICE = (
    "PyRosetta is being used by this FrustraPy code path (the atomic REF2015 "
    "energy backend and/or the PyRosetta side-chain mutation backend). PyRosetta "
    "is distributed by RosettaCommons and is licensed for NON-COMMERCIAL and "
    "academic use only. By running this path you are relying on a "
    "non-commercially-licensed component; commercial use requires a separate "
    "license from RosettaCommons (https://www.pyrosetta.org/, "
    "https://els2.comotion.uw.edu/product/pyrosetta). The default lammps, native, "
    "and tmol energy backends and the threading and modeller mutation backends do "
    "not use PyRosetta and carry no such restriction. Set "
    f"{SUPPRESS_ENV_VAR}=1 to silence this one-time notice."
)


def _suppressed() -> bool:
    """True when the suppression environment variable is set to an on value."""
    val = os.environ.get(SUPPRESS_ENV_VAR, "").strip().lower()
    return val not in ("", "0", "false", "no", "off")


def warn_pyrosetta_noncommercial(force: bool = False) -> None:
    """Emit the PyRosetta non-commercial license notice once per process.

    Called at the PyRosetta init boundary so it fires only when a PyRosetta-backed
    path actually runs, and never for the license-clean lammps/native/tmol or
    threading/modeller paths. Subsequent calls in the same process are no-ops. The
    notice can be silenced with the ``FRUSTRAPY_SUPPRESS_PYROSETTA_NOTICE``
    environment variable but is ON by default.

    Parameters
    ----------
    force:
        Re-emit even if it has already fired this process. Intended for tests.
    """
    global _NOTICE_EMITTED
    if _NOTICE_EMITTED and not force:
        return
    _NOTICE_EMITTED = True
    if _suppressed():
        return
    logger.warning(_NOTICE)


def _reset_notice_for_tests() -> None:
    """Reset the once-per-process guard so a test can observe the notice again."""
    global _NOTICE_EMITTED
    _NOTICE_EMITTED = False
