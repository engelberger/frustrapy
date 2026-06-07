"""The native C++ / CUDA frustration backend.

``NativeBackend`` reimplements the AWSEM energy reductions that :class:`LammpsBackend`
gets from the precompiled binary, using the compiled ``frustrapy_native`` extension (the
``native/`` subproject: CPU now, optional CUDA later). It is parity-gated against the
``lammps`` reference.

The extension is an optional build artifact, so the import is lazy and graceful: nothing
is imported at module load, and ``compute_energies`` raises a clear, actionable error if
the extension was never built. Installing ``frustrapy`` without a compiler still works and
the ``lammps`` default is unaffected.

N1 status: the binding surface and geometry kernels exist; the energy reduction
(``frustrapy_native.compute_frustration``) is a stub until N2. ``compute_energies``
therefore raises ``NotImplementedError`` until the N2 reductions land.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import FrustrationBackend

if TYPE_CHECKING:
    from ..core import Pdb
    from ..analysis.frustration_calculator import FrustrationCalculator


def _load_native():
    """Import the compiled native core, or raise an actionable error if unbuilt."""
    try:
        import frustrapy_native  # noqa: PLC0415 - optional, intentionally lazy
    except ImportError as exc:  # pragma: no cover - exercised only without the build
        raise ImportError(
            "The native frustration core is not built. Install it with "
            "`pip install ./native` (CPU) or "
            "`pip install ./native -C cmake.define.FRUSTRAPY_NATIVE_CUDA=ON` (CUDA). "
            "See docs/NATIVE_BACKEND_DESIGN.md."
        ) from exc
    return frustrapy_native


class NativeBackend(FrustrationBackend):
    """Native C++/CUDA energy backend (parity-gated against ``lammps``).

    The on-disk output contract is identical to the ``lammps`` path: this backend writes
    ``tertiary_frustration.dat`` in the same column layout, and the shared
    :meth:`process_results` / :meth:`compute_density` post-processing run unchanged.
    """

    name = "native"

    def compute_energies(self, calculator: "FrustrationCalculator", pdb: "Pdb") -> None:
        """Run the native energy reduction, writing ``tertiary_frustration.dat``.

        N1: the reduction is not yet implemented (N2 lands it). Importing the extension
        is still validated here so a missing build reports a clear error rather than a
        late failure.
        """
        _load_native()  # validate the extension is present (clear error if not)
        raise NotImplementedError(
            "NativeBackend energy reduction is not implemented yet (N2). The native core "
            "skeleton (N1) ships the build and binding surface only. Use backend='lammps' "
            "for the reference path."
        )
