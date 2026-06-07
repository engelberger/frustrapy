"""The reference frustration backend: the AWSEM/LAMMPS subprocess path.

``LammpsBackend`` wraps the existing engine unchanged. The native energy, the
~1000-decoy ensemble, the standard deviation, and the index are all computed by the
precompiled AWSEM/LAMMPS binary (``lmp_serial_{seq_dist}_{OS}``) that writes
``tertiary_frustration.dat``; this backend just drives that subprocess via the
calculator's existing :class:`~frustrapy.analysis.lammps_runner.LammpsRunner` path.
It is the numerical reference: same binary + same coefficient files + same glue,
so the numbers match ``frustratometeR`` bit-for-bit (Phase 4 parity gate).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import FrustrationBackend

if TYPE_CHECKING:
    from ..core import Pdb
    from ..analysis.frustration_calculator import FrustrationCalculator


class LammpsBackend(FrustrationBackend):
    """AWSEM/LAMMPS subprocess backend (the reference engine)."""

    name = "lammps"

    def compute_energies(self, calculator: "FrustrationCalculator", pdb: "Pdb") -> None:
        """Run the precompiled AWSEM/LAMMPS binary, producing
        ``tertiary_frustration.dat`` in the job directory."""
        calculator._run_lammps_calculation(pdb)
