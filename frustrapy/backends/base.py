"""The pluggable frustration-engine interface.

FrustraPy computes three frustration indices (configurational, mutational,
singleresidue). Every index shares one energy model (AWSEM) and one Z-score
equation; they differ only in how the decoy ensemble is generated. The numerical
core of a run is therefore:

  1. compute the native energy and the decoy-ensemble statistics (mean, sd) for
     each probed unit (contact i-j, or site i), writing ``tertiary_frustration.dat``
     in the job directory;
  2. parse that raw output into the frustration table, deriving ``FrstIndex`` and,
     for contact modes, the ``FrstState`` class via the contact cutoffs;
  3. for configurational/mutational, compute the 5 Angstrom spatial-density /
     proportion summary.

``FrustrationBackend`` is the seam that makes step 1 -- the energy model -- pluggable.
Steps 2 and 3 are pure post-processing of step 1's output and are shared across all
backends, so they are concrete methods here; only :meth:`compute_energies` is
abstract. This mirrors the parity spine: the energy model is the swappable part
(a precompiled AWSEM/LAMMPS binary today), while parsing, the threshold
classification, and the density kernel are common glue.

A backend does not own file layout, chain selection, PDB cleaning, or the LAMMPS
input-deck preparation -- those happen in :class:`FrustrationCalculator` before the
backend is invoked, and the backend reads the prepared job directory. The contract
a backend must satisfy, given a prepared ``pdb`` (its job dir already contains the
cleaned structure and the prepared input files) plus the calculator's ``mode`` and
``seq_dist``:

  * after :meth:`compute_energies`, ``{job_dir}/tertiary_frustration.dat`` exists and
    holds the native energy and decoy mean/sd per probed unit;
  * after :meth:`process_results`, ``{job_dir}/FrustrationData/{base}.pdb_{mode}``
    exists with the canonical columns (14 for contacts incl. ``FrstIndex``/``FrstState``,
    8 for singleresidue);
  * :meth:`compute_density` returns the 5 Angstrom density results for contact modes
    (and ``None`` for singleresidue).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Optional

if TYPE_CHECKING:  # avoid import cycles / heavy imports at module load
    from ..core import Pdb
    from ..core.data_classes import FrustrationDensityResults
    from ..analysis.frustration_calculator import FrustrationCalculator


class FrustrationBackend(ABC):
    """Abstract energy backend behind the frustration engine.

    Subclasses implement :meth:`compute_energies` (the backend-specific energy
    model). :meth:`process_results` and :meth:`compute_density` are common
    post-processing and are provided here; a subclass overrides them only if it
    genuinely needs different post-processing (the reference ``LammpsBackend`` does
    not).
    """

    #: Stable identifier used for backend selection and the registry.
    name: ClassVar[str] = "base"

    #: Whether the calculator must run the AWSEM/LAMMPS input-deck preparation
    #: (``PdbCoords2Lammps.sh`` -> ``.coord`` / ``data.*`` / ``.in`` /
    #: ``fix_backbone_coeff.data`` / ``gamma.dat`` ..., plus the mode keyword swap and
    #: the ``run 10000`` -> ``run 0`` patch) before :meth:`compute_energies`. The
    #: AWSEM-based backends (``lammps``, ``native``) consume those prepared files, so
    #: they need it (the default, ``True``). A backend whose energy model does not read
    #: the LAMMPS deck (the all-atom Rosetta ``atomic`` backend, which works directly
    #: from the cleaned ``{base}.pdb`` and the equivalences file the calculator already
    #: writes) sets this ``False`` so the calculator skips that subprocess prep
    #: entirely. The cleaned PDB and the ``{base}.pdb_equivalences.txt`` map are written
    #: regardless (they are calculator-level, not LAMMPS-specific), so the shared
    #: :meth:`process_results` / :meth:`compute_density` post-processing is unaffected.
    requires_lammps_prep: ClassVar[bool] = True

    @abstractmethod
    def compute_energies(self, calculator: "FrustrationCalculator", pdb: "Pdb") -> None:
        """Run the energy model: compute the native energy and the decoy-ensemble
        statistics for the structure described by ``pdb``, writing
        ``tertiary_frustration.dat`` in ``pdb.job_dir``.

        Args:
            calculator: the owning :class:`FrustrationCalculator`, carrying the run
                configuration (``mode``, ``seq_dist``, ``debug``, ...) and the
                prepared input files in the job directory.
            pdb: the prepared :class:`~frustrapy.core.Pdb` (job dir, base name,
                scripts dir).
        """
        raise NotImplementedError

    def process_results(self, calculator: "FrustrationCalculator", pdb: "Pdb") -> None:
        """Parse ``tertiary_frustration.dat`` into the frustration table and derive
        ``FrstIndex``/``FrstState``. Shared across backends (pure post-processing)."""
        calculator._process_results(pdb)

    def compute_density(
        self, calculator: "FrustrationCalculator", pdb: "Pdb"
    ) -> Optional["FrustrationDensityResults"]:
        """Compute the 5 Angstrom spatial-density / proportion summary. Shared across
        backends (pure post-processing). Caller invokes this only for the contact
        modes (configurational/mutational)."""
        return calculator._calculate_frustration_density(pdb)
