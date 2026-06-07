from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union, Tuple
from pathlib import Path
import logging
import math
import numpy as np
import pandas as pd
from .exceptions import FrustraEvoError
import shutil
from Bio import SeqIO
from frustrapy import calculate_frustration  # Import here to avoid circular imports

logger = logging.getLogger(__name__)


def _evo_frustration_worker(job: Dict) -> None:
    """Picklable top-level worker for the FrustraEvo per-structure frustration
    precompute (T2 performance). Runs one ``calculate_frustration`` job; the
    result is the on-disk ``.done/`` table the downstream parsers read, so
    nothing is returned (the heavy ``Pdb`` object is never shipped back)."""
    calculate_frustration(**job)


@dataclass
class ResidueEquivalence:
    """Maps MSA positions to PDB residue numbers"""

    msa_pos: int
    pdb_pos: int
    residue: str
    chain: str
    structure: str


class ContactMatrix:
    """Stores frustration contacts for a single structure"""

    def __init__(self, size: int, structure_id: str):
        self.size = size
        self.structure_id = structure_id
        # Initialize with -100 as in legacy code
        self.matrix = np.full((size + 2, size + 2), -100.0)
        self.equivalences: Dict[int, ResidueEquivalence] = {}

    def add_contact(
        self, msa_pos1: int, msa_pos2: int, frustration_value: float
    ) -> None:
        """Add a contact between two MSA positions"""
        self.matrix[msa_pos1][msa_pos2] = frustration_value

    def get_contact(self, msa_pos1: int, msa_pos2: int) -> Optional[float]:
        """Get frustration value for a contact if it exists"""
        value = self.matrix[msa_pos1][msa_pos2]
        return None if value == -100.0 else value


class InformationContentCalculator:
    """Calculates sequence and frustration information content"""

    FRUSTRATION_CUTOFFS = {
        "MIN": 0.78,  # Minimally frustrated cutoff
        "MAX": -1.0,  # Maximally frustrated cutoff
    }

    REQUIRED_DIRECTORIES: Set[str] = {
        "data",
        "plots",
        "equivalences",
        "Frustration",
        "pdbs",
        "msa",
        "logs",
    }

    def __init__(
        self,
        msa_data: "MSAData",
        results_dir: Path,
        reference_pdb: str,
        pdb_dir: Optional[Path] = None,
        mode: str = "configurational",
        n_procs: Optional[int] = None,
    ):
        self.msa_data = msa_data
        self.results_dir = Path(results_dir)
        self.reference_pdb = reference_pdb
        self.mode = mode
        self.pdb_dir = pdb_dir
        # T2: number of per-structure frustration subprocesses to run
        # concurrently in :meth:`_precompute_frustration`. None => use all
        # cores; 1 => serial (byte-identical to the pre-T2 path).
        self.n_procs = n_procs

        # Identifiers (in MSA order) that passed the sequence check against
        # their PDB. Populated by _validate_sequences; the original FrustraEvo
        # drives every downstream step off this validated set, not the raw MSA
        # (e.g. for Alpha-globins it drops 1fsx-A and keeps the 20 that match).
        self.valid_ids: List[str] = []

        # Initialize paths
        self.frustration_dir = self.results_dir / "Frustration"
        # Singleresidue frustration is computed separately and drives the
        # residue equivalences (the original FrustraEvo always runs a
        # singleresidue pass for FinalAlign/Equivalences); keep it in its own
        # results tree so it does not collide with the contact-mode .done dir.
        self.frustration_sr_dir = self.results_dir / "Frustration_SR"
        self.equivalences_dir = self.results_dir / "equivalences"
        self.msa_dir = self.results_dir / "msa"
        self.data_dir = self.results_dir / "data"
        self.logs_dir = self.results_dir / "logs"
        self.pdb_dest_dir = self.results_dir / "pdbs"

    def _setup_directories(self) -> None:
        """Create required directory structure"""
        try:
            for directory in self.REQUIRED_DIRECTORIES:
                dir_path = self.results_dir / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {dir_path}")
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            raise FrustraEvoError("Directory setup failed") from e

    def _copy_required_files(self) -> None:
        """Copy input files to working directories"""
        try:
            # Copy PDB files
            if self.pdb_dir and self.pdb_dir.exists():
                for pdb_file in self.pdb_dir.glob("*.pdb"):
                    dest = self.pdb_dest_dir / pdb_file.name
                    shutil.copy2(pdb_file, dest)
                    logger.debug(f"Copied PDB file: {pdb_file.name}")

            # Copy MSA file if available
            if self.msa_data.fasta_file:
                dest = self.msa_dir / self.msa_data.fasta_file.name
                shutil.copy2(self.msa_data.fasta_file, dest)
                logger.debug(f"Copied MSA file: {self.msa_data.fasta_file.name}")

        except Exception as e:
            logger.error(f"Failed to copy files: {e}")
            raise FrustraEvoError("File copying failed") from e

    def _validate_sequences(self) -> None:
        """Validate sequences against PDB structures"""
        try:
            clean_msa = self.msa_dir / "MSA_Clean.fasta"
            error_log = self.logs_dir / "ErrorSeq.log"

            valid_sequences = []
            self.valid_ids = []
            with error_log.open("w") as out_log:
                for record in SeqIO.parse(self.msa_data.fasta_file, "fasta"):
                    seq_id = record.id
                    sequence = str(record.seq).replace("-", "")

                    # Check PDB existence
                    pdb_file = self.pdb_dest_dir / f"{seq_id}.pdb"
                    if not pdb_file.exists():
                        out_log.write(f"Missing PDB file: {seq_id}\n")
                        continue

                    # Compare sequences
                    pdb_sequence = self._get_pdb_sequence(pdb_file)
                    if sequence == pdb_sequence:
                        valid_sequences.append(record)
                        self.valid_ids.append(seq_id)
                    else:
                        out_log.write(f"Sequence mismatch for {seq_id}\n")

            # Write validated sequences
            with clean_msa.open("w") as out_msa:
                SeqIO.write(valid_sequences, out_msa, "fasta")

        except Exception as e:
            logger.error(f"Failed to validate sequences: {e}")
            raise FrustraEvoError("Sequence validation failed") from e

    # One-letter codes for the 20 standard residues, in the exact dict used by
    # the original FrustraEvo (Functions.py::obtain_seq).
    _AA_CODES = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }

    def _get_pdb_residues(self, pdb_file: Path) -> List[Tuple[int, str, str]]:
        """Extract ordered residues from a PDB, byte-faithful to the original
        FrustraEvo ``obtain_seq`` (Functions.py:150-164).

        The original takes the FIRST ``ATOM`` record of each new residue number
        (not specifically the CA), and only counts it when the alternate-location
        indicator (col 17, ``line[16]``) and insertion code (col 27,
        ``line[26]``) are both blank and the residue is one of the 20 standard
        types. The residue counter ``nn`` advances on every ATOM line, so a
        residue whose first atom carries an altLoc/insertion code is skipped
        entirely — this selection is what decides which structures pass the
        sequence check, and it differs from a CA-only scan (which mis-selected
        the family members and broke parity).

        Returns a list of ``(resnum, chain, one_letter)`` in structure order.
        """
        residues: List[Tuple[int, str, str]] = []
        nn = -100
        with pdb_file.open() as f:
            for lpdb in f:
                if lpdb[0:4] == "ATOM" and len(lpdb) > 60:
                    res_num = int(lpdb[22:26])
                    aa = lpdb[17:20]
                    if (
                        nn != res_num
                        and lpdb[16] == " "
                        and lpdb[26] == " "
                        and aa in self._AA_CODES
                    ):
                        residues.append((res_num, lpdb[21], self._AA_CODES[aa]))
                    nn = res_num
        return residues

    def _get_pdb_sequence(self, pdb_file: Path) -> str:
        """Extract the one-letter sequence from a PDB file (see
        :meth:`_get_pdb_residues` for the exact selection rule)."""
        return "".join(r[2] for r in self._get_pdb_residues(pdb_file))

    def _load_equivalences(self, structure_id: str) -> Dict[int, ResidueEquivalence]:
        """Load residue equivalences for a structure"""
        equiv_file = self.equivalences_dir / f"Equival_{structure_id}.txt"
        equivalences = {}

        try:
            with equiv_file.open() as f:
                next(f)  # Skip header
                for line in f:
                    fields = line.strip().split()
                    equiv = ResidueEquivalence(
                        msa_pos=int(fields[0]),
                        pdb_pos=int(fields[1]),
                        residue=fields[2],
                        chain=fields[3],
                        structure=fields[4],
                    )
                    equivalences[equiv.msa_pos] = equiv

            return equivalences

        except Exception as e:
            logger.error(f"Failed to load equivalences for {structure_id}: {e}")
            raise FrustraEvoError(f"Equivalence loading failed: {str(e)}")

    def _precompute_frustration(self) -> None:
        """Run every per-structure frustration calculation up front, in parallel
        (T2 performance).

        ``analyze_family`` needs two frustration tables per family member: a
        ``mode`` (configurational/mutational) contact table read by
        :meth:`_load_contact_matrices`, and a ``singleresidue`` table read by
        :meth:`_calculate_equivalences` via :meth:`_run_singleresidue`. Each
        ``calculate_frustration`` run is an independent LAMMPS single-point that
        writes to its own private ``{id}.done/`` directory, so the whole ``2*N``
        set is embarrassingly parallel. The pre-T2 code ran them as two serial
        loops (one inside ``_calculate_equivalences``, one inside
        ``_load_contact_matrices``); this runs them all in one bounded
        ``ProcessPoolExecutor`` and the downstream loops then just read the
        cached tables.

        **Parity:** output is byte-identical to the serial path — only the order
        in which the independent subprocesses run changes, and each writes to a
        separate directory, so there is no shared-state race. ``graphics=False``
        means no inner mutation pool is spawned, so the outer pool never nests a
        second ``cpu_count()`` pool. Idempotent: a job whose output table already
        exists is skipped, matching the downstream ``if not ...exists()`` guards.
        """
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        structure_ids = self.valid_ids or self.msa_data.identifiers

        # Pre-create the two results trees so the workers never race on
        # ``os.makedirs`` (``calculate_frustration`` creates ``results_dir``
        # without ``exist_ok`` — a TOCTOU race when N workers target the same,
        # not-yet-existing dir; ``Frustration_SR`` in particular is not in
        # ``REQUIRED_DIRECTORIES``).
        self.frustration_dir.mkdir(parents=True, exist_ok=True)
        self.frustration_sr_dir.mkdir(parents=True, exist_ok=True)

        # ``calculate_frustration`` strips HETATMs by saving over its input PDB
        # in place (``_process_structure``), so two jobs that share the same
        # source file (a structure's configurational + singleresidue passes)
        # would race on that write. Stage each job a private copy of the input
        # under ``_frust_inputs/<mode>/`` so no two concurrent jobs ever touch
        # the same file; the canonical ``pdbs/<id>.pdb`` is left untouched (only
        # ATOM lines are read downstream).
        stage_root = self.results_dir / "_frust_inputs"

        def _stage(sid: str, sub: str) -> Path:
            stage_dir = stage_root / sub
            stage_dir.mkdir(parents=True, exist_ok=True)
            staged = stage_dir / f"{sid}.pdb"
            shutil.copy2(self.pdb_dest_dir / f"{sid}.pdb", staged)
            return staged

        jobs: List[Dict] = []
        for sid in structure_ids:
            if not (self.pdb_dest_dir / f"{sid}.pdb").exists():
                continue
            conf_table = (
                self.frustration_dir
                / f"{sid}.done/FrustrationData/{sid}.pdb_{self.mode}"
            )
            if not conf_table.exists():
                jobs.append(
                    dict(
                        pdb_file=str(_stage(sid, self.mode)),
                        mode=self.mode,
                        results_dir=str(self.frustration_dir),
                        graphics=False,
                        # No pml/pymol scripts: the IC step reads only the
                        # numeric `.pdb_{mode}` table, and visualization writes
                        # shared-named files into the parent results dir (it
                        # globs `*_{mode}.{ext}` there), which collides across
                        # concurrent workers. Off => parity-safe + faster.
                        visualization=False,
                        debug=True,
                        n_cpus=1,
                    )
                )
            sr_table = (
                self.frustration_sr_dir
                / f"{sid}.done/FrustrationData/{sid}.pdb_singleresidue"
            )
            if not sr_table.exists():
                jobs.append(
                    dict(
                        pdb_file=str(_stage(sid, "singleresidue")),
                        mode="singleresidue",
                        results_dir=str(self.frustration_sr_dir),
                        graphics=False,
                        visualization=False,  # see note on the contact job above
                        debug=True,
                        n_cpus=1,
                    )
                )

        if not jobs:
            return

        cores = multiprocessing.cpu_count()
        requested = cores if self.n_procs is None else int(self.n_procs)
        n_workers = max(1, min(requested, cores, len(jobs)))

        if n_workers <= 1:
            for job in jobs:
                _evo_frustration_worker(job)
            return

        logger.info(
            f"Precomputing frustration for {len(jobs)} job(s) across "
            f"{n_workers} worker(s)"
        )
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(_evo_frustration_worker, jobs))

    def _load_contact_matrices(self) -> List[ContactMatrix]:
        """Load frustration contact matrices for all structures using legacy approach"""
        matrices = []
        structure_ids = self.valid_ids or self.msa_data.identifiers
        total_structures = len(structure_ids)
        logger.info(f"Loading {total_structures} contact matrices")

        for structure_id in structure_ids:
            try:
                # Debug structure processing
                logger.debug(f"Processing structure: {structure_id}")

                # 1. Load PDB-MSA position mapping exactly like legacy
                equiv_map = {}  # MSA -> PDB mapping
                rev_equiv_map = {}  # PDB -> MSA mapping
                with (
                    self.equivalences_dir / f"Equival_{structure_id}.txt"
                ).open() as f:
                    next(f)  # Skip header
                    for line in f:
                        fields = line.strip().split("\t")
                        # Reference-gap-stripped columns where this structure has
                        # a gap are written with PDB_pos == "N/A" (mirroring the
                        # original Equivalences step); they map to no contact.
                        if fields[1] == "N/A":
                            continue
                        msa_pos, pdb_pos = int(fields[0]), int(fields[1])
                        equiv_map[msa_pos] = pdb_pos
                        rev_equiv_map[pdb_pos] = msa_pos

                logger.debug(f"Loaded {len(equiv_map)} position mappings")

                # 2. Initialize matrix with exact legacy dimensions
                matrix = ContactMatrix(self.msa_data.length, structure_id)
                matrix.matrix.fill(-100.0)  # Legacy uses -100 as null value

                # 3. Calculate frustration using FrustraPy
                pdb_file = self.pdb_dest_dir / f"{structure_id}.pdb"
                # 4. Load frustration data using legacy column indices
                frust_file = (
                    self.frustration_dir
                    / f"{structure_id}.done/FrustrationData/{structure_id}.pdb_{self.mode}"
                )
                # graphics=False: the IC calculation only reads the
                # `.pdb_{mode}` frustration table, not the per-structure plots.
                # Generating them for every family member is wasteful and pulls
                # in the optional `kaleido` dependency (plot_5andens write_image).
                # T2: `_precompute_frustration` normally fills this table in
                # parallel beforehand; only run here if it is still missing
                # (e.g. precompute was skipped). The returned objects are unused
                # downstream — only `frust_file` is read.
                if not frust_file.exists():
                    calculate_frustration(
                        pdb_file=str(pdb_file),
                        mode=self.mode,
                        results_dir=str(self.frustration_dir),
                        graphics=False,
                        debug=True,
                    )

                contact_count = 0
                with frust_file.open() as f:
                    next(f)  # Skip header
                    for line in f:
                        try:
                            fields = line.strip().split()
                            pdb_pos1, pdb_pos2 = int(fields[0]), int(fields[1])
                            # FrstIndex is column index 11 in the 14-column
                            # configurational/mutational table (Res1 Res2
                            # ChainRes1 ChainRes2 DensityRes1 DensityRes2 AA1 AA2
                            # NativeEnergy DecoyEnergy SDEnergy FrstIndex Welltype
                            # FrstState). Index 9 is DecoyEnergy — reading it here
                            # mis-classified every contact as highly frustrated.
                            frust_value = float(fields[11])  # FrstIndex column

                            # Only process if both positions are mapped
                            if pdb_pos1 in rev_equiv_map and pdb_pos2 in rev_equiv_map:
                                msa_pos1 = rev_equiv_map[pdb_pos1]
                                msa_pos2 = rev_equiv_map[pdb_pos2]

                                # Add contact both ways like legacy
                                matrix.add_contact(msa_pos1, msa_pos2, frust_value)
                                matrix.add_contact(msa_pos2, msa_pos1, frust_value)
                                contact_count += 1

                                logger.debug(
                                    f"Added symmetric contact {msa_pos1}-{msa_pos2} "
                                    f"(PDB: {pdb_pos1}-{pdb_pos2}) = {frust_value}"
                                )

                        except (ValueError, IndexError) as e:
                            logger.warning(
                                f"Invalid line in {structure_id} frustration file: "
                                f"{line.strip()} - {str(e)}"
                            )
                            continue

                # Store equivalences for later use
                matrix.equivalences = {
                    msa_pos: ResidueEquivalence(
                        msa_pos=msa_pos,
                        pdb_pos=pdb_pos,
                        residue=self._get_residue(structure_id, pdb_pos),
                        chain="A",  # Legacy assumes chain A
                        structure=structure_id,
                    )
                    for msa_pos, pdb_pos in equiv_map.items()
                }

                logger.info(
                    f"Processed {structure_id}: {contact_count} contacts, "
                    f"{len(matrix.equivalences)} positions"
                )
                matrices.append(matrix)

            except Exception as e:
                logger.error(
                    f"Failed to process {structure_id}: {str(e)}", exc_info=True
                )
                continue

        if not matrices:
            raise FrustraEvoError("No valid contact matrices could be loaded")

        logger.info(f"Successfully loaded {len(matrices)} contact matrices")
        return matrices

    def _get_residue(self, structure_id: str, pdb_pos: int) -> str:
        """Get residue type from PDB file at given position"""
        pdb_file = self.pdb_dest_dir / f"{structure_id}.pdb"
        with pdb_file.open() as f:
            for line in f:
                if line.startswith("ATOM") and line[22:26].strip() == str(pdb_pos):
                    return line[17:20].strip()
        return "UNK"

    def _calculate_frustration_state(self, value: float) -> str:
        """Determine frustration state based on cutoffs"""
        if value >= self.FRUSTRATION_CUTOFFS["MIN"]:
            return "MIN"
        elif value <= self.FRUSTRATION_CUTOFFS["MAX"]:
            return "MAX"
        return "NEU"

    # Expected (background) frustration-state probabilities used by the ORIGINAL
    # FrustraEvo IC computation (Scripts/IC_Conts_Conf.py::information_content):
    # pMIN=0.4, pMAX=0.1, pNEU=0.5. h_background is COMPUTED with math.log(p, 2)
    # (NOT a truncated literal, NOT math.log2) so the float is bit-for-bit identical
    # to the original: -(0.4*log2(0.4) + 0.1*log2(0.1) + 0.5*log2(0.5))
    #               = 1.360964047443681
    _H_BACKGROUND = -(
        0.4 * math.log(0.4, 2)
        + 0.1 * math.log(0.1, 2)
        + 0.5 * math.log(0.5, 2)
    )

    @staticmethod
    def _h_term(p: float):
        """Shannon term H = -(p*log2(p)) for p>0, else int 0.

        Mirrors the original Hmin/Hmax/Hneu EXACTLY: uses math.log(p, 2) and
        returns the int 0 (not 0.0) when p == 0 — so str() prints "0" for an
        absent state and "-0.0" for a fully conserved state (p == 1.0 gives
        -(1.0*0.0) = -0.0). These exact reprs are required for byte parity.
        """
        if p > 0:
            return -(p * math.log(p, 2))
        return 0

    def _calculate_contact_stats(self, values: List[float]) -> Dict:
        """Per-contact IC stats, byte-identical to the original FrustraEvo
        IC_Conts_Conf.py. NO probability rounding, NO entropy/IC rounding, NO
        max(0, ..) clamp (IC may be negative) — the original writes str(float)
        verbatim and its R annotation step pastes the strings unchanged."""
        if len(values) <= 1:
            return {}

        # Count states
        states = [self._calculate_frustration_state(v) for v in values]
        num = {"MIN": 0, "NEU": 0, "MAX": 0}
        for state in states:
            num[state] += 1

        conts = len(values)
        p_neu = float(num["NEU"]) / float(conts)
        p_min = float(num["MIN"]) / float(conts)
        p_max = float(num["MAX"]) / float(conts)

        h_neu = self._h_term(p_neu)
        h_min = self._h_term(p_min)
        h_max = self._h_term(p_max)
        # Original sum order: HMIN + HMAX + HNEU
        h_total = h_min + h_max + h_neu

        ic_total = self._H_BACKGROUND - h_total  # no clamp, no round
        ic_min = ic_total * p_min
        ic_max = ic_total * p_max
        ic_neu = ic_total * p_neu

        # Conserved state — original tie order MIN > NEU > MAX
        mx = max(num["MIN"], num["NEU"], num["MAX"])
        if mx == num["MIN"]:
            conserved_state = "MIN"
        elif mx == num["NEU"]:
            conserved_state = "NEU"
        else:
            conserved_state = "MAX"

        return {
            "counts": num,
            "probabilities": {"NEU": p_neu, "MIN": p_min, "MAX": p_max},
            "entropy_terms": {"NEU": h_neu, "MIN": h_min, "MAX": h_max},
            "h_total": h_total,
            "ic_terms": {"NEU": ic_neu, "MIN": ic_min, "MAX": ic_max},
            "ic_total": ic_total,
            "conserved_state": conserved_state,
        }

    def _calculate_total_ic(self, h_total: float, sample_size: int) -> float:
        """Calculate total information content with background correction"""
        # Legacy uses fixed background entropy value
        h_background = 1.36096404744368

        # Small sample correction (not used in legacy)
        # correction = (3 - 1) / (2 * np.log(2) * sample_size)

        return round(max(0.0, h_background - h_total), 15)

    def calculate(self) -> pd.DataFrame:
        """Calculate information content for all contacts"""
        try:
            # Load matrices
            matrices = self._load_contact_matrices()
            if not matrices:
                raise FrustraEvoError("No valid contact matrices found")

            total = len(matrices)

            # Reference equivalences map the shared coordinate (MSA_pos, 1..N
            # over reference-non-gap columns) to the reference's real ATOM
            # residue number, residue and chain. Loaded ONCE.
            ref_equiv = self._load_equivalences(self.reference_pdb)
            ref_n = len(ref_equiv)

            # Process all reference contact pairs, in MSA-column order. The
            # original FrustraEvo iterates over the reference's positions but
            # counts a contact across ALL structures that have it (the
            # reference need not be one of them); the contact is emitted with
            # the reference's residue/number labels.
            results = []
            for i in range(1, ref_n + 1):
                for j in range(i + 1, ref_n + 1):
                    # Collect values across all structures that share the
                    # contact (use `is not None` so a legitimate 0.0 frustration
                    # value is not dropped by a truthiness test).
                    values = []
                    for matrix in matrices:
                        if (value := matrix.get_contact(i, j)) is not None:
                            values.append(value)

                    # A single occurrence carries no information content.
                    if len(values) > 1:
                        stats = self._calculate_contact_stats(values)

                        ref_res1 = ref_equiv[i]
                        ref_res2 = ref_equiv[j]

                        # Coordinate convention (matches the original's
                        # add_ref_Cmaps annotation, Functions.py:799-830):
                        #   Res1/Res2       = shared MSA columns i, j
                        #   NumRes*_Ref     = reference REAL ATOM residue numbers
                        #   AA*             = reference residues
                        results.append(
                            {
                                "Res1": i,
                                "Res2": j,
                                "AA1": ref_res1.residue,
                                "AA2": ref_res2.residue,
                                "NumRes1_Ref": ref_res1.pdb_pos,
                                "Chain1_Ref": ref_res1.chain,
                                "NumRes2_Ref": ref_res2.pdb_pos,
                                "Chain2_Ref": ref_res2.chain,
                                "Prot_Ref": self.reference_pdb,
                                "NoContacts": len(values),
                                "FreqConts": len(values) / total,
                                "pNEU": stats["probabilities"]["NEU"],
                                "pMIN": stats["probabilities"]["MIN"],
                                "pMAX": stats["probabilities"]["MAX"],
                                "HNEU": stats["entropy_terms"]["NEU"],
                                "HMIN": stats["entropy_terms"]["MIN"],
                                "HMAX": stats["entropy_terms"]["MAX"],
                                "Htotal": stats["h_total"],
                                "ICNEU": stats["ic_terms"]["NEU"],
                                "ICMIN": stats["ic_terms"]["MIN"],
                                "ICMAX": stats["ic_terms"]["MAX"],
                                "ICtotal": stats["ic_total"],
                                "FstConserved": stats["conserved_state"],
                            }
                        )

            # Ensure columns are in correct order
            column_order = [
                "Res1",
                "Res2",
                "AA1",
                "AA2",
                "NumRes1_Ref",
                "Chain1_Ref",
                "NumRes2_Ref",
                "Chain2_Ref",
                "Prot_Ref",
                "NoContacts",
                "FreqConts",
                "pNEU",
                "pMIN",
                "pMAX",
                "HNEU",
                "HMIN",
                "HMAX",
                "Htotal",
                "ICNEU",
                "ICMIN",
                "ICMAX",
                "ICtotal",
                "FstConserved",
            ]

            output_file = (
                self.results_dir
                / f"IC_{self.mode.capitalize()}_{self.reference_pdb}.csv"
            )
            # Write byte-for-byte like the original: every value is str()'d and
            # tab-joined. This preserves the original's exact int-vs-float reprs
            # (e.g. "0" for an absent state, "-0.0" for a fully conserved one,
            # "0.0" for a float zero) that pandas.to_csv would homogenize away.
            with output_file.open("w") as out:
                out.write("\t".join(column_order) + "\n")
                for row in results:
                    out.write(
                        "\t".join(str(row[col]) for col in column_order) + "\n"
                    )

            # Return a DataFrame for the public API / summary counts (the file
            # on disk is the authoritative, parity-gated artifact).
            df = pd.DataFrame(results, columns=column_order)

            return df

        except Exception as e:
            logger.error(f"Calculation failed: {str(e)}")
            raise FrustraEvoError(f"Information content calculation failed: {str(e)}")

    # Background frustration-state entropy used by the ORIGINAL FrustraEvo
    # single-residue IC step (``Scripts/Logo.R``). Logo.R is R code, so its
    # ``log2`` is C ``log2`` — mirror it with ``math.log2`` (NOT ``math.log(p, 2)``,
    # which the *contact* IC port uses to match its Python original). On the same
    # libm these agree, but keep the provenance explicit:
    #     -(0.4*log2(0.4) + 0.1*log2(0.1) + 0.5*log2(0.5)) = 1.360964047443681
    _H_BACKGROUND_SR = -(
        0.4 * math.log2(0.4)
        + 0.1 * math.log2(0.1)
        + 0.5 * math.log2(0.5)
    )

    @staticmethod
    def _r_cat_format(x: float) -> str:
        """Format a double the way the original FrustraEvo's R single-residue
        step prints it. ``Logo.R`` emits every IC value with R's default
        ``cat`` (``getOption("digits") == 7``), which for these magnitudes is
        exactly C ``%.7g``; R also prints a (possibly negative) zero as ``"0"``.
        Verified to round-trip every value of the original ``IC_SingleRes`` files
        byte-for-byte."""
        if x == 0:  # True for both +0.0 and -0.0; R prints "0"
            return "0"
        return "%.7g" % x

    @staticmethod
    def _singleres_state(frst_index: float) -> str:
        """Classify a single-residue ``FrstIndex`` into a frustration state with
        the ORIGINAL FrustraEvo single-residue cutoffs (``Functions.py:604-609``):
        ``> 0.55`` -> minimally (``MIN``), ``< -1`` -> maximally (``MAX``), else
        neutral (``NEU``). These are the single-residue cutoffs (0.55 / -1), NOT
        the contact cutoffs (0.78 / -1) used by the IC_Conf/IC_Mut path."""
        if frst_index > 0.55:
            return "MIN"
        if frst_index < -1:
            return "MAX"
        return "NEU"

    def _write_singleres_ic(self) -> Path:
        """Write the per-residue frustration information-content table
        (``IC_SingleRes``), byte-faithful to the original FrustraEvo.

        Port of ``Scripts/Logo.R`` + the ``add_ref`` annotation
        (``Functions.py:664-692``). For every reference-gap-stripped MSA column
        (the shared 1..N coordinate, same as IC_Conf's ``Res``) it counts, across
        all family structures that have a residue there, how many are minimally /
        neutrally / maximally frustrated **at the single-residue level**, then
        computes the corrected frustration information content:

            correction = (3 - 1) / (2 * ln(2) * total)      # small-sample
            shannon    = -(p_min*log2(p_min) + p_neu*log2(p_neu) + p_max*log2(p_max))
            IC_total   = H_background - shannon - correction
            IC_state   = p_state * IC_total

        with ``H_background`` computed via :data:`_H_BACKGROUND_SR`. There is NO
        clamp and NO rounding — IC_total may be negative (states more mixed than
        the background). The conserved-state tie order is the original's
        (``Functions.py:69-76``): ``MAX`` if it strictly dominates, else ``NEU``
        if it strictly beats ``MIN``, else ``MIN``.

        The per-structure, per-position state is rederived from each member's
        single-residue frustration table (the same tables the equivalences were
        built from): the equivalence file gives ``MSA_pos -> PDB_pos`` (``N/A``
        for a structure gap) and the single-residue table gives that residue's
        ``FrstIndex``. Values are written with :meth:`_r_cat_format` so the file
        is byte-identical to the R-produced original.

        Returns:
            Path to the written ``IC_SingleRes_<reference_pdb>.csv`` file.
        """
        structure_ids = self.valid_ids or self.msa_data.identifiers

        # counts[msa_pos] = {"MIN": n, "NEU": n, "MAX": n} across all structures
        counts: Dict[int, Dict[str, int]] = {}
        for structure_id in structure_ids:
            # FrstIndex by PDB residue number from this member's single-residue
            # frustration table (Res in col 0, FrstIndex in col 7).
            sr_file = (
                self.frustration_sr_dir
                / f"{structure_id}.done/FrustrationData/{structure_id}.pdb_singleresidue"
            )
            if not sr_file.exists():
                logger.warning(
                    f"Missing single-residue table for {structure_id}; "
                    f"skipping in IC_SingleRes"
                )
                continue
            frst_by_res: Dict[str, float] = {}
            with sr_file.open() as f:
                next(f)  # header
                for line in f:
                    sp = line.split()
                    if len(sp) > 7:
                        frst_by_res[sp[0]] = float(sp[7])

            equiv_file = self.equivalences_dir / f"Equival_{structure_id}.txt"
            with equiv_file.open() as f:
                next(f)  # header
                for line in f:
                    fields = line.rstrip("\n").split("\t")
                    msa_pos, pdb_pos = fields[0], fields[1]
                    if pdb_pos == "N/A":
                        continue  # structure has a gap at this column
                    frst_index = frst_by_res.get(pdb_pos)
                    if frst_index is None:
                        continue
                    state = self._singleres_state(frst_index)
                    bucket = counts.setdefault(
                        int(msa_pos), {"MIN": 0, "NEU": 0, "MAX": 0}
                    )
                    bucket[state] += 1

        # Reference residue/number per shared column (add_ref's vectorAA/num).
        ref_equiv = self._load_equivalences(self.reference_pdb)
        ref_n = len(ref_equiv)

        output_file = (
            self.results_dir / f"IC_SingleRes_{self.reference_pdb}.csv"
        )
        header = (
            "Res\tAA_Ref\tNum_Ref\tProt_Ref\t%Min\t%Neu\t%Max\t"
            "CantMin\tCantNeu\tCantMax\tICMin\tICNeu\tICMax\tICTot\tFrustIC"
        )
        with output_file.open("w") as out:
            out.write(header + "\n")
            for pos in range(1, ref_n + 1):
                bucket = counts.get(pos, {"MIN": 0, "NEU": 0, "MAX": 0})
                n_min, n_neu, n_max = bucket["MIN"], bucket["NEU"], bucket["MAX"]
                total = n_min + n_neu + n_max
                if total == 0:
                    # Cannot happen for a valid reference (it is non-gap at every
                    # retained column, so it always contributes >= 1); guard
                    # against a 0/0 NaN rather than emit one.
                    logger.warning(
                        f"No single-residue states at column {pos}; skipping"
                    )
                    continue

                # Logo.R math (Scripts/Logo.R:39-67).
                correction = (3 - 1) / (2 * math.log(2) * total)
                p_min = n_min / total
                p_neu = n_neu / total
                p_max = n_max / total

                def _sh(p: float) -> float:
                    return p * math.log2(p) if p > 0 else 0

                # Sum order matches Logo.R: min + neu + max.
                shannon = -(_sh(p_min) + _sh(p_neu) + _sh(p_max))
                ic_total = self._H_BACKGROUND_SR - shannon - correction
                ic_min = p_min * ic_total
                ic_neu = p_neu * ic_total
                ic_max = p_max * ic_total

                # Conserved state, original tie order (Functions.py:69-76).
                if n_max > n_neu:
                    estado = "MAX" if n_max > n_min else "MIN"
                else:
                    estado = "NEU" if n_neu > n_min else "MIN"

                ref_res = ref_equiv[pos]
                out.write(
                    "\t".join(
                        [
                            str(pos),
                            ref_res.residue,
                            str(ref_res.pdb_pos),
                            self.reference_pdb,
                            self._r_cat_format(p_min),
                            self._r_cat_format(p_neu),
                            self._r_cat_format(p_max),
                            str(n_min),
                            str(n_neu),
                            str(n_max),
                            self._r_cat_format(ic_min),
                            self._r_cat_format(ic_neu),
                            self._r_cat_format(ic_max),
                            self._r_cat_format(ic_total),
                            estado,
                        ]
                    )
                    + "\n"
                )

        logger.debug(f"Wrote single-residue IC table: {output_file}")
        return output_file

    def _write_sequence_ic(self) -> Path:
        """Write the per-column sequence Shannon entropy table (SeqIC).

        Faithful port of the original FrustraEvo ``Scripts/Seq_IC.py``: the
        original computes Shannon entropy for every column of
        ``OutPutFiles/MSA_<JobId>.fasta``, which is the **reference-gap-stripped**
        alignment (every column where the reference sequence has a gap is
        dropped, the rest renumbered ``1..N``). Our ``MSA_Final.fasta`` keeps
        those reference-gap columns (and lists the reference first), so we strip
        them here to land on the same column set.

        Entropy is order-independent per column, so the differing sequence order
        between the two tools does not matter. Values are written with bare
        ``str()`` on the NumPy ``float64`` result exactly as the original does,
        preserving the ``-0.0`` / full-precision reprs byte-for-byte.

        Returns:
            Path to the written ``SeqIC_<reference_pdb>.tab`` file.
        """

        def shannon_entropy(column):
            # Mirror Seq_IC.py:shannon_entropy exactly (NumPy unique + log2).
            _, counts = np.unique(list(column), return_counts=True)
            probabilities = counts / len(column)
            return -np.sum(probabilities * np.log2(probabilities))

        final_msa = self.msa_dir / "MSA_Final.fasta"
        if not final_msa.exists():
            raise FileNotFoundError(f"Final MSA file not found: {final_msa}")

        records = list(SeqIO.parse(final_msa, "fasta"))
        if not records:
            raise FrustraEvoError("MSA_Final.fasta is empty; cannot compute SeqIC")

        ids = [r.id for r in records]
        if self.reference_pdb not in ids:
            raise ValueError(
                f"Reference sequence {self.reference_pdb} not found in MSA_Final"
            )
        sequences = [str(r.seq) for r in records]
        reference = sequences[ids.index(self.reference_pdb)]

        # Reference-gap strip: keep only columns where the reference is not a gap.
        kept_columns = [j for j in range(len(reference)) if reference[j] != "-"]
        alignment = np.array([list(s) for s in sequences])

        output_file = self.results_dir / f"SeqIC_{self.reference_pdb}.tab"
        with output_file.open("w") as out:
            out.write("Position\tEntropy\n")
            for position, col_index in enumerate(kept_columns):
                entropy = shannon_entropy(alignment[:, col_index])
                out.write(f"{position + 1}\t{entropy}\n")

        logger.debug(f"Wrote sequence IC table: {output_file}")
        return output_file

    def _calculate_entropy_term(self, probability: float) -> float:
        """Calculate Shannon entropy term"""
        if probability <= 0:
            return 0.0
        return -probability * np.log2(probability)

    def _prepare_reference_alignment(self) -> None:
        """
        Prepare reference sequence alignment.

        Raises:
            FrustraEvoError: If reference alignment preparation fails
            ValueError: If reference PDB is not specified or found
        """
        try:
            logger.info("Preparing reference alignment")

            if not self.reference_pdb:
                raise ValueError("Reference PDB must be specified")

            clean_msa = self.msa_dir / "MSA_Clean.fasta"
            ref_msa = self.msa_dir / "MSA_Clean_Ref.fasta"

            if not clean_msa.exists():
                raise FileNotFoundError(f"Clean MSA file not found: {clean_msa}")

            # Extract reference sequence
            ref_found = False
            records = []

            # First pass to find reference sequence
            for record in SeqIO.parse(clean_msa, "fasta"):
                if record.id == self.reference_pdb:
                    records.insert(0, record)  # Add reference sequence first
                    ref_found = True
                else:
                    records.append(record)

            if not ref_found:
                raise ValueError(
                    f"Reference sequence {self.reference_pdb} not found in MSA"
                )

            # Write aligned sequences
            with ref_msa.open("w") as out:
                SeqIO.write(records, out, "fasta")

            logger.debug(f"Created reference alignment with {len(records)} sequences")

        except Exception as e:
            logger.error(f"Failed to prepare reference alignment: {str(e)}")
            raise FrustraEvoError("Reference alignment preparation failed") from e

    def _create_final_alignment(self) -> None:
        """
        Create final alignment with position mapping.

        Raises:
            FrustraEvoError: If final alignment creation fails
        """
        try:
            logger.info("Creating final alignment")

            ref_msa = self.msa_dir / "MSA_Clean_Ref.fasta"
            final_msa = self.msa_dir / "MSA_Final.fasta"
            positions_file = self.data_dir / "Positions.txt"

            if not ref_msa.exists():
                raise FileNotFoundError(f"Reference MSA file not found: {ref_msa}")

            # Process alignment and create position mapping
            with final_msa.open("w") as out_msa, positions_file.open("w") as out_pos:
                for record in SeqIO.parse(ref_msa, "fasta"):
                    seq_id = record.id
                    sequence = str(record.seq)

                    # Write sequence to final MSA
                    out_msa.write(f">{seq_id}\n{sequence}\n")

                    # Create position mapping for reference sequence
                    if seq_id == self.reference_pdb:
                        out_pos.write(f">{seq_id}\n")
                        positions = []
                        pos = 1

                        for aa in sequence:
                            if aa != "-":
                                positions.append(str(pos))
                                pos += 1

                        out_pos.write(" ".join(positions) + "\n")

            logger.debug(f"Created final alignment and position mapping")

        except Exception as e:
            logger.error(f"Failed to create final alignment: {str(e)}")
            raise FrustraEvoError("Final alignment creation failed") from e

    def _prepare_logo_data(self) -> None:
        """
        Prepare sequence data for logo generation.

        Raises:
            FrustraEvoError: If logo data preparation fails
        """
        try:
            logger.info("Preparing logo data")

            final_msa = self.msa_dir / "MSA_Final.fasta"
            logo_data = self.data_dir / "Logo.fasta"

            if not final_msa.exists():
                raise FileNotFoundError(f"Final MSA file not found: {final_msa}")

            # Extract sequences without headers
            with logo_data.open("w") as out:
                for record in SeqIO.parse(final_msa, "fasta"):
                    out.write(f"{record.seq}\n")

            logger.debug(f"Created logo data file: {logo_data}")

        except Exception as e:
            logger.error(f"Failed to prepare logo data: {str(e)}")
            raise FrustraEvoError("Logo data preparation failed") from e

    def _validate_logo_data(self) -> None:
        """
        Validate logo data and combine equivalence files.

        Raises:
            FrustraEvoError: If logo data validation fails
        """
        try:
            logger.info("Validating logo data")

            equiv_dir = self.results_dir / "equivalences"
            combined_file = equiv_dir / "AllEquivalences.txt"

            if not equiv_dir.exists() or not any(equiv_dir.iterdir()):
                raise FileNotFoundError(f"No equivalence files found in: {equiv_dir}")

            # Combine and validate equivalence files
            valid_entries = []
            header_written = False

            for equiv_file in equiv_dir.glob("Equival_*.txt"):
                with equiv_file.open() as f:
                    header = next(f, None)  # Skip header line
                    if not header_written and header:
                        valid_entries.append(header)
                        header_written = True

                    for line in f:
                        fields = line.strip().split("\t")
                        if len(fields) > 4:  # Valid line with enough fields
                            try:
                                pos = int(fields[0])
                                if pos <= 0:
                                    logger.warning(
                                        f"Invalid position in {equiv_file.name}: {pos}"
                                    )
                                    continue
                                valid_entries.append(line)
                            except ValueError:
                                logger.warning(
                                    f"Invalid data format in {equiv_file.name}: {line.strip()}"
                                )
                                continue

            # Write validated entries
            with combined_file.open("w") as out:
                out.writelines(valid_entries)

            logger.debug(
                f"Combined {len(valid_entries)-1} valid entries into {combined_file}"
            )

        except Exception as e:
            logger.error(f"Failed to validate logo data: {str(e)}")
            raise FrustraEvoError("Logo data validation failed") from e

    def _calculate_conservation(self) -> Dict[int, float]:
        """
        Calculate position-specific conservation scores.

        Returns:
            Dict[int, float]: Position-specific conservation scores

        Raises:
            FrustraEvoError: If conservation calculation fails
        """
        try:
            logger.info("Calculating conservation scores")

            conservation_scores = {}
            matrix = self.msa_data.conservation_matrix

            # Calculate Shannon entropy for each position
            for pos in range(matrix.shape[1]):
                freqs = matrix[:, pos]
                entropy = 0.0

                # Calculate position-specific entropy
                for freq in freqs[freqs > 0]:  # Only consider non-zero frequencies
                    entropy -= freq * np.log2(freq)

                # Convert entropy to conservation score
                max_entropy = np.log2(20)  # Maximum possible entropy (20 amino acids)
                conservation_scores[pos + 1] = 1 - (entropy / max_entropy)

            logger.debug(
                f"Calculated conservation scores for {len(conservation_scores)} positions"
            )
            return conservation_scores

        except Exception as e:
            logger.error(f"Failed to calculate conservation scores: {str(e)}")
            raise FrustraEvoError("Conservation calculation failed") from e

    def _get_sequence_diff(self, seq1: str, seq2: str) -> str:
        """
        Generate a visual difference between two sequences.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            str: Visual difference markers
        """
        return "".join(
            " " if (i >= len(seq1) or i >= len(seq2) or seq1[i] == seq2[i]) else "^"
            for i in range(max(len(seq1), len(seq2)))
        )

    def _calculate_equivalences(self) -> None:
        """
        Calculate and save residue equivalences for all structures.

        Raises:
            FrustraEvoError: If equivalence calculation fails
        """
        try:
            logger.info("Calculating residue equivalences")

            # Ensure equivalences directory exists
            equiv_dir = self.results_dir / "equivalences"
            equiv_dir.mkdir(exist_ok=True)

            # Reference aligned sequence defines the shared coordinate: the
            # original FrustraEvo strips every alignment column where the
            # reference has a gap and renumbers the survivors 1..N (long.txt).
            try:
                ref_idx = self.msa_data.identifiers.index(self.reference_pdb)
            except ValueError:
                raise FrustraEvoError(
                    f"Reference {self.reference_pdb} not found in the MSA"
                )
            ref_aln_seq = self.msa_data.sequences[ref_idx]

            # Process each validated sequence
            for seq_id in (self.valid_ids or self.msa_data.identifiers):
                try:
                    # Get PDB file path
                    pdb_file = self.pdb_dest_dir / f"{seq_id}.pdb"
                    if not pdb_file.exists():
                        logger.warning(f"PDB file not found for {seq_id}")
                        continue

                    # Get sequence from MSA
                    msa_idx = self.msa_data.identifiers.index(seq_id)
                    msa_seq = self.msa_data.sequences[msa_idx]

                    # Singleresidue frustration provides the residue numbering
                    # and per-residue index used to build the equivalences (the
                    # original derives PDB_pos from the frustration output, not
                    # the raw PDB — they differ for altLoc/insertion residues).
                    sr_lines = self._run_singleresidue(seq_id, pdb_file)

                    # Calculate equivalences
                    equiv_file = equiv_dir / f"Equival_{seq_id}.txt"
                    self._save_structure_equivalences(
                        structure_id=seq_id,
                        msa_seq=msa_seq,
                        ref_aln_seq=ref_aln_seq,
                        sr_lines=sr_lines,
                        output_file=equiv_file,
                    )
                    logger.debug(f"Saved equivalences for {seq_id}")

                except Exception as e:
                    logger.error(
                        f"Failed to process equivalences for {seq_id}: {str(e)}"
                    )
                    continue

            # Verify equivalences were created
            equiv_files = list(equiv_dir.glob("Equival_*.txt"))
            if not equiv_files:
                raise FrustraEvoError("No equivalence files were created")

            logger.info(f"Created {len(equiv_files)} equivalence files")

        except Exception as e:
            logger.error(f"Failed to calculate equivalences: {str(e)}")
            raise FrustraEvoError("Equivalence calculation failed") from e

    def _run_singleresidue(self, structure_id: str, pdb_file: Path) -> List[str]:
        """Run singleresidue frustration for a structure and return the lines of
        its ``.pdb_singleresidue`` table (line 0 is the header). This is the
        residue list the original FrustraEvo's FinalAlign/Equivalences walk over.
        """
        sr_file = (
            self.frustration_sr_dir
            / f"{structure_id}.done/FrustrationData/{structure_id}.pdb_singleresidue"
        )
        if not sr_file.exists():
            calculate_frustration(
                pdb_file=str(pdb_file),
                mode="singleresidue",
                results_dir=str(self.frustration_sr_dir),
                graphics=False,
                debug=True,
            )
        with sr_file.open() as f:
            return f.readlines()

    @staticmethod
    def _build_positions(ref_aln_seq: str, msa_seq: str, sr_lines: List[str]) -> List[str]:
        """Port of the original FrustraEvo ``FinalAlign`` per-structure walk
        (Functions.py:500-552). Strips every alignment column where the
        *reference* has a gap and, for the survivors, emits this structure's
        singleresidue residue number (``splitres[0]``), ``'G'`` where the
        structure has a gap, or ``'Z'`` for an unknown residue. The counter
        ``q`` advances over the structure's residues (including in reference-gap
        columns) to index the singleresidue table."""
        vector = [0 if ch in ("-", "Z") else 1 for ch in ref_aln_seq]
        positions: List[str] = []
        q = 0
        for j, cj in enumerate(msa_seq):
            if j >= len(vector):
                break
            if vector[j] == 0:  # reference gap column -> stripped
                if cj != "-":
                    q += 1
            else:  # reference-non-gap column -> emit one token
                if cj == "Z" or cj == "X":
                    q += 1
                    positions.append("Z")
                elif cj == "-":
                    positions.append("G")
                else:
                    q += 1
                    sp = sr_lines[q].split() if q < len(sr_lines) else []
                    positions.append(sp[0] if sp else "G")
        return positions

    def _save_structure_equivalences(
        self,
        structure_id: str,
        msa_seq: str,
        ref_aln_seq: str,
        sr_lines: List[str],
        output_file: Path,
    ) -> None:
        """
        Save residue equivalences for a single structure in the shared,
        reference-gap-stripped coordinate, byte-faithful to the original
        FrustraEvo ``FinalAlign`` + ``Equivalences`` (Functions.py:481-610).

        Column 0 (``MSA_pos``) numbers 1..N over the alignment columns where the
        *reference* is not a gap. Column 1 (``PDB_pos``) is this structure's
        residue number taken from the **singleresidue frustration output**
        (resynced by matching residue numbers, which is what lets altLoc /
        insertion residues — e.g. 2b7h-A's D74 — line up), or ``N/A`` where the
        structure has a gap. The downstream IC reads only columns 0–3, so the
        per-row frustration value/state the original also stores are folded into
        residue/chain here.
        """
        try:
            positions = self._build_positions(ref_aln_seq, msa_seq, sr_lines)

            # Equivalences walk (Functions.py:562-610): re-read the singleresidue
            # table line by line, resyncing on the residue number so the row for
            # MSA column `ter` carries the matching frustration residue.
            rows: List[Tuple[str, str, str, str]] = []  # (msa, pdb, aa, chain)
            cur = sr_lines[0] if sr_lines else ""
            splitres = cur.split()
            sr_ptr = 0
            ter = 0
            n = len(positions)
            while ter < n:
                ter += 1
                token = positions[ter - 1]
                chain = splitres[1] if len(splitres) > 1 else "A"
                if token in ("G", "Z"):
                    rows.append((str(ter), "N/A", "N/A", chain))
                    continue
                sr_ptr += 1
                cur = sr_lines[sr_ptr].rstrip("\n") if sr_ptr < len(sr_lines) else ""
                splitres = cur.split()
                if cur == "":
                    continue
                if len(splitres) < 7 and (len(splitres) < 5 or splitres[4] != "Missing"):
                    break
                if splitres and int(splitres[0]) < int(token):
                    while True:
                        sr_ptr += 1
                        cur = (
                            sr_lines[sr_ptr].rstrip("\n")
                            if sr_ptr < len(sr_lines)
                            else ""
                        )
                        splitres = cur.split()
                        if (splitres and splitres[0] == token) or len(cur) < 1:
                            break
                if len(splitres) == 6:
                    ter -= 1
                if splitres and splitres[0] == token and len(splitres) > 7:
                    rows.append((str(ter), splitres[0], splitres[3], splitres[1]))

            # Write equivalences in FrustraPy's own 5-column layout (the
            # downstream loaders read MSA_pos, PDB_pos, Residue, Chain).
            with output_file.open("w") as f:
                f.write("MSA_pos\tPDB_pos\tResidue\tChain\tStructure\n")
                for msa_pos, pdb_pos, aa, chain in rows:
                    f.write(
                        f"{msa_pos}\t{pdb_pos}\t{aa}\t{chain}\t{structure_id}\n"
                    )

        except Exception as e:
            logger.error(
                f"Failed to save equivalences for {structure_id}: {str(e)}"
            )
            raise
