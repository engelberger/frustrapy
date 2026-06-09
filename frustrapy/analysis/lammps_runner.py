import os
import subprocess
import logging
from typing import Optional
from ..utils import get_os

logger = logging.getLogger(__name__)

# Default wall-clock ceiling for a single LAMMPS single-point energy run. A wedged
# run must not hang the caller indefinitely (the project conventions: every subprocess.run
# needs a timeout). 1CRN runs in ~1 s; this is generous for large proteins.
DEFAULT_LAMMPS_TIMEOUT = 3600


class LammpsRunner:
    """Handles LAMMPS calculations for frustration analysis."""

    # get_os() value -> the suffix used in the committed binary names.
    _OS_SUFFIX = {"linux": "Linux", "osx": "MacOS"}

    def __init__(
        self,
        job_dir: str,
        pdb_base: str,
        seq_dist: int,
        scripts_dir: str,
        debug: bool = False,
        timeout: Optional[int] = DEFAULT_LAMMPS_TIMEOUT,
    ):
        """
        Initialize LAMMPS runner.

        Args:
            job_dir: Directory for job execution
            pdb_base: Base name for PDB files
            seq_dist: Sequence distance parameter
            scripts_dir: Directory containing LAMMPS scripts and executables
            debug: Enable debug logging
            timeout: Per-run wall-clock ceiling in seconds (None disables it)
        """
        self.job_dir = job_dir
        self.pdb_base = pdb_base
        self.seq_dist = seq_dist
        self.scripts_dir = scripts_dir
        self.debug = debug
        self.timeout = timeout
        self.os_type = get_os()

        logger.debug(f"Initialized LammpsRunner with:")
        logger.debug(f"  job_dir: {job_dir}")
        logger.debug(f"  pdb_base: {pdb_base}")
        logger.debug(f"  seq_dist: {seq_dist}")
        logger.debug(f"  scripts_dir: {scripts_dir}")
        logger.debug(f"  os_type: {self.os_type}")

    def run(self) -> None:
        """Run the LAMMPS single-point calculation.

        Unified across Linux and macOS: both copy the OS-appropriate binary into
        the job dir, then execute it with absolute paths, the input fed on stdin,
        and cwd pinned to the job dir. This fixes the historical macOS bugs
        (relative ``.in`` opened before chdir; stdin never actually fed to the
        process) while preserving the verified Linux behavior, and drops the
        ``shell=True`` form (P0-C/P0-D ≡ P1-5/6/7).
        """
        try:
            os_name = self._OS_SUFFIX[self.os_type]
        except KeyError:
            raise ValueError(f"Unsupported operating system: {self.os_type}")

        self._copy_lammps_executable(os_name)
        self._execute_lammps(os_name)

    def _copy_lammps_executable(self, os_name: str) -> None:
        """Copy the OS-appropriate LAMMPS executable into the job dir."""
        src = os.path.join(self.scripts_dir, f"lmp_serial_{self.seq_dist}_{os_name}")
        if not os.path.exists(src):
            raise FileNotFoundError(f"LAMMPS executable not found: {src}")
        subprocess.run(["cp", src, self.job_dir], check=True, timeout=self.timeout)

    def _execute_lammps(self, os_name: str) -> None:
        """Execute LAMMPS with the input on stdin and cwd pinned to the job dir."""
        executable = os.path.join(self.job_dir, f"lmp_serial_{self.seq_dist}_{os_name}")
        input_file = os.path.join(self.job_dir, f"{self.pdb_base}.in")

        if not os.path.exists(executable):
            raise FileNotFoundError(f"LAMMPS executable not found: {executable}")
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        os.chmod(executable, 0o755)

        logger.debug(f"Executing LAMMPS: {executable} < {input_file}")
        logger.debug(f"Working directory: {self.job_dir}")

        try:
            with open(input_file) as stdin_f:
                result = subprocess.run(
                    [executable],
                    stdin=stdin_f,
                    cwd=self.job_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )

            if self.debug:
                logger.debug("LAMMPS Output:")
                logger.debug(result.stdout)
                if result.stderr:
                    logger.debug("LAMMPS Errors:")
                    logger.debug(result.stderr)

        except subprocess.CalledProcessError as e:
            logger.error(f"LAMMPS execution failed with return code {e.returncode}")
            logger.error(f"Command: {e.cmd}")
            logger.error(f"Working directory: {self.job_dir}")
            logger.error(f"Directory contents: {os.listdir(self.job_dir)}")
            logger.error("Error details:")
            logger.error(e.stderr)
            raise
        except subprocess.TimeoutExpired as e:
            logger.error(f"LAMMPS execution timed out after {self.timeout}s")
            logger.error(f"Command: {e.cmd}")
            logger.error(f"Working directory: {self.job_dir}")
            raise
