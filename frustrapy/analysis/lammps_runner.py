import os
import subprocess
import logging
from typing import Optional
from ..utils import get_os

logger = logging.getLogger(__name__)


class LammpsRunner:
    """Handles LAMMPS calculations for frustration analysis."""

    def __init__(
        self,
        job_dir: str,
        pdb_base: str,
        seq_dist: int,
        scripts_dir: str,
        debug: bool = False,
    ):
        """
        Initialize LAMMPS runner.

        Args:
            job_dir: Directory for job execution
            pdb_base: Base name for PDB files
            seq_dist: Sequence distance parameter
            scripts_dir: Directory containing LAMMPS scripts and executables
            debug: Enable debug logging
        """
        self.job_dir = job_dir
        self.pdb_base = pdb_base
        self.seq_dist = seq_dist
        self.scripts_dir = scripts_dir
        self.debug = debug
        self.os_type = get_os()

        logger.debug(f"Initialized LammpsRunner with:")
        logger.debug(f"  job_dir: {job_dir}")
        logger.debug(f"  pdb_base: {pdb_base}")
        logger.debug(f"  seq_dist: {seq_dist}")
        logger.debug(f"  scripts_dir: {scripts_dir}")
        logger.debug(f"  os_type: {self.os_type}")

    def run(self) -> None:
        """Run LAMMPS calculation based on OS type."""
        if self.os_type == "linux":
            self._run_linux()
        elif self.os_type == "osx":
            self._run_osx()
        else:
            raise ValueError(f"Unsupported operating system: {self.os_type}")

    def _run_linux(self) -> None:
        """Run LAMMPS calculation on Linux."""
        self._copy_lammps_executable("Linux")
        self._execute_lammps_linux()

    def _run_osx(self) -> None:
        """Run LAMMPS calculation on MacOS."""
        self._copy_lammps_executable("MacOS")
        subprocess.run(["chmod", "+x", f"lmp_serial_{self.seq_dist}_MacOS"])
        self._execute_lammps_osx()

    def _copy_lammps_executable(self, os_name: str) -> None:
        """Copy LAMMPS executable for the appropriate OS."""
        subprocess.run(
            [
                "cp",
                os.path.join(self.scripts_dir, f"lmp_serial_{self.seq_dist}_{os_name}"),
                self.job_dir,
            ]
        )

    def _execute_lammps_linux(self) -> None:
        """Execute LAMMPS on Linux with error handling."""
        executable = os.path.join(self.job_dir, f"lmp_serial_{self.seq_dist}_Linux")
        input_file = os.path.join(self.job_dir, f"{self.pdb_base}.in")

        if not os.path.exists(executable):
            raise FileNotFoundError(f"LAMMPS executable not found: {executable}")
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        os.chmod(executable, 0o755)

        cmd = f"{executable} < {input_file}"
        logger.debug(f"Executing LAMMPS command: {cmd}")
        logger.debug(f"Working directory: {self.job_dir}")

        self._execute_lammps(cmd)

    def _execute_lammps_osx(self) -> None:
        """Execute LAMMPS on MacOS with error handling."""
        with open(f"{self.pdb_base}.in") as f:
            input_data = f.read()
        self._execute_lammps(f"./lmp_serial_{self.seq_dist}_MacOS", input_data)

    def _execute_lammps(self, cmd: str, input_data: Optional[str] = None) -> None:
        """Execute LAMMPS command with proper error handling."""
        try:
            logger.debug(f"Executing command: {cmd}")
            logger.debug(f"Working directory: {self.job_dir}")
            logger.debug(f"Directory contents: {os.listdir(self.job_dir)}")

            # Ensure we're in the correct directory
            original_dir = os.getcwd()
            os.chdir(self.job_dir)

            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    check=True,
                    capture_output=True,
                    text=True,
                )

                if self.debug:
                    logger.debug("LAMMPS Output:")
                    logger.debug(result.stdout)
                    if result.stderr:
                        logger.debug("LAMMPS Errors:")
                        logger.debug(result.stderr)

            finally:
                # Restore original directory
                os.chdir(original_dir)

        except subprocess.CalledProcessError as e:
            logger.error(f"LAMMPS execution failed with return code {e.returncode}")
            logger.error(f"Command: {e.cmd}")
            logger.error(f"Working directory: {self.job_dir}")
            logger.error(f"Directory contents: {os.listdir(self.job_dir)}")
            logger.error("Error details:")
            logger.error(e.stderr)
            raise
        except Exception as e:
            logger.error(f"Unexpected error running LAMMPS: {str(e)}")
            logger.error(f"Command: {cmd}")
            logger.error(f"Working directory: {self.job_dir}")
            raise
