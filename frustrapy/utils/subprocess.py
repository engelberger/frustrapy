import subprocess
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from ..analysis.exceptions import SubprocessError

logger = logging.getLogger(__name__)

@dataclass
class SubprocessResult:
    """Holds the result of a subprocess execution with additional parsed information."""
    returncode: int
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    @property
    def has_warnings(self) -> bool:
        """Return True if warnings were detected in stderr."""
        return len(self.warnings) > 0

def run_subprocess(cmd, cwd=None, timeout=300, check=True, capture_output=True, text=True):
    """Run subprocess with robust error handling and logging."""
    logger.debug(f"Running command: {' '.join(cmd)} in {cwd or 'current working directory'}")
    try:
        result = subprocess.run(
            cmd,
            check=check,
            cwd=cwd,
            timeout=timeout,
            capture_output=capture_output,
            text=text,
        )
        
        # Log outputs
        if result.stdout:
            logger.debug(f"Command stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Command stderr:\n{result.stderr}")
        
        # Parse warnings from stderr if available
        warnings = []
        if result.stderr:
            for line in result.stderr.splitlines():
                if "WARNING" in line:
                    warnings.append(line.strip())
        
        # Return enhanced result object
        return SubprocessResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            warnings=warnings
        )
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Command '{' '.join(e.cmd)}' failed with return code {e.returncode}"  
        logger.error(error_msg)
        raise SubprocessError(
            error_msg,
            cmd=list(e.cmd),
            returncode=e.returncode,
            stdout=e.stdout,
            stderr=e.stderr,
        )
    except subprocess.TimeoutExpired as e:
        error_msg = f"Command '{' '.join(e.cmd)}' timed out after {e.timeout} seconds"
        logger.error(error_msg)
        raise SubprocessError(error_msg, cmd=list(e.cmd)) 