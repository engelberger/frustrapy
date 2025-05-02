import subprocess
import logging
from ..analysis.exceptions import SubprocessError

logger = logging.getLogger(__name__)

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
        if result.stdout:
            logger.debug(f"Command stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Command stderr:\n{result.stderr}")
        return result
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