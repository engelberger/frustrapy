import shutil
import os
import logging
from pathlib import Path
from ..analysis.exceptions import FileOperationError

logger = logging.getLogger(__name__)

def safe_copy(src: str, dst: str, overwrite: bool = False) -> None:
    """Safely copy a file from src to dst with verification."""
    src_path = Path(src)
    dst_path = Path(dst)
    dst_dir = dst_path.parent

    # Log detailed information about source and destination
    logger.debug(f"Copying file from {src_path} to {dst_path}")
    logger.debug(f"Source path (absolute): {src_path.absolute()}")
    logger.debug(f"Source exists: {src_path.exists()}")
    
    if src_path.exists():
        logger.debug(f"Source file size: {src_path.stat().st_size} bytes")
        logger.debug(f"Source readable: {os.access(str(src_path), os.R_OK)}")
    
    logger.debug(f"Destination path (absolute): {dst_path.absolute()}")
    logger.debug(f"Destination directory exists: {dst_dir.exists()}")
    
    if dst_dir.exists():
        logger.debug(f"Destination directory writable: {os.access(str(dst_dir), os.W_OK)}")
        logger.debug(f"Destination directory contents before copy: {[f.name for f in dst_dir.iterdir()]}")

    if not src_path.exists():
        raise FileOperationError(f"Source file not found", src=str(src_path), dst=str(dst_path))
    
    if dst_path.exists():
        if overwrite:
            logger.debug(f"Removing existing destination file: {dst_path}")
            dst_path.unlink()
        else:
            raise FileOperationError(f"Destination file already exists", src=str(src_path), dst=str(dst_path))
    
    try:
        # Ensure destination directory exists
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy file with metadata preservation
        shutil.copy2(str(src_path), str(dst_path))
        
        # Verify copy
        if not dst_path.exists():
            raise FileOperationError(f"Copy operation failed to create destination file", src=str(src_path), dst=str(dst_path))
        
        logger.debug(f"Destination file exists after copy: {dst_path.exists()}")
        logger.debug(f"Destination file size after copy: {dst_path.stat().st_size} bytes")
        logger.debug(f"Destination directory contents after copy: {[f.name for f in dst_dir.iterdir()]}")
        
        if src_path.stat().st_size != dst_path.stat().st_size:
            raise FileOperationError(f"File size mismatch after copy", src=str(src_path), dst=str(dst_path))
        
        logger.debug(f"Successfully copied file from {src_path} to {dst_path}")
    except Exception as e:
        logger.error(f"Failed to copy file: {e}")
        logger.error(f"Source path (absolute): {src_path.absolute()}")
        logger.error(f"Source exists: {src_path.exists()}")
        if src_path.exists():
            logger.error(f"Source readable: {os.access(str(src_path), os.R_OK)}")
        logger.error(f"Destination path (absolute): {dst_path.absolute()}")
        logger.error(f"Destination directory exists: {dst_dir.exists()}")
        if dst_dir.exists():
            logger.error(f"Destination directory writable: {os.access(str(dst_dir), os.W_OK)}")
        raise FileOperationError(f"File copy failed: {e}", src=str(src_path), dst=str(dst_path))

def safe_move(src: str, dst: str, overwrite: bool = False) -> None:
    """Safely move a file from src to dst with verification."""
    src_path = Path(src)
    dst_path = Path(dst)
    
    logger.debug(f"Moving file from {src_path} to {dst_path}")
    
    if not src_path.exists():
        raise FileOperationError(f"Source file not found", src=str(src_path), dst=str(dst_path))
    
    if dst_path.exists():
        if overwrite:
            logger.debug(f"Removing existing destination file: {dst_path}")
            dst_path.unlink()
        else:
            raise FileOperationError(f"Destination file already exists", src=str(src_path), dst=str(dst_path))
    
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dst_path))
        
        if not dst_path.exists():
            raise FileOperationError(f"Move operation failed to create destination file", src=str(src_path), dst=str(dst_path))
        
        logger.debug(f"Successfully moved file from {src_path} to {dst_path}")
    except Exception as e:
        logger.error(f"File move failed: {e}")
        raise FileOperationError(f"File move failed: {e}", src=str(src_path), dst=str(dst_path))

def ensure_file_exists(path: str) -> None:
    """Ensure that a file exists at the given path."""
    p = Path(path)
    if not p.exists():
        raise FileOperationError(f"Required file not found", src=path) 