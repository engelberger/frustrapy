import shutil
from pathlib import Path
from ..analysis.exceptions import FileOperationError

def safe_copy(src: str, dst: str, overwrite: bool = False) -> None:
    """Safely copy a file from src to dst with verification."""
    src_path = Path(src)
    dst_path = Path(dst)

    if not src_path.exists():
        raise FileOperationError(f"Source file not found", src=str(src_path), dst=str(dst_path))
    if dst_path.exists() and not overwrite:
        raise FileOperationError(f"Destination file already exists", src=str(src_path), dst=str(dst_path))
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src_path), str(dst_path))
        # Verify copy
        if not dst_path.exists():
            raise FileOperationError(f"Copy operation failed to create destination file", src=str(src_path), dst=str(dst_path))
        if src_path.stat().st_size != dst_path.stat().st_size:
            raise FileOperationError(f"File size mismatch after copy", src=str(src_path), dst=str(dst_path))
    except Exception as e:
        raise FileOperationError(f"File copy failed: {e}", src=str(src_path), dst=str(dst_path))

def safe_move(src: str, dst: str, overwrite: bool = False) -> None:
    """Safely move a file from src to dst with verification."""
    src_path = Path(src)
    dst_path = Path(dst)
    if not src_path.exists():
        raise FileOperationError(f"Source file not found", src=str(src_path), dst=str(dst_path))
    if dst_path.exists():
        if overwrite:
            dst_path.unlink()
        else:
            raise FileOperationError(f"Destination file already exists", src=str(src_path), dst=str(dst_path))
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dst_path))
        if not dst_path.exists():
            raise FileOperationError(f"Move operation failed to create destination file", src=str(src_path), dst=str(dst_path))
    except Exception as e:
        raise FileOperationError(f"File move failed: {e}", src=str(src_path), dst=str(dst_path))

def ensure_file_exists(path: str) -> None:
    """Ensure that a file exists at the given path."""
    p = Path(path)
    if not p.exists():
        raise FileOperationError(f"Required file not found", src=path) 