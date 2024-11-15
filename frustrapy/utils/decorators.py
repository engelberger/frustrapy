import time
import logging
from functools import wraps


def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger = logging.getLogger(__name__)
        logger.debug(f"Starting {func.__name__} with args: {args}, kwargs: {kwargs}")

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds")
            logger.exception(f"Exception in {func.__name__}: {str(e)}")
            raise

    return wrapper


def log_memory_usage():
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        logger = logging.getLogger(__name__)
        logger.debug(
            f"Memory usage - RSS: {memory_info.rss / 1024 / 1024:.2f} MB, "
            f"VMS: {memory_info.vms / 1024 / 1024:.2f} MB"
        )
    except ImportError:
        logging.getLogger(__name__).debug(
            "psutil not installed - memory usage logging disabled"
        )
