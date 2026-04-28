"""
Following https://github.com/mCodingLLC/VideosSampleCode/tree/master/videos/135_modern_logging
"""

from __future__ import annotations

import contextlib
import logging


def get_logger(log_context: str | None = None) -> logging.Logger:
    """Get a logger for a context. Avoid making it too granular!"""
    if log_context is None:
        log_context = "wrf_massive"
    else:
        log_context = f"wrf_massive.{log_context}"
    logger = logging.getLogger(log_context)
    return logger


def get_logger_stream_redirect(
    log_context: str | None = None, level_stdout: str = "info", level_stderr: str = "warning"
) -> logging.Logger:
    """Get a logger and set it up for redirecting stdout/stderr to it."""
    # Base logger
    logger = get_logger(log_context)

    # Link levels to logger methods
    log_stdout = getattr(logger, level_stdout.lower())
    log_stderr = getattr(logger, level_stderr.lower())

    # Add stream handler and write function to support redirecting stdout/stderr to logger.
    logger.addHandler(logging.StreamHandler())
    logger.write = lambda msg: log_stdout(msg.strip())
    logger.error = lambda msg: log_stderr(msg.strip())

    return logger


@contextlib.contextmanager
def warnings_to_logger(logger: logging.Logger):
    """Redirect warnings to the logger.

    See Also
    --------
    https://docs.python.org/3/library/warnings.html#warnings.catch_warnings

    """
    import warnings

    def warning_to_logger(message, category, filename, lineno, file=None, line=None):
        """Redirect warnings to logger with standard formatting."""
        warn_str = warnings.formatwarning(message, category, filename, lineno, line)
        logger.warning(warn_str.strip())

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = warning_to_logger
        yield
