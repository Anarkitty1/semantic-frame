"""
Logging configuration for the benchmark suite.

Provides a centralized logger with configurable verbosity levels.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

# Module-level logger
logger = logging.getLogger("benchmarks")

# Log level names for configuration
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(
    level: LogLevel = "INFO",
    format_string: str | None = None,
    stream: bool = True,
    file_path: str | None = None,
) -> logging.Logger:
    """Configure the benchmark logger.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string. Defaults to timestamp + level + message.
        stream: Whether to output to stderr
        file_path: Optional file path to write logs to

    Returns:
        Configured logger instance

    Example:
        >>> from benchmarks.logging_config import setup_logging, logger
        >>> setup_logging(level="DEBUG")
        >>> logger.info("Starting benchmark run")
    """
    # Clear existing handlers
    logger.handlers.clear()

    # Set level
    logger.setLevel(getattr(logging, level))

    # Default format
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)s] %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Add stream handler
    if stream:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Add file handler
    if file_path:
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a child logger for a specific module.

    Args:
        name: Optional name for the child logger. If None, returns the main logger.

    Returns:
        Logger instance

    Example:
        >>> from benchmarks.logging_config import get_logger
        >>> log = get_logger("claude_client")
        >>> log.debug("API call completed")
    """
    if name is None:
        return logger
    return logger.getChild(name)
