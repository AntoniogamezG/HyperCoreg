"""
Logging configuration for HyperCoreg.

This module provides centralized logging setup with both console and file outputs.
"""

import os
import sys
import uuid
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

# Logger name used throughout the package
LOGGER_NAME = "COREG_PROCESSING"


def setup_logging(
    output_dir: Optional[str] = None,
    log_level: int = logging.INFO,
    verbose: bool = False
) -> logging.Logger:
    """
    Configure structured logging with both file and console outputs.

    Args:
        output_dir: Directory for log files. If None, logs only to console.
        log_level: Logging level (default: INFO)
        verbose: If True, use DEBUG level for console output.

    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)  # Capture all, filter at handler level
    logger.handlers.clear()

    # Console handler
    console_level = logging.DEBUG if verbose else log_level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(levelname)-8s | %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if output directory provided)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:4]
        log_filename = f"coreg_processing_{timestamp}_{unique_id}.log"
        log_path = os.path.join(output_dir, log_filename)

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_path}")

    return logger


def get_logger() -> logging.Logger:
    """
    Get the HyperCoreg logger instance.

    Returns:
        Logger instance
    """
    return logging.getLogger(LOGGER_NAME)


def log_section_header(title: str):
    """
    Log a formatted section header.

    Args:
        title: Section title to display
    """
    logger = get_logger()
    logger.info("")
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)


def log_subsection_header(title: str):
    """
    Log a formatted subsection header.

    Args:
        title: Subsection title to display
    """
    logger = get_logger()
    logger.info("")
    logger.info("-" * 40)
    logger.info(title)
    logger.info("-" * 40)


# Initialize default logger on import
_default_logger = logging.getLogger(LOGGER_NAME)
_default_logger.setLevel(logging.INFO)
if not _default_logger.handlers:
    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler.setFormatter(logging.Formatter('%(levelname)-8s | %(message)s'))
    _default_logger.addHandler(_console_handler)
