"""
Logger Module
===========
This module provides logging configuration for the SVG Prompt Analyzer.
"""

import logging
import sys


def setup_logger(log_level: str = "INFO") -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Get the numeric level from the log level name
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure logging format
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("svg_generator.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create a logger instance
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at level {log_level}")
    
    return logger
