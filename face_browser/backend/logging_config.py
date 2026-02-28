"""Centralized logging configuration for the Face Browser backend.

This module provides consistent logging setup across all services with:
- Colored console output for better readability
- Progress indicators for long-running operations
- Structured format with timestamps and module names
"""

import logging
import sys
import warnings
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color-coded log levels for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        
        # Format the message
        result = super().format(record)
        
        # Reset levelname for future use
        record.levelname = levelname
        
        return result


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    use_colors: bool = True
) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs to
        use_colors: Use colored output for console (default: True)
    """
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Suppress specific third-party warnings
    warnings.filterwarnings("ignore", message=".*`rcond` parameter will change.*", category=FutureWarning)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if use_colors:
        console_format = ColoredFormatter(
            fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        console_format = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Convenience function for progress logging
def log_progress(logger: logging.Logger, current: int, total: int, message: str = "") -> None:
    """
    Log progress with percentage and optional message.
    
    Args:
        logger: Logger instance
        current: Current progress count
        total: Total count
        message: Optional message prefix
    """
    percentage = (current / total * 100) if total > 0 else 0
    prefix = f"{message}: " if message else ""
    logger.info(f"🎯 {prefix}{current}/{total} ({percentage:.1f}%)")
