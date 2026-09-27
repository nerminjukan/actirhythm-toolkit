"""Logging configuration for the thesis analysis pipeline."""

import sys
from pathlib import Path
from loguru import logger


def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    """
    Configure logging for the pipeline.
    
    Parameters
    ----------
    log_dir : str, default="logs"
        Directory to store log files
    log_level : str, default="INFO"
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Remove default logger
    logger.remove()
    
    # Add console logger with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )
    
    # Add file logger for all messages
    logger.add(
        log_path / "pipeline_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="1 day",
        retention="30 days",
        compression="zip",
    )
    
    # Add separate error log
    logger.add(
        log_path / "errors_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="1 day",
        retention="30 days",
        compression="zip",
    )
    
    logger.info("Logging configured successfully")
    return logger


def log_parameters(params: dict, stage: str):
    """
    Log pipeline parameters.
    
    Parameters
    ----------
    params : dict
        Dictionary of parameters
    stage : str
        Pipeline stage name
    """
    logger.info(f"=== {stage} Parameters ===")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * (len(stage) + 20))


# Default setup
if __name__ != "__main__":
    # Auto-configure when imported
    setup_logging()
