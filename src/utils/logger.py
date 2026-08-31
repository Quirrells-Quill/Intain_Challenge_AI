import logging
import sys
from pathlib import Path

def get_logger(name: str, log_file: str = "app.log", level: int = logging.INFO) -> logging.Logger:
    """
    Creates and configures a standardized logger for Intain-Sight.
    
    This logger enforces both console output for interactive sessions and
    file-based logging to support the Verification Agent's audit trail.
    
    Args:
        name (str): Name of the logger, typically __name__ of the calling module.
        log_file (str): Name of the log file to write to. Default is "app.log".
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        
    Returns:
        logging.Logger: A configured logger instance ready for use.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger is instantiated multiple times
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(level)
    
    # Standard format containing timestamp, module, level, and message
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console Handler for real-time monitoring
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler for audit trailing and historical logs
    log_path = Path("reports") / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
