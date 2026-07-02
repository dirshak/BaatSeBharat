import logging
import os
import sys
from datetime import datetime

def setup_logger(name, log_dir='./logs', level=logging.INFO):
    """
    Set up logger with file and console handlers
    
    Args:
        name: Logger name (usually __name__)
        log_dir: Directory for log files
        level: Logging level
    
    Returns:
        logger: Configured logger instance
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler (detailed). Explicit utf-8 because log messages
    # throughout this codebase use unicode symbols (checkmarks, emoji) and
    # the default Windows console codepage (cp1252) can't encode them --
    # without this, FileHandler.emit() raises UnicodeEncodeError on every
    # such message, which logging.Handler.handleError() silently swallows,
    # so the log line is just lost rather than the process crashing.
    log_file = os.path.join(
        log_dir,
        f"{datetime.now().strftime('%Y%m%d')}_{name.replace('.', '_')}.log"
    )
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(detailed_formatter)

    # Console handler (simple). Same unicode issue applies to stdout on
    # Windows; reconfigure it to utf-8 with a safe fallback instead of
    # raising/losing the message.
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Example usage
if __name__ == "__main__":
    logger = setup_logger(__name__)
    logger.info("Logger setup complete")
    logger.debug("This is a debug message")
    logger.warning("This is a warning")
    logger.error("This is an error")
