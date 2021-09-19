import logging
from enum import Enum


class LogLevel(str, Enum):
    critical = "CRITICAL"
    fatal = "FATAL"
    error = "ERROR"
    warning = "WARNING"
    info = "INFO"
    debug = "DEBUG"

    def __str__(self) -> str:
        return str(self.value)


def basicConfig(level: LogLevel = LogLevel.info, **kwargs):
    # Set default configuration
    kwargs.setdefault("format", "%(asctime)s.%(msecs)03d [%(levelname)s]  %(message)s")
    kwargs.setdefault("datefmt", "%Y-%m-%d %H:%M:%S")

    # Configure logging
    logging.basicConfig(level=str(level), **kwargs)


def getLogger(name: str) -> logging.Logger:
    return logging.getLogger(name)
