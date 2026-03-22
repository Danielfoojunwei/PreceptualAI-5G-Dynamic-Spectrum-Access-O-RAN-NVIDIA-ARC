"""
Structured logging configuration for SpectrAI.

Uses ``structlog`` with JSON output for production deployments and
coloured console output for development.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import structlog


_CONFIGURED = False


def configure_logging(
    level: str = "INFO",
    fmt: str = "json",
    log_file: Optional[str] = None,
) -> None:
    """
    Configure structured logging for the process.

    Should be called once at application startup.  Subsequent calls are
    silently ignored to prevent double-configuration.

    Args:
        level: Log level string (``DEBUG``, ``INFO``, ``WARNING``, etc.).
        fmt: Output format — ``"json"`` for production, ``"console"`` for
             human-readable development output.
        log_file: Optional path to a log file.  When set, a
                  ``logging.FileHandler`` is added alongside the console
                  handler.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Common processors shared by all formatters
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    # Configure structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging (used by libraries and structlog's formatter)
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)

    handlers = [console_handler]

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    for h in handlers:
        root_logger.addHandler(h)
    root_logger.setLevel(numeric_level)

    # Suppress noisy third-party loggers
    for name in ("grpc", "hpack", "urllib3", "asyncio"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None, **initial_context: object) -> structlog.stdlib.BoundLogger:
    """
    Return a structured logger bound to the given name.

    Any keyword arguments are added as initial context fields on the
    logger instance.

    Usage::

        log = get_logger(__name__, component="xapp")
        log.info("started", port=4560)
        # -> {"event": "started", "component": "xapp", "port": 4560, ...}
    """
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger
