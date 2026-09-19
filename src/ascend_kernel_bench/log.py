"""Process-wide logging via loguru, with optional Rich tracebacks.

setup_logging is idempotent per process and is never called on import,
so pytest and embedding code keep their own logging setup.
"""

from __future__ import annotations

import sys

from loguru import logger

_CONFIGURED = False


def setup_logging(*, level: str = "INFO", rich_tracebacks: bool = True) -> None:
    """Configure loguru once per process; later calls are no-ops."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<level>{level: <8}</level> | {message}",
        colorize=True,
        backtrace=False,
        diagnose=False,
    )
    if rich_tracebacks:
        from rich.traceback import install

        install(show_locals=False, width=80)
    _CONFIGURED = True


def die(message: str, code: int = 1) -> None:
    """Log message at error level and exit the process."""
    logger.error(message)
    raise SystemExit(code)
