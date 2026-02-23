"""Structured logging configuration.

Uses structlog with JSON output for production and pretty console output for
local development. Call `configure_logging()` once at application startup,
then use `get_logger(__name__)` everywhere else.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger


def _add_log_level(
    logger: WrappedLogger,  # noqa: ARG001
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Inject the log level string into every log record."""
    event_dict["level"] = method_name.upper()
    return event_dict


def configure_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """Initialise structlog for the application.

    Args:
        log_level: One of DEBUG / INFO / WARNING / ERROR.
        log_format: 'json' for production, 'console' for human-readable dev output.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if log_format == "console":
        renderer: Any = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors
        + [
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Also silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "github"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a bound structlog logger for the given module.

    Usage::

        log = get_logger(__name__)
        log.info("fetching PR", pr_number=42, repo="owner/repo")
    """
    return structlog.get_logger().bind(logger=name)


def bind_pipeline_context(run_id: str, pr_number: int, repo: str) -> None:
    """Bind pipeline-level context vars so they appear in every log line.

    Call this once at the start of a pipeline run. The context is
    thread-local (structlog.contextvars) and cleared between requests
    in the FastAPI webhook handler.
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        run_id=run_id,
        pr_number=pr_number,
        repo=repo,
    )


def bind_agent_context(agent_name: str) -> None:
    """Bind the current agent name so log lines show which agent emitted them."""
    structlog.contextvars.bind_contextvars(agent=agent_name)


def clear_pipeline_context() -> None:
    """Clear all bound context vars after a pipeline run completes."""
    structlog.contextvars.clear_contextvars()
