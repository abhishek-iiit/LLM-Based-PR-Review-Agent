"""FastAPI webhook server for GitHub PR events.

Receives GitHub pull_request webhooks and triggers the review pipeline
asynchronously, responding to GitHub immediately with HTTP 200.

Endpoints:
    GET  /health         — liveness probe
    POST /webhook/github — GitHub webhook receiver
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from pr_review_agent.config.settings import Settings, get_settings
from pr_review_agent.utils.logging import configure_logging, get_logger

_START_TIME = time.time()
log = get_logger(__name__)


# ── Application startup ────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Configure logging and validate settings on startup."""
    settings = get_settings()
    configure_logging(log_level=settings.log_level, log_format=settings.log_format)
    log.info(
        "PR Review Agent webhook server starting",
        model=settings.llm_model,
        port=settings.port,
        webhook_validation=bool(settings.webhook_secret),
    )
    yield
    log.info("PR Review Agent webhook server shutting down")


app = FastAPI(
    title="PR Review Agent",
    description="Automated GitHub PR review powered by Claude and LangGraph.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _verify_github_signature(payload: bytes, signature_header: str, secret: str) -> bool:
    """Verify GitHub webhook HMAC-SHA256 signature.

    Args:
        payload: Raw request body bytes.
        signature_header: Value of X-Hub-Signature-256 header.
        secret: Configured webhook secret.

    Returns:
        True if signature is valid, False otherwise.
    """
    if not signature_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)


async def _run_pipeline_background(pr_number: int, repo: str) -> None:
    """Run the review pipeline in a background async task.

    Errors are caught and logged — they must NOT propagate to prevent
    crashing the webhook server.
    """
    try:
        # Run blocking pipeline in threadpool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            _execute_pipeline,
            pr_number,
            repo,
        )
    except Exception:
        log.exception(
            "background pipeline task failed",
            pr_number=pr_number,
            repo=repo,
        )


def _execute_pipeline(pr_number: int, repo: str) -> None:
    """Synchronous pipeline runner called from a thread pool."""
    from pr_review_agent.graph.pipeline import run_pipeline
    from pr_review_agent.services.review_poster import ReviewPoster

    state = run_pipeline(pr_number=pr_number, repo=repo)

    if state.get("error"):
        log.error("pipeline finished with error", error=state["error"])
        return

    poster = ReviewPoster()
    poster.post_review(state, repo=repo, pr_number=pr_number)


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["ops"])
async def health() -> JSONResponse:
    """Liveness probe. Returns server status and uptime."""
    return JSONResponse(
        content={
            "status": "ok",
            "version": "0.1.0",
            "uptime_seconds": round(time.time() - _START_TIME, 1),
        }
    )


@app.post("/webhook/github", tags=["webhook"])
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str | None = Header(default=None),
    x_github_event: str | None = Header(default=None),
) -> JSONResponse:
    """Receive and process GitHub pull_request webhook events.

    Validates the HMAC signature (if WEBHOOK_SECRET is configured), then
    schedules the review pipeline as a background task and returns 200 immediately.
    """
    settings: Settings = get_settings()
    payload_bytes = await request.body()

    # ── Signature verification ─────────────────────────────────────────────
    if settings.webhook_secret:
        if not x_hub_signature_256:
            log.warning("webhook received without signature header")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing X-Hub-Signature-256 header.",
            )
        if not _verify_github_signature(
            payload_bytes, x_hub_signature_256, settings.webhook_secret
        ):
            log.warning("webhook signature verification failed")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook signature.",
            )

    # ── Event filtering ────────────────────────────────────────────────────
    if x_github_event != "pull_request":
        log.debug("ignoring non-PR event", event=x_github_event)
        return JSONResponse(content={"status": "ignored", "reason": f"event={x_github_event}"})

    payload: dict[str, Any] = await request.json()
    action: str = payload.get("action", "")

    if action not in ("opened", "synchronize", "reopened"):
        log.debug("ignoring PR action", action=action)
        return JSONResponse(content={"status": "ignored", "reason": f"action={action}"})

    # ── Extract PR details ─────────────────────────────────────────────────
    pr = payload.get("pull_request", {})
    pr_number: int | None = pr.get("number")
    repo_data = payload.get("repository", {})
    repo: str | None = repo_data.get("full_name")

    if not pr_number or not repo:
        log.error("missing pr_number or repo in webhook payload")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Payload missing pull_request.number or repository.full_name.",
        )

    log.info(
        "webhook received, scheduling review",
        action=action,
        repo=repo,
        pr_number=pr_number,
    )

    # Schedule pipeline as background task (non-blocking)
    background_tasks.add_task(_run_pipeline_background, pr_number, repo)

    return JSONResponse(
        content={
            "status": "accepted",
            "pr_number": pr_number,
            "repo": repo,
            "action": action,
        },
        status_code=status.HTTP_202_ACCEPTED,
    )
