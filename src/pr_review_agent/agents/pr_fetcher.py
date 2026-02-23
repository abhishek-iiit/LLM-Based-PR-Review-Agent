"""PR Fetcher Agent — first node in the LangGraph pipeline.

Fetches PR metadata and file changes from GitHub and populates the shared
AgentState for downstream agents.
"""

from __future__ import annotations

import time
from typing import Any

from github import GithubException

from pr_review_agent.models.state import AgentState, PipelineStats
from pr_review_agent.services.github_service import GitHubService
from pr_review_agent.utils.logging import bind_agent_context, get_logger

log = get_logger(__name__)

AGENT_NAME = "pr_fetcher"


class PRFetcherAgent:
    """LangGraph node that fetches PR data from GitHub.

    Injects pr_metadata and file_changes into AgentState.
    On failure, sets state["error"] so the graph can route to error_handler.

    Args:
        github_service: GitHub service instance. Created from settings if not provided.
    """

    def __init__(self, github_service: GitHubService | None = None) -> None:
        self._github = github_service or GitHubService()

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """Execute the PR fetch step.

        Args:
            state: Current pipeline state. Must contain `pr_number` and `repo`.

        Returns:
            Partial state dict with pr_metadata, file_changes, and updated stats.
        """
        bind_agent_context(AGENT_NAME)
        start = time.perf_counter()

        pr_number: int = state["pr_number"]
        repo: str = state["repo"]

        log.info("PR fetcher agent starting", pr_number=pr_number, repo=repo)

        try:
            metadata = self._github.get_pr_metadata(repo, pr_number)
            files = self._github.get_pr_files(repo, pr_number)
        except GithubException as exc:
            error_msg = (
                f"GitHub API error fetching PR #{pr_number} from {repo}: "
                f"HTTP {exc.status} — {exc.data}"
            )
            log.error("PR fetcher failed", error=error_msg)
            return {
                "error": error_msg,
                "pr_metadata": None,
                "file_changes": [],
            }
        except Exception as exc:
            error_msg = f"Unexpected error in PR fetcher: {exc}"
            log.exception("PR fetcher unexpected error")
            return {
                "error": error_msg,
                "pr_metadata": None,
                "file_changes": [],
            }

        elapsed = time.perf_counter() - start
        log.info(
            "PR fetcher agent complete",
            title=metadata.title,
            author=metadata.author,
            files_fetched=len(files),
            additions=metadata.additions,
            deletions=metadata.deletions,
            elapsed=round(elapsed, 2),
        )

        # Merge stats with whatever is already in state
        existing_stats: PipelineStats = state.get("stats", PipelineStats())
        updated_stats = PipelineStats(
            total_tokens_used=existing_stats.total_tokens_used,
            llm_calls=existing_stats.llm_calls,
            files_reviewed=len(files),
            issues_found=existing_stats.issues_found,
            test_suggestions_count=existing_stats.test_suggestions_count,
            duration_seconds=existing_stats.duration_seconds + elapsed,
        )

        return {
            "pr_metadata": metadata,
            "file_changes": files,
            "error": None,
            "stats": updated_stats,
        }
