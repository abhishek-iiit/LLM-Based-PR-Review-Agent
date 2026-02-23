"""LangGraph pipeline definition.

Assembles the four agent nodes into a StateGraph with:
- Linear edges: pr_fetcher → code_reviewer → test_coverage → doc_summarizer
- Conditional routing: if pr_fetcher sets state["error"], route to error_handler
- In-memory checkpointing for local development
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from pr_review_agent.agents.code_reviewer import CodeReviewAgent
from pr_review_agent.agents.doc_summarizer import DocSummarizerAgent
from pr_review_agent.agents.pr_fetcher import PRFetcherAgent
from pr_review_agent.agents.test_coverage import TestCoverageAgent
from pr_review_agent.models.state import AgentState, PipelineStats
from pr_review_agent.utils.logging import (
    bind_pipeline_context,
    clear_pipeline_context,
    get_logger,
)

log = get_logger(__name__)

# ── Node names ────────────────────────────────────────────────────────────────

NODE_PR_FETCHER = "pr_fetcher"
NODE_CODE_REVIEWER = "code_reviewer"
NODE_TEST_COVERAGE = "test_coverage"
NODE_DOC_SUMMARIZER = "doc_summarizer"
NODE_ERROR_HANDLER = "error_handler"


# ── Error handler node ────────────────────────────────────────────────────────


def _error_handler_node(state: AgentState) -> dict[str, Any]:
    """Handle pipeline errors.

    Logs the error and returns a terminal state. Downstream nodes are skipped.
    """
    error = state.get("error", "Unknown error")
    log.error("pipeline error encountered, halting", error=error)
    return {
        "error": error,
        "summary": None,
    }


# ── Conditional routing ───────────────────────────────────────────────────────


def _route_after_fetcher(state: AgentState) -> str:
    """Route to error_handler if pr_fetcher failed, else continue."""
    if state.get("error"):
        return NODE_ERROR_HANDLER
    return NODE_CODE_REVIEWER


# ── Graph builder ─────────────────────────────────────────────────────────────


def build_graph(
    pr_fetcher: PRFetcherAgent | None = None,
    code_reviewer: CodeReviewAgent | None = None,
    test_coverage: TestCoverageAgent | None = None,
    doc_summarizer: DocSummarizerAgent | None = None,
) -> CompiledStateGraph:
    """Construct and compile the LangGraph StateGraph.

    Args:
        pr_fetcher: Custom PR fetcher agent (uses default if None).
        code_reviewer: Custom code reviewer agent (uses default if None).
        test_coverage: Custom test coverage agent (uses default if None).
        doc_summarizer: Custom doc summarizer agent (uses default if None).

    Returns:
        Compiled LangGraph app with MemorySaver checkpointing.
    """
    _pr_fetcher = pr_fetcher or PRFetcherAgent()
    _code_reviewer = code_reviewer or CodeReviewAgent()
    _test_coverage = test_coverage or TestCoverageAgent()
    _doc_summarizer = doc_summarizer or DocSummarizerAgent()

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node(NODE_PR_FETCHER, _pr_fetcher)
    graph.add_node(NODE_CODE_REVIEWER, _code_reviewer)
    graph.add_node(NODE_TEST_COVERAGE, _test_coverage)
    graph.add_node(NODE_DOC_SUMMARIZER, _doc_summarizer)
    graph.add_node(NODE_ERROR_HANDLER, _error_handler_node)

    # Entry point
    graph.add_edge(START, NODE_PR_FETCHER)

    # Conditional routing after fetcher
    graph.add_conditional_edges(
        NODE_PR_FETCHER,
        _route_after_fetcher,
        {
            NODE_CODE_REVIEWER: NODE_CODE_REVIEWER,
            NODE_ERROR_HANDLER: NODE_ERROR_HANDLER,
        },
    )

    # Linear edges for the happy path
    graph.add_edge(NODE_CODE_REVIEWER, NODE_TEST_COVERAGE)
    graph.add_edge(NODE_TEST_COVERAGE, NODE_DOC_SUMMARIZER)
    graph.add_edge(NODE_DOC_SUMMARIZER, END)
    graph.add_edge(NODE_ERROR_HANDLER, END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ── Public run function ───────────────────────────────────────────────────────


def run_pipeline(
    pr_number: int,
    repo: str,
    pr_fetcher: PRFetcherAgent | None = None,
    code_reviewer: CodeReviewAgent | None = None,
    test_coverage: TestCoverageAgent | None = None,
    doc_summarizer: DocSummarizerAgent | None = None,
) -> AgentState:
    """Run the full PR review pipeline synchronously.

    Args:
        pr_number: GitHub PR number.
        repo: Full repository name, e.g. 'owner/repo'.
        pr_fetcher: Optional custom PRFetcherAgent.
        code_reviewer: Optional custom CodeReviewAgent.
        test_coverage: Optional custom TestCoverageAgent.
        doc_summarizer: Optional custom DocSummarizerAgent.

    Returns:
        Final AgentState after all nodes have run.
    """
    run_id = str(uuid.uuid4())
    bind_pipeline_context(run_id=run_id, pr_number=pr_number, repo=repo)

    log.info("pipeline starting", pr_number=pr_number, repo=repo, run_id=run_id)
    pipeline_start = time.perf_counter()

    app = build_graph(
        pr_fetcher=pr_fetcher,
        code_reviewer=code_reviewer,
        test_coverage=test_coverage,
        doc_summarizer=doc_summarizer,
    )

    initial_state: AgentState = {
        "pr_number": pr_number,
        "repo": repo,
        "run_id": run_id,
        "pr_metadata": None,
        "file_changes": [],
        "code_issues": [],
        "test_suggestions": [],
        "summary": None,
        "stats": PipelineStats(),
        "error": None,
        "metadata": {},
    }

    config = {"configurable": {"thread_id": run_id}}
    final_state: AgentState = app.invoke(initial_state, config=config)

    total_elapsed = time.perf_counter() - pipeline_start

    if final_state.get("error"):
        log.error(
            "pipeline completed with error",
            error=final_state["error"],
            elapsed=round(total_elapsed, 2),
        )
    else:
        stats = final_state.get("stats", PipelineStats())
        log.info(
            "pipeline completed successfully",
            files_reviewed=stats.files_reviewed,
            issues_found=stats.issues_found,
            test_suggestions=stats.test_suggestions_count,
            total_tokens=stats.total_tokens_used,
            llm_calls=stats.llm_calls,
            elapsed=round(total_elapsed, 2),
        )

    clear_pipeline_context()
    return final_state
