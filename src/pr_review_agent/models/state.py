"""Domain models and LangGraph shared state schema.

All data classes are Pydantic models for validation and serialisation.
The `AgentState` TypedDict is the single mutable object that flows through
every node in the LangGraph pipeline.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# ── Enumerations ──────────────────────────────────────────────────────────────


class Severity(StrEnum):
    """Issue severity levels, ordered from most to least severe."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueType(StrEnum):
    """Category of a code issue."""

    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    MAINTAINABILITY = "maintainability"
    TEST = "test"


class RiskLevel(StrEnum):
    """Overall PR risk level for the summary."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FileStatus(StrEnum):
    """GitHub file change status."""

    ADDED = "added"
    MODIFIED = "modified"
    REMOVED = "removed"
    RENAMED = "renamed"
    COPIED = "copied"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


# ── Domain Models ─────────────────────────────────────────────────────────────


class PRMetadata(BaseModel):
    """Metadata for a GitHub pull request."""

    pr_number: int = Field(..., gt=0, description="PR number")
    repo: str = Field(..., description="Full repo name, e.g. 'owner/repo'")
    title: str = Field(..., description="PR title")
    author: str = Field(..., description="GitHub username of the PR author")
    base_branch: str = Field(..., description="Target branch (e.g. main)")
    head_branch: str = Field(..., description="Source branch")
    url: str = Field(..., description="HTML URL to the PR on GitHub")
    body: str = Field(default="", description="PR description body")
    additions: int = Field(default=0, ge=0, description="Total lines added")
    deletions: int = Field(default=0, ge=0, description="Total lines removed")
    changed_files: int = Field(default=0, ge=0, description="Number of files changed")


class FileChange(BaseModel):
    """Represents a single file changed in a PR."""

    filename: str = Field(..., description="File path relative to repo root")
    status: FileStatus = Field(..., description="Change status")
    additions: int = Field(default=0, ge=0)
    deletions: int = Field(default=0, ge=0)
    patch: str = Field(default="", description="Unified diff patch string")
    language: str = Field(default="", description="Detected programming language")

    @property
    def is_binary(self) -> bool:
        """Binary files have no patch."""
        return not self.patch

    @property
    def is_test_file(self) -> bool:
        """Heuristic: files under test*/ directories or named test_*.py."""
        lower = self.filename.lower()
        return (
            "/test" in lower
            or lower.startswith("test")
            or lower.endswith("_test.py")
            or lower.endswith("_spec.js")
            or lower.endswith(".test.ts")
        )


class CodeIssue(BaseModel):
    """A single code issue found by static analysis or LLM review."""

    file: str = Field(..., description="File path where the issue was found")
    line: int | None = Field(default=None, ge=1, description="Line number, if known")
    issue_type: IssueType = Field(..., description="Category of the issue")
    severity: Severity = Field(..., description="Issue severity")
    message: str = Field(..., description="Human-readable description of the issue")
    suggestion: str = Field(default="", description="Recommended fix or action")
    source: str = Field(
        default="llm",
        description="Origin of the issue: 'static' for rule-based, 'llm' for AI-detected",
    )


class TestSuggestion(BaseModel):
    """A pytest stub suggestion for an untested function or class."""

    file: str = Field(..., description="Source file that needs testing")
    symbol_name: str = Field(..., description="Function or class name to be tested")
    symbol_type: str = Field(default="function", description="'function' or 'class'")
    test_stub: str = Field(..., description="Generated pytest stub code")


class PRSummary(BaseModel):
    """Structured summary produced by the Doc Summarizer agent."""

    purpose: str = Field(..., description="What this PR accomplishes")
    risk_level: RiskLevel = Field(..., description="Overall risk assessment")
    key_changes: list[str] = Field(
        default_factory=list, description="Bullet-point list of significant changes"
    )
    focus_areas: list[str] = Field(
        default_factory=list, description="Areas reviewers should pay extra attention to"
    )
    breaking_changes: bool = Field(
        default=False, description="Whether the PR contains likely breaking changes"
    )


class PipelineStats(BaseModel):
    """Runtime statistics accumulated across the pipeline run."""

    total_tokens_used: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    files_reviewed: int = Field(default=0, ge=0)
    issues_found: int = Field(default=0, ge=0)
    test_suggestions_count: int = Field(default=0, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0.0)


# ── LangGraph Shared State ─────────────────────────────────────────────────────


def _merge_issues(left: list[CodeIssue], right: list[CodeIssue]) -> list[CodeIssue]:
    """Reducer that appends new issues to the existing list."""
    return left + right


def _merge_suggestions(
    left: list[TestSuggestion], right: list[TestSuggestion]
) -> list[TestSuggestion]:
    """Reducer that appends new test suggestions to the existing list."""
    return left + right


class AgentState(TypedDict, total=False):
    """Shared mutable state threaded through every LangGraph node.

    Fields are populated incrementally as each agent runs:
      - pr_fetcher       → pr_metadata, file_changes
      - code_reviewer    → code_issues
      - test_coverage    → test_suggestions
      - doc_summarizer   → summary
      - any agent        → error (triggers error_handler routing)

    The `Annotated` reducers tell LangGraph how to merge partial updates
    when a node returns only a subset of the state.
    """

    # Inputs (set before the graph runs)
    pr_number: int
    repo: str
    run_id: str  # UUID for this pipeline run, used in logs

    # PR Fetcher outputs
    pr_metadata: PRMetadata | None
    file_changes: list[FileChange]

    # Code Reviewer outputs (reducer: append)
    code_issues: Annotated[list[CodeIssue], _merge_issues]

    # Test Coverage outputs (reducer: append)
    test_suggestions: Annotated[list[TestSuggestion], _merge_suggestions]

    # Doc Summarizer output
    summary: PRSummary | None

    # Accumulated stats
    stats: PipelineStats

    # Error state — if set, graph routes to error_handler instead of next agent
    error: str | None

    # Arbitrary metadata agents can attach (e.g. LLM token counts per step)
    metadata: dict[str, Any]
