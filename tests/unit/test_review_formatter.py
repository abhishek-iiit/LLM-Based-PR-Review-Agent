"""Tests for the review formatter."""

from __future__ import annotations

import pytest

from pr_review_agent.models.state import (
    AgentState,
    CodeIssue,
    IssueType,
    PipelineStats,
    PRMetadata,
    PRSummary,
    RiskLevel,
    Severity,
    TestSuggestion,
    FileChange,
    FileStatus,
)
from pr_review_agent.services.review_poster import ReviewFormatter


@pytest.fixture()
def formatter() -> ReviewFormatter:
    return ReviewFormatter()


@pytest.fixture()
def full_state(sample_agent_state: AgentState) -> AgentState:
    return sample_agent_state


class TestReviewFormatter:
    def test_header_contains_pr_title(
        self, formatter: ReviewFormatter, full_state: AgentState
    ) -> None:
        output = formatter.format_full_review(full_state)
        assert "Add user authentication" in output

    def test_header_contains_author(
        self, formatter: ReviewFormatter, full_state: AgentState
    ) -> None:
        output = formatter.format_full_review(full_state)
        assert "@dev-user" in output

    def test_risk_badge_present_for_high_risk(
        self, formatter: ReviewFormatter, full_state: AgentState
    ) -> None:
        output = formatter.format_full_review(full_state)
        assert "risk-high" in output or "high" in output.lower()

    def test_summary_section_present(
        self, formatter: ReviewFormatter, full_state: AgentState
    ) -> None:
        output = formatter.format_full_review(full_state)
        assert "JWT-based user authentication" in output

    def test_issues_section_shows_file(
        self, formatter: ReviewFormatter, full_state: AgentState
    ) -> None:
        output = formatter.format_full_review(full_state)
        assert "src/auth/service.py" in output

    def test_issues_section_shows_severity_emoji(
        self, formatter: ReviewFormatter, full_state: AgentState
    ) -> None:
        output = formatter.format_full_review(full_state)
        # Critical = red circle
        assert "🔴" in output

    def test_test_suggestions_in_details_block(
        self, formatter: ReviewFormatter, full_state: AgentState
    ) -> None:
        output = formatter.format_full_review(full_state)
        assert "<details>" in output
        assert "authenticate" in output

    def test_footer_shows_stats(
        self, formatter: ReviewFormatter, full_state: AgentState
    ) -> None:
        output = formatter.format_full_review(full_state)
        assert "1,500" in output or "1500" in output  # total_tokens

    def test_no_issues_shows_checkmark(self, formatter: ReviewFormatter) -> None:
        state = AgentState(
            pr_number=1,
            repo="o/r",
            run_id="x",
            pr_metadata=None,
            file_changes=[],
            code_issues=[],
            test_suggestions=[],
            summary=None,
            stats=PipelineStats(),
            error=None,
            metadata={},
        )
        output = formatter.format_full_review(state)
        assert "No issues found" in output

    def test_truncation_at_limit(self, formatter: ReviewFormatter) -> None:
        long_text = "A" * 1000
        result = formatter.truncate(long_text, max_chars=500)
        assert len(result) <= 500
        assert "truncated" in result.lower()

    def test_no_truncation_within_limit(self, formatter: ReviewFormatter) -> None:
        text = "Short text"
        result = formatter.truncate(text, max_chars=1000)
        assert result == text

    def test_pipe_characters_escaped_in_table(self, formatter: ReviewFormatter) -> None:
        """Pipe chars in messages must be escaped for valid Markdown tables."""
        issue = CodeIssue(
            file="app.py",
            line=1,
            issue_type=IssueType.BUG,
            severity=Severity.MEDIUM,
            message="Use foo | bar pattern",
            suggestion="Replace with baz",
        )
        state = AgentState(
            pr_number=1,
            repo="o/r",
            run_id="x",
            pr_metadata=None,
            file_changes=[],
            code_issues=[issue],
            test_suggestions=[],
            summary=None,
            stats=PipelineStats(),
            error=None,
            metadata={},
        )
        output = formatter.format_full_review(state)
        # The pipe in the message must be escaped to avoid breaking the table
        assert "\\|" in output
