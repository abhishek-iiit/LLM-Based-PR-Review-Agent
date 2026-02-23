"""Tests for domain models and state schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pr_review_agent.models.state import (
    CodeIssue,
    FileChange,
    FileStatus,
    IssueType,
    PRMetadata,
    PRSummary,
    RiskLevel,
    Severity,
)
from pr_review_agent.models.state import (
    TestSuggestion as Suggestion,
)


class TestPRMetadata:
    def test_valid_metadata(self) -> None:
        m = PRMetadata(
            pr_number=1,
            repo="owner/repo",
            title="Test PR",
            author="alice",
            base_branch="main",
            head_branch="feature",
            url="https://github.com/owner/repo/pull/1",
        )
        assert m.pr_number == 1
        assert m.author == "alice"

    def test_pr_number_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            PRMetadata(
                pr_number=0,
                repo="owner/repo",
                title="t",
                author="a",
                base_branch="main",
                head_branch="feat",
                url="https://github.com",
            )

    def test_body_defaults_to_empty_string(self) -> None:
        m = PRMetadata(
            pr_number=1,
            repo="o/r",
            title="t",
            author="a",
            base_branch="main",
            head_branch="feat",
            url="https://github.com",
        )
        assert m.body == ""


class TestFileChange:
    def test_is_binary_when_no_patch(self) -> None:
        fc = FileChange(filename="logo.png", status=FileStatus.ADDED, patch="")
        assert fc.is_binary is True

    def test_is_not_binary_when_patch_exists(self) -> None:
        fc = FileChange(filename="app.py", status=FileStatus.MODIFIED, patch="@@ -1 +1 @@\n+x=1")
        assert fc.is_binary is False

    def test_is_test_file_detection(self) -> None:
        assert FileChange(filename="tests/unit/test_foo.py", status=FileStatus.ADDED).is_test_file
        assert FileChange(filename="foo_test.py", status=FileStatus.ADDED).is_test_file
        assert not FileChange(filename="src/foo.py", status=FileStatus.MODIFIED).is_test_file

    def test_language_defaults_empty(self) -> None:
        fc = FileChange(filename="Makefile", status=FileStatus.MODIFIED)
        assert fc.language == ""


class TestCodeIssue:
    def test_valid_issue(self) -> None:
        issue = CodeIssue(
            file="app.py",
            line=10,
            issue_type=IssueType.SECURITY,
            severity=Severity.CRITICAL,
            message="eval() used",
            suggestion="Remove eval()",
        )
        assert issue.source == "llm"  # default
        assert issue.line == 10

    def test_line_can_be_none(self) -> None:
        issue = CodeIssue(
            file="app.py",
            issue_type=IssueType.BUG,
            severity=Severity.LOW,
            message="some bug",
        )
        assert issue.line is None


class TestTestSuggestion:
    def test_valid_suggestion(self) -> None:
        ts = Suggestion(
            file="app.py",
            symbol_name="my_func",
            symbol_type="function",
            test_stub="def test_my_func(): pass",
        )
        assert ts.symbol_name == "my_func"

    def test_symbol_type_defaults(self) -> None:
        ts = Suggestion(file="a.py", symbol_name="fn", test_stub="...")
        assert ts.symbol_type == "function"


class TestPRSummary:
    def test_valid_summary(self) -> None:
        s = PRSummary(
            purpose="Adds auth",
            risk_level=RiskLevel.HIGH,
        )
        assert s.key_changes == []
        assert s.focus_areas == []
        assert s.breaking_changes is False

    def test_invalid_risk_level_raises(self) -> None:
        with pytest.raises(ValidationError):
            PRSummary(purpose="test", risk_level="unknown_level")  # type: ignore[arg-type]
