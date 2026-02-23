"""Shared pytest fixtures for unit tests."""

from __future__ import annotations

import pytest

from pr_review_agent.models.state import (
    AgentState,
    CodeIssue,
    FileChange,
    FileStatus,
    IssueType,
    PipelineStats,
    PRMetadata,
    PRSummary,
    RiskLevel,
    Severity,
    TestSuggestion,
)


@pytest.fixture()
def sample_pr_metadata() -> PRMetadata:
    return PRMetadata(
        pr_number=42,
        repo="owner/repo",
        title="Add user authentication",
        author="dev-user",
        base_branch="main",
        head_branch="feature/auth",
        url="https://github.com/owner/repo/pull/42",
        body="Adds JWT-based authentication.",
        additions=150,
        deletions=20,
        changed_files=5,
    )


@pytest.fixture()
def sample_python_file_change() -> FileChange:
    """A Python file with both added and removed lines."""
    return FileChange(
        filename="src/auth/service.py",
        status=FileStatus.MODIFIED,
        additions=10,
        deletions=3,
        patch=(
            "@@ -1,5 +1,12 @@\n"
            " import os\n"
            "+import pickle\n"
            " \n"
            "+def authenticate(token):\n"
            "+    data = pickle.loads(token)  # unsafe\n"
            "+    secret = 'hardcoded_secret123'\n"
            "+    print('debug info:', data)\n"
            "+    return data\n"
            " \n"
            "-def old_func():\n"
            "-    pass\n"
        ),
        language="python",
    )


@pytest.fixture()
def sample_binary_file_change() -> FileChange:
    return FileChange(
        filename="assets/logo.png",
        status=FileStatus.ADDED,
        additions=0,
        deletions=0,
        patch="",
        language="unknown",
    )


@pytest.fixture()
def sample_code_issue() -> CodeIssue:
    return CodeIssue(
        file="src/auth/service.py",
        line=5,
        issue_type=IssueType.SECURITY,
        severity=Severity.CRITICAL,
        message="Hardcoded credential detected.",
        suggestion="Use environment variables.",
        source="static",
    )


@pytest.fixture()
def sample_test_suggestion() -> TestSuggestion:
    return TestSuggestion(
        file="src/auth/service.py",
        symbol_name="authenticate",
        symbol_type="function",
        test_stub=(
            "import pytest\n"
            "from auth.service import authenticate\n\n"
            "def test_authenticate_valid_token():\n"
            "    # TODO: implement\n"
            "    pass\n"
        ),
    )


@pytest.fixture()
def sample_pr_summary() -> PRSummary:
    return PRSummary(
        purpose="Adds JWT-based user authentication to the API.",
        risk_level=RiskLevel.HIGH,
        key_changes=["Added authenticate() function", "Removed old_func()"],
        focus_areas=["Security of token handling", "Input validation"],
        breaking_changes=False,
    )


@pytest.fixture()
def sample_agent_state(
    sample_pr_metadata: PRMetadata,
    sample_python_file_change: FileChange,
    sample_code_issue: CodeIssue,
    sample_test_suggestion: TestSuggestion,
    sample_pr_summary: PRSummary,
) -> AgentState:
    return AgentState(
        pr_number=42,
        repo="owner/repo",
        run_id="test-run-id",
        pr_metadata=sample_pr_metadata,
        file_changes=[sample_python_file_change],
        code_issues=[sample_code_issue],
        test_suggestions=[sample_test_suggestion],
        summary=sample_pr_summary,
        stats=PipelineStats(
            total_tokens_used=1500,
            llm_calls=3,
            files_reviewed=1,
            issues_found=1,
            test_suggestions_count=1,
            duration_seconds=4.2,
        ),
        error=None,
        metadata={},
    )
