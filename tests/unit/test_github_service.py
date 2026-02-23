"""Tests for the GitHub API service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pr_review_agent.models.state import FileStatus
from pr_review_agent.services.github_service import GitHubService, _detect_language


# ── Language detection tests ──────────────────────────────────────────────────


class TestDetectLanguage:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("app.py", "python"),
            ("index.js", "javascript"),
            ("component.tsx", "typescript"),
            ("main.go", "go"),
            ("Service.java", "java"),
            ("Dockerfile", "docker"),
            ("config.yaml", "yaml"),
            ("config.yml", "yaml"),
            ("schema.sql", "sql"),
            ("main.rs", "rust"),
            ("unknown.xyz", "unknown"),
        ],
    )
    def test_language_detection(self, filename: str, expected: str) -> None:
        assert _detect_language(filename) == expected


# ── GitHubService tests ────────────────────────────────────────────────────────


@pytest.fixture()
def mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.github_token = "ghp_testtoken1234"
    settings.max_retries = 1
    return settings


@pytest.fixture()
def github_service(mock_settings: MagicMock) -> GitHubService:
    with patch("pr_review_agent.services.github_service.Github"), patch(
        "pr_review_agent.services.github_service.Auth"
    ):
        svc = GitHubService(settings=mock_settings)
        return svc


class TestGetPRMetadata:
    def test_returns_pr_metadata(self, github_service: GitHubService) -> None:
        mock_pr = MagicMock()
        mock_pr.number = 42
        mock_pr.title = "Test PR"
        mock_pr.user.login = "alice"
        mock_pr.base.ref = "main"
        mock_pr.head.ref = "feature"
        mock_pr.html_url = "https://github.com/owner/repo/pull/42"
        mock_pr.body = "Description"
        mock_pr.additions = 100
        mock_pr.deletions = 10
        mock_pr.changed_files = 3

        github_service._client.get_repo.return_value.get_pull.return_value = mock_pr

        metadata = github_service.get_pr_metadata("owner/repo", 42)

        assert metadata.pr_number == 42
        assert metadata.title == "Test PR"
        assert metadata.author == "alice"
        assert metadata.additions == 100


class TestGetPRFiles:
    def test_returns_file_changes(self, github_service: GitHubService) -> None:
        mock_file = MagicMock()
        mock_file.filename = "src/app.py"
        mock_file.status = "modified"
        mock_file.additions = 5
        mock_file.deletions = 2
        mock_file.patch = "@@ -1,2 +1,5 @@\n+new line\n"

        mock_pr = MagicMock()
        mock_pr.get_files.return_value = [mock_file]
        github_service._client.get_repo.return_value.get_pull.return_value = mock_pr

        files = github_service.get_pr_files("owner/repo", 42)

        assert len(files) == 1
        assert files[0].filename == "src/app.py"
        assert files[0].status == FileStatus.MODIFIED
        assert files[0].language == "python"

    def test_binary_file_has_empty_patch(self, github_service: GitHubService) -> None:
        mock_file = MagicMock()
        mock_file.filename = "assets/logo.png"
        mock_file.status = "added"
        mock_file.additions = 0
        mock_file.deletions = 0
        mock_file.patch = None  # GitHub returns None for binary files

        mock_pr = MagicMock()
        mock_pr.get_files.return_value = [mock_file]
        github_service._client.get_repo.return_value.get_pull.return_value = mock_pr

        files = github_service.get_pr_files("owner/repo", 42)
        assert files[0].patch == ""
        assert files[0].is_binary is True

    def test_unknown_status_mapped_to_changed(self, github_service: GitHubService) -> None:
        mock_file = MagicMock()
        mock_file.filename = "foo.py"
        mock_file.status = "some_unknown_status"
        mock_file.additions = 1
        mock_file.deletions = 0
        mock_file.patch = "+x = 1\n"

        mock_pr = MagicMock()
        mock_pr.get_files.return_value = [mock_file]
        github_service._client.get_repo.return_value.get_pull.return_value = mock_pr

        files = github_service.get_pr_files("owner/repo", 42)
        assert files[0].status == FileStatus.CHANGED


class TestPostPRReviewComment:
    def test_posts_comment(self, github_service: GitHubService) -> None:
        mock_pr = MagicMock()
        github_service._client.get_repo.return_value.get_pull.return_value = mock_pr

        github_service.post_pr_review_comment("owner/repo", 42, "Great PR!")
        mock_pr.create_issue_comment.assert_called_once_with("Great PR!")
