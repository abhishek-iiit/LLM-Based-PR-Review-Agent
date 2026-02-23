"""GitHub API service.

Wraps PyGithub to provide a clean interface for fetching PR data and
posting review comments. All public methods retry on transient errors.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

from github import Auth, Github, GithubException, RateLimitExceededException
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pr_review_agent.config.settings import Settings, get_settings
from pr_review_agent.models.state import FileChange, FileStatus, PRMetadata
from pr_review_agent.utils.logging import get_logger

if TYPE_CHECKING:
    from github.PullRequest import PullRequest
    from github.Repository import Repository

log = get_logger(__name__)

# Language detection by file extension
_EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".rs": "rust",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".c": "c",
    ".sh": "shell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".md": "markdown",
    ".sql": "sql",
    ".tf": "terraform",
    ".dockerfile": "docker",
}


def _detect_language(filename: str) -> str:
    """Detect language from file extension."""
    if filename.lower() == "dockerfile":
        return "docker"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _EXT_TO_LANGUAGE.get(suffix, "unknown")


class GitHubService:
    """Service for all GitHub API interactions.

    Args:
        settings: Application settings. Defaults to the global singleton.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        auth = Auth.Token(self._settings.github_token)
        self._client = Github(auth=auth)

    # ── Internal retry decorator ───────────────────────────────────────────────

    def _make_retry(self) -> Any:
        """Build a tenacity retry decorator using settings."""
        return retry(
            retry=retry_if_exception_type((RateLimitExceededException, IOError)),
            stop=stop_after_attempt(self._settings.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            reraise=True,
        )

    def _get_repo(self, repo_name: str) -> Repository:
        """Fetch Repository object, reusing the authenticated client."""
        return self._client.get_repo(repo_name)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_pr_metadata(self, repo: str, pr_number: int) -> PRMetadata:
        """Fetch core metadata for a pull request.

        Args:
            repo: Full repository name, e.g. 'owner/repo'.
            pr_number: Pull request number.

        Returns:
            Populated PRMetadata model.

        Raises:
            GithubException: On 404 / 403 from the API.
        """
        log.info("fetching PR metadata", repo=repo, pr_number=pr_number)
        start = time.perf_counter()

        @self._make_retry()  # type: ignore[untyped-decorator]
        def _fetch() -> PullRequest:
            return self._get_repo(repo).get_pull(pr_number)

        try:
            pr = _fetch()
        except GithubException as exc:
            log.error(
                "failed to fetch PR metadata",
                repo=repo,
                pr_number=pr_number,
                status=exc.status,
                data=exc.data,
            )
            raise

        metadata = PRMetadata(
            pr_number=pr.number,
            repo=repo,
            title=pr.title,
            author=pr.user.login,
            base_branch=pr.base.ref,
            head_branch=pr.head.ref,
            url=pr.html_url,
            body=pr.body or "",
            additions=pr.additions,
            deletions=pr.deletions,
            changed_files=pr.changed_files,
        )

        log.info(
            "PR metadata fetched",
            title=metadata.title,
            author=metadata.author,
            changed_files=metadata.changed_files,
            elapsed=round(time.perf_counter() - start, 2),
        )
        return metadata

    def get_pr_files(self, repo: str, pr_number: int) -> list[FileChange]:
        """Fetch all changed files with diffs for a PR.

        Uses PyGithub's paginated list to handle PRs with many files.

        Args:
            repo: Full repository name.
            pr_number: Pull request number.

        Returns:
            List of FileChange models, one per changed file.
        """
        log.info("fetching PR files", repo=repo, pr_number=pr_number)
        start = time.perf_counter()

        @self._make_retry()  # type: ignore[untyped-decorator]
        def _fetch() -> list[FileChange]:
            gh_repo = self._get_repo(repo)
            pr = gh_repo.get_pull(pr_number)
            files: list[FileChange] = []

            for gh_file in pr.get_files():
                try:
                    status = FileStatus(gh_file.status)
                except ValueError:
                    status = FileStatus.CHANGED

                files.append(
                    FileChange(
                        filename=gh_file.filename,
                        status=status,
                        additions=gh_file.additions,
                        deletions=gh_file.deletions,
                        patch=gh_file.patch or "",
                        language=_detect_language(gh_file.filename),
                    )
                )

            return files

        files = cast(list[FileChange], _fetch())

        log.info(
            "PR files fetched",
            file_count=len(files),
            elapsed=round(time.perf_counter() - start, 2),
        )
        return files

    def post_pr_review_comment(self, repo: str, pr_number: int, body: str) -> None:
        """Post a top-level review comment on a PR.

        Args:
            repo: Full repository name.
            pr_number: Pull request number.
            body: Markdown comment body (max 65535 chars).
        """
        log.info(
            "posting review comment",
            repo=repo,
            pr_number=pr_number,
            body_length=len(body),
        )

        @self._make_retry()  # type: ignore[untyped-decorator]
        def _post() -> None:
            gh_repo = self._get_repo(repo)
            pr = gh_repo.get_pull(pr_number)
            pr.create_issue_comment(body)

        try:
            _post()
            log.info("review comment posted successfully")
        except GithubException as exc:
            log.error(
                "failed to post review comment",
                repo=repo,
                pr_number=pr_number,
                status=exc.status,
                data=exc.data,
            )
            raise

    def post_inline_comment(
        self,
        repo: str,
        pr_number: int,
        commit_sha: str,
        filename: str,
        line: int,
        body: str,
    ) -> None:
        """Post an inline review comment on a specific line in a file.

        Args:
            repo: Full repository name.
            pr_number: Pull request number.
            commit_sha: Head commit SHA (required by GitHub API).
            filename: File path relative to repo root.
            line: Line number in the diff.
            body: Markdown comment body.
        """
        log.debug(
            "posting inline comment",
            repo=repo,
            pr_number=pr_number,
            filename=filename,
            line=line,
        )

        @self._make_retry()  # type: ignore[untyped-decorator]
        def _post() -> None:
            gh_repo = self._get_repo(repo)
            pr = gh_repo.get_pull(pr_number)
            commit = gh_repo.get_commit(commit_sha)
            pr.create_review_comment(
                body=body,
                commit=commit,
                path=filename,
                line=line,
            )

        try:
            _post()
        except GithubException as exc:
            # Inline comment failures are non-fatal; log and continue
            log.warning(
                "failed to post inline comment (non-fatal)",
                filename=filename,
                line=line,
                status=exc.status,
            )
