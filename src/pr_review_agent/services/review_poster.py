"""Review formatter and GitHub comment poster.

ReviewFormatter converts AgentState into a rich Markdown string.
ReviewPoster uses GitHubService to post it as a PR comment.
"""

from __future__ import annotations

from collections import defaultdict

from pr_review_agent.config.settings import Settings, get_settings
from pr_review_agent.models.state import (
    AgentState,
    CodeIssue,
    PipelineStats,
    PRMetadata,
    PRSummary,
    RiskLevel,
    Severity,
    TestSuggestion,
)
from pr_review_agent.services.github_service import GitHubService
from pr_review_agent.utils.logging import get_logger

log = get_logger(__name__)

# ── Emoji / badge maps ────────────────────────────────────────────────────────

_SEVERITY_EMOJI: dict[Severity, str] = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH: "🟠",
    Severity.MEDIUM: "🟡",
    Severity.LOW: "🟢",
    Severity.INFO: "🔵",
}

_RISK_BADGE: dict[RiskLevel, str] = {
    RiskLevel.LOW: "![risk: low](https://img.shields.io/badge/risk-low-brightgreen)",
    RiskLevel.MEDIUM: "![risk: medium](https://img.shields.io/badge/risk-medium-yellow)",
    RiskLevel.HIGH: "![risk: high](https://img.shields.io/badge/risk-high-orange)",
    RiskLevel.CRITICAL: "![risk: critical](https://img.shields.io/badge/risk-critical-red)",
}


# ── Formatter ─────────────────────────────────────────────────────────────────


class ReviewFormatter:
    """Converts pipeline AgentState into a GitHub-flavoured Markdown comment."""

    def format_full_review(self, state: AgentState) -> str:
        """Build the complete Markdown review comment.

        Args:
            state: Final AgentState after all agents have run.

        Returns:
            Markdown string ready to post as a GitHub PR comment.
        """
        sections: list[str] = []

        sections.append(self._header(state))
        sections.append(self._summary_section(state.get("summary")))
        sections.append(self._issues_section(state.get("code_issues", [])))
        sections.append(self._test_suggestions_section(state.get("test_suggestions", [])))
        sections.append(self._footer(state.get("stats")))

        return "\n\n---\n\n".join(filter(None, sections))

    # ── Section builders ──────────────────────────────────────────────────────

    def _header(self, state: AgentState) -> str:
        metadata: PRMetadata | None = state.get("pr_metadata")
        summary: PRSummary | None = state.get("summary")

        title = metadata.title if metadata else f"PR #{state.get('pr_number', '?')}"
        author = f"@{metadata.author}" if metadata else "unknown"
        risk_badge = (
            _RISK_BADGE.get(summary.risk_level, "") if summary else ""
        )

        lines = [
            "# 🤖 Automated PR Review",
            f"**{title}** by {author}  {risk_badge}",
        ]
        if summary and summary.breaking_changes:
            lines.append("\n> ⚠️ **This PR may contain breaking changes.**")

        return "\n".join(lines)

    def _summary_section(self, summary: PRSummary | None) -> str:
        if not summary:
            return "## 📋 Summary\n\n_Summary not available._"

        lines = [
            "## 📋 Summary",
            "",
            summary.purpose,
        ]

        if summary.key_changes:
            lines += ["", "**Key changes:**"]
            lines += [f"- {c}" for c in summary.key_changes]

        if summary.focus_areas:
            lines += ["", "**Reviewer focus areas:**"]
            lines += [f"- {a}" for a in summary.focus_areas]

        return "\n".join(lines)

    def _issues_section(self, issues: list[CodeIssue]) -> str:
        if not issues:
            return "## ✅ Code Issues\n\n_No issues found._"

        # Group by file
        by_file: dict[str, list[CodeIssue]] = defaultdict(list)
        for issue in issues:
            by_file[issue.file].append(issue)

        # Sort files by worst severity
        def _worst_severity(file_issues: list[CodeIssue]) -> int:
            severities = list(Severity)
            return min(severities.index(i.severity) for i in file_issues)

        sorted_files = sorted(by_file.keys(), key=lambda f: _worst_severity(by_file[f]))

        # Severity summary line
        counts: dict[Severity, int] = defaultdict(int)
        for issue in issues:
            counts[issue.severity] += 1

        summary_parts = [
            f"{_SEVERITY_EMOJI[s]} {counts[s]} {s.value}"
            for s in Severity
            if counts[s] > 0
        ]
        summary_line = " · ".join(summary_parts)

        lines = [
            f"## 🔍 Code Issues ({len(issues)} total)",
            "",
            summary_line,
        ]

        for filename in sorted_files:
            file_issues = sorted(
                by_file[filename],
                key=lambda i: list(Severity).index(i.severity),
            )
            lines.append(f"\n### `{filename}`\n")
            lines.append("| Severity | Line | Type | Issue | Suggestion |")
            lines.append("|----------|------|------|-------|------------|")
            for issue in file_issues:
                emoji = _SEVERITY_EMOJI.get(issue.severity, "⚪")
                line_ref = str(issue.line) if issue.line else "—"
                # Escape pipes in message/suggestion for markdown table
                msg = issue.message.replace("|", "\\|")
                sug = issue.suggestion.replace("|", "\\|") if issue.suggestion else "—"
                lines.append(
                    f"| {emoji} {issue.severity.value} | {line_ref} "
                    f"| {issue.issue_type.value} | {msg} | {sug} |"
                )

        return "\n".join(lines)

    def _test_suggestions_section(self, suggestions: list[TestSuggestion]) -> str:
        if not suggestions:
            return "## ✅ Test Coverage\n\n_No test gaps detected._"

        lines = [
            f"## 🧪 Test Coverage Gaps ({len(suggestions)} untested symbol(s))",
            "",
            "_The following new functions/classes appear to lack test coverage. "
            "Generated pytest stubs are provided below._",
        ]

        for suggestion in suggestions:
            stub_display = suggestion.test_stub or "# No stub generated"
            lines.append(
                f"\n<details>\n"
                f"<summary><code>{suggestion.file}</code> — "
                f"<code>{suggestion.symbol_name}</code> ({suggestion.symbol_type})</summary>\n\n"
                f"```python\n{stub_display}\n```\n\n"
                f"</details>"
            )

        return "\n".join(lines)

    def _footer(self, stats: PipelineStats | None) -> str:
        if not stats:
            return "_Review generated by [PR Review Agent](https://github.com)_"

        return (
            f"---\n"
            f"*🤖 Generated by **PR Review Agent** · "
            f"{stats.files_reviewed} files · "
            f"{stats.issues_found} issues · "
            f"{stats.test_suggestions_count} test gaps · "
            f"{stats.total_tokens_used:,} tokens · "
            f"{stats.duration_seconds:.1f}s*"
        )

    def truncate(self, text: str, max_chars: int) -> str:
        """Truncate Markdown text to GitHub's comment character limit.

        Appends a note when truncation occurs.

        Args:
            text: Full Markdown string.
            max_chars: Maximum allowed character count.

        Returns:
            Possibly truncated Markdown string.
        """
        if len(text) <= max_chars:
            return text

        truncation_note = (
            "\n\n---\n"
            "_⚠️ This review was truncated due to GitHub's comment size limit. "
            "Run the review locally to see the full output._"
        )
        limit = max_chars - len(truncation_note)
        return text[:limit] + truncation_note


# ── Poster ────────────────────────────────────────────────────────────────────


class ReviewPoster:
    """Posts a formatted review comment to a GitHub PR.

    Args:
        github_service: GitHub service instance.
        settings: Application settings.
        formatter: ReviewFormatter instance (uses default if not provided).
    """

    def __init__(
        self,
        github_service: GitHubService | None = None,
        settings: Settings | None = None,
        formatter: ReviewFormatter | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._github = github_service or GitHubService()
        self._formatter = formatter or ReviewFormatter()

    def post_review(self, state: AgentState, repo: str, pr_number: int) -> str:
        """Format and post the complete review to GitHub.

        Args:
            state: Final pipeline AgentState.
            repo: Full repository name.
            pr_number: PR number.

        Returns:
            The Markdown comment body that was posted.
        """
        body = self._formatter.format_full_review(state)
        body = self._formatter.truncate(body, self._settings.github_comment_max_chars)

        log.info(
            "posting review to GitHub",
            repo=repo,
            pr_number=pr_number,
            comment_length=len(body),
        )

        self._github.post_pr_review_comment(repo=repo, pr_number=pr_number, body=body)
        log.info("review posted successfully")
        return body
