"""Doc Summarizer Agent — fourth node in the LangGraph pipeline.

Aggregates data from all previous agents and uses the LLM to produce a
structured, executive-level PR summary with risk assessment.
"""

from __future__ import annotations

import time
from typing import Any

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
from pr_review_agent.services.llm_service import LLMService
from pr_review_agent.utils.logging import bind_agent_context, get_logger

log = get_logger(__name__)

AGENT_NAME = "doc_summarizer"

# ── Prompt ────────────────────────────────────────────────────────────────────

_SUMMARY_PROMPT = """You are a senior engineering lead reviewing a pull request.
Produce a concise, actionable summary based on the information below.

## PR Details
- Title: {title}
- Author: {author}
- Base branch: {base_branch}
- Files changed: {changed_files} ({additions} additions, {deletions} deletions)
- PR Description: {body}

## Issues Found
Total: {total_issues}
- Critical: {critical_count}
- High: {high_count}
- Medium: {medium_count}
- Low: {low_count}

Issue samples:
{issue_samples}

## Test Gaps
{test_gap_count} new functions/classes lack test coverage.

## Changed Files
{file_list}

Produce a JSON summary with this exact schema:
{{
  "purpose": "<1-2 sentence description of what this PR accomplishes>",
  "risk_level": "<critical|high|medium|low>",
  "key_changes": ["<change 1>", "<change 2>", "..."],
  "focus_areas": ["<area reviewers should focus on>", "..."],
  "breaking_changes": <true|false>
}}

Risk level guide:
- critical: Security vulnerabilities, data loss risk, or production-breaking bugs
- high: Significant bugs, major logic changes, or missing critical tests
- medium: Moderate complexity, some issues, needs careful review
- low: Small, well-tested changes with minor issues
"""


# ── Agent ─────────────────────────────────────────────────────────────────────


class DocSummarizerAgent:
    """LangGraph node that generates a structured PR summary using LLM.

    Args:
        llm_service: LLM service instance. Created from settings if not provided.
    """

    def __init__(self, llm_service: LLMService | None = None) -> None:
        self._llm = llm_service or LLMService()

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """Generate the PR summary.

        Args:
            state: Pipeline state with all previous agent outputs.

        Returns:
            Partial state update with summary and final stats.
        """
        bind_agent_context(AGENT_NAME)
        start = time.perf_counter()

        metadata: PRMetadata | None = state.get("pr_metadata")
        issues: list[CodeIssue] = state.get("code_issues", [])
        test_suggestions: list[TestSuggestion] = state.get("test_suggestions", [])

        log.info(
            "doc summarizer starting",
            has_metadata=metadata is not None,
            issue_count=len(issues),
            test_suggestions=len(test_suggestions),
        )

        summary = self._generate_summary(metadata, issues, test_suggestions, state)

        elapsed = time.perf_counter() - start
        log.info(
            "doc summarizer complete",
            risk_level=summary.risk_level,
            key_changes_count=len(summary.key_changes),
            elapsed=round(elapsed, 2),
        )

        existing_stats: PipelineStats = state.get("stats", PipelineStats())
        updated_stats = PipelineStats(
            total_tokens_used=existing_stats.total_tokens_used
            + self._llm.total_tokens.total,
            llm_calls=existing_stats.llm_calls + self._llm.call_count,
            files_reviewed=existing_stats.files_reviewed,
            issues_found=existing_stats.issues_found,
            test_suggestions_count=existing_stats.test_suggestions_count,
            duration_seconds=existing_stats.duration_seconds + elapsed,
        )

        return {
            "summary": summary,
            "stats": updated_stats,
        }

    def _generate_summary(
        self,
        metadata: PRMetadata | None,
        issues: list[CodeIssue],
        test_suggestions: list[TestSuggestion],
        state: AgentState,
    ) -> PRSummary:
        """Build the LLM prompt and parse the structured summary."""
        if not metadata:
            return PRSummary(
                purpose="PR metadata unavailable.",
                risk_level=RiskLevel.MEDIUM,
                key_changes=[],
                focus_areas=["Metadata could not be fetched — review manually."],
                breaking_changes=False,
            )

        # Severity counts
        severity_counts: dict[Severity, int] = {s: 0 for s in Severity}
        for issue in issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1

        # Issue samples (top 5 by severity)
        sorted_issues = sorted(
            issues,
            key=lambda i: list(Severity).index(i.severity),
        )
        issue_samples = "\n".join(
            f"- [{i.severity.upper()}] {i.file}: {i.message}"
            for i in sorted_issues[:5]
        )
        if not issue_samples:
            issue_samples = "None"

        # File list
        file_changes = state.get("file_changes", [])
        file_list = "\n".join(
            f"- {fc.filename} ({fc.status}, +{fc.additions}/-{fc.deletions})"
            for fc in file_changes[:20]  # Cap at 20 files
        )
        if len(file_changes) > 20:
            file_list += f"\n... and {len(file_changes) - 20} more files"

        prompt = _SUMMARY_PROMPT.format(
            title=metadata.title,
            author=metadata.author,
            base_branch=metadata.base_branch,
            changed_files=metadata.changed_files,
            additions=metadata.additions,
            deletions=metadata.deletions,
            body=metadata.body[:500] if metadata.body else "No description provided.",
            total_issues=len(issues),
            critical_count=severity_counts.get(Severity.CRITICAL, 0),
            high_count=severity_counts.get(Severity.HIGH, 0),
            medium_count=severity_counts.get(Severity.MEDIUM, 0),
            low_count=severity_counts.get(Severity.LOW, 0),
            issue_samples=issue_samples,
            test_gap_count=len(test_suggestions),
            file_list=file_list or "No files changed.",
        )

        try:
            return self._llm.invoke_structured(prompt, PRSummary)
        except Exception as exc:
            log.warning("LLM summary failed, using fallback", error=str(exc))
            return self._fallback_summary(metadata, issues, test_suggestions)

    @staticmethod
    def _fallback_summary(
        metadata: PRMetadata,
        issues: list[CodeIssue],
        test_suggestions: list[TestSuggestion],
    ) -> PRSummary:
        """Produce a minimal summary without LLM when the call fails."""
        critical_or_high = [
            i for i in issues if i.severity in (Severity.CRITICAL, Severity.HIGH)
        ]
        risk = (
            RiskLevel.HIGH if critical_or_high else
            RiskLevel.MEDIUM if issues else
            RiskLevel.LOW
        )
        return PRSummary(
            purpose=f"PR '{metadata.title}' by @{metadata.author}.",
            risk_level=risk,
            key_changes=[f"Changed {metadata.changed_files} file(s)"],
            focus_areas=[
                f"{len(critical_or_high)} critical/high severity issues found"
            ] if critical_or_high else [],
            breaking_changes=False,
        )
