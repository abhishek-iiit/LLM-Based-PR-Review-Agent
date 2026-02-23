"""Code Review Agent — second node in the LangGraph pipeline.

Combines static rule-based analysis with Claude LLM semantic review.
Produces a list of CodeIssue objects covering bugs, security, style,
and performance concerns.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from pr_review_agent.config.settings import Settings, get_settings
from pr_review_agent.models.state import (
    AgentState,
    CodeIssue,
    FileChange,
    IssueType,
    PipelineStats,
    Severity,
)
from pr_review_agent.services.llm_service import LLMService
from pr_review_agent.tools.static_analyzer import StaticAnalyzer
from pr_review_agent.utils.logging import bind_agent_context, get_logger

log = get_logger(__name__)

AGENT_NAME = "code_reviewer"

# ── LLM response schema ───────────────────────────────────────────────────────


class _LLMIssue(BaseModel):
    """Schema for a single issue returned by the LLM."""

    line: int | None = Field(default=None)
    type: str = Field(default="bug")
    severity: str = Field(default="medium")
    message: str
    suggestion: str = Field(default="")


class _LLMReviewResponse(BaseModel):
    """Top-level schema for the LLM code review response."""

    issues: list[_LLMIssue] = Field(default_factory=list)


# ── Prompt templates ──────────────────────────────────────────────────────────

_REVIEW_PROMPT = """You are an expert code reviewer. Analyse the following file diff and identify issues.

File: {filename}
Language: {language}

Diff:
```
{diff}
```

Focus on:
1. **Bugs**: Logic errors, off-by-one errors, null dereferences, unclosed resources
2. **Security**: Injection vulnerabilities, insecure defaults, hardcoded secrets, unsafe deserialization
3. **Performance**: N+1 queries, blocking I/O in async code, unnecessary data loading
4. **Style/Maintainability**: Overly complex functions, missing error handling, misleading names

Rules:
- Only report issues in ADDED lines (lines starting with '+' in the diff)
- Do NOT report style preferences (formatting, naming conventions)
- Each issue must have a concrete, actionable suggestion
- If no issues are found, return an empty issues array

Respond with a JSON object matching this schema:
{{
  "issues": [
    {{
      "line": <integer or null>,
      "type": "<bug|security|performance|style|maintainability>",
      "severity": "<critical|high|medium|low|info>",
      "message": "<clear description of the problem>",
      "suggestion": "<specific recommendation to fix it>"
    }}
  ]
}}
"""


# ── Agent ─────────────────────────────────────────────────────────────────────


class CodeReviewAgent:
    """LangGraph node that reviews code using static analysis + LLM.

    Args:
        llm_service: LLM service instance. Created from settings if not provided.
        analyzer: Static analyzer instance. Uses default rules if not provided.
        settings: Application settings. Defaults to global singleton.
    """

    def __init__(
        self,
        llm_service: LLMService | None = None,
        analyzer: StaticAnalyzer | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = llm_service or LLMService()
        self._analyzer = analyzer or StaticAnalyzer()

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """Run code review on all changed files.

        Args:
            state: Pipeline state containing file_changes from PRFetcherAgent.

        Returns:
            Partial state update with code_issues and updated stats.
        """
        bind_agent_context(AGENT_NAME)
        start = time.perf_counter()

        file_changes: list[FileChange] = state.get("file_changes", [])
        log.info("code reviewer starting", file_count=len(file_changes))

        all_issues: list[CodeIssue] = []

        # Run static analysis across all files in bulk
        static_issues = self._analyzer.analyze_files(file_changes)
        all_issues.extend(static_issues)
        log.info("static analysis done", static_issue_count=len(static_issues))

        # LLM review per file (skip binary and very small diffs)
        llm_issues: list[CodeIssue] = []
        for fc in file_changes:
            if fc.is_binary or not fc.patch or fc.additions == 0:
                continue
            file_issues = self._review_file_with_llm(fc)
            llm_issues.extend(file_issues)

        all_issues.extend(llm_issues)
        log.info("LLM review done", llm_issue_count=len(llm_issues))

        elapsed = time.perf_counter() - start
        log.info(
            "code reviewer complete",
            total_issues=len(all_issues),
            elapsed=round(elapsed, 2),
        )

        # Update stats
        existing_stats: PipelineStats = state.get("stats", PipelineStats())
        updated_stats = PipelineStats(
            total_tokens_used=existing_stats.total_tokens_used + self._llm.total_tokens.total,
            llm_calls=existing_stats.llm_calls + self._llm.call_count,
            files_reviewed=existing_stats.files_reviewed,
            issues_found=existing_stats.issues_found + len(all_issues),
            test_suggestions_count=existing_stats.test_suggestions_count,
            duration_seconds=existing_stats.duration_seconds + elapsed,
        )

        return {
            "code_issues": all_issues,
            "stats": updated_stats,
        }

    def _review_file_with_llm(self, fc: FileChange) -> list[CodeIssue]:
        """Send a single file's diff to the LLM for semantic review.

        Truncates large diffs to stay within token limits.

        Args:
            fc: The changed file to review.

        Returns:
            List of CodeIssue instances from the LLM.
        """
        diff = fc.patch
        truncated = False

        if len(diff) > self._settings.max_diff_chars:
            diff = diff[: self._settings.max_diff_chars]
            truncated = True
            log.debug(
                "diff truncated for LLM",
                file=fc.filename,
                original_length=len(fc.patch),
                truncated_to=self._settings.max_diff_chars,
            )

        if truncated:
            diff += f"\n\n... [diff truncated at {self._settings.max_diff_chars} chars]"

        prompt = _REVIEW_PROMPT.format(
            filename=fc.filename,
            language=fc.language or "unknown",
            diff=diff,
        )

        try:
            response = self._llm.invoke_structured(prompt, _LLMReviewResponse)
        except Exception as exc:
            log.warning(
                "LLM review failed for file",
                file=fc.filename,
                error=str(exc),
            )
            return []

        issues: list[CodeIssue] = []
        for raw in response.issues:
            try:
                issue_type = IssueType(raw.type)
            except ValueError:
                issue_type = IssueType.BUG

            try:
                severity = Severity(raw.severity)
            except ValueError:
                severity = Severity.MEDIUM

            issues.append(
                CodeIssue(
                    file=fc.filename,
                    line=raw.line,
                    issue_type=issue_type,
                    severity=severity,
                    message=raw.message,
                    suggestion=raw.suggestion,
                    source="llm",
                )
            )

        log.debug(
            "LLM reviewed file",
            file=fc.filename,
            issue_count=len(issues),
        )
        return issues
