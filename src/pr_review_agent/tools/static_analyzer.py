"""Static analysis engine — rule-based code scanner.

Detects code issues without calling an LLM, providing fast, deterministic
feedback that complements the AI-powered review. Rules are organised by
language and annotated with severity and issue type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pr_review_agent.models.state import CodeIssue, FileChange, IssueType, Severity
from pr_review_agent.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Rule:
    """A single static analysis rule.

    Attributes:
        pattern: Regex pattern matched against each diff line (added lines only).
        severity: Issue severity.
        issue_type: Category of the issue.
        message: Human-readable description.
        suggestion: Recommended fix.
        languages: Languages this rule applies to. Empty = all languages.
    """

    pattern: str
    severity: Severity
    issue_type: IssueType
    message: str
    suggestion: str
    languages: tuple[str, ...] = field(default_factory=tuple)


# ── Rule Registry ─────────────────────────────────────────────────────────────

_RULES: list[Rule] = [
    # Security
    Rule(
        pattern=r"\beval\s*\(",
        severity=Severity.CRITICAL,
        issue_type=IssueType.SECURITY,
        message="Use of `eval()` detected — arbitrary code execution risk.",
        suggestion="Replace `eval()` with safer alternatives (ast.literal_eval, "
        "dedicated parsing, etc.).",
        languages=("python",),
    ),
    Rule(
        pattern=r"\bexec\s*\(",
        severity=Severity.CRITICAL,
        issue_type=IssueType.SECURITY,
        message="Use of `exec()` detected — arbitrary code execution risk.",
        suggestion="Avoid `exec()`. If dynamic code is necessary, use subprocess with "
        "strict argument validation.",
        languages=("python",),
    ),
    Rule(
        pattern=r'(?i)(password|secret|api_key|apikey|token|auth_token)\s*=\s*["\'][^"\']{4,}["\']',
        severity=Severity.CRITICAL,
        issue_type=IssueType.SECURITY,
        message="Hardcoded credential detected.",
        suggestion="Use environment variables or a secrets manager. "
        "Remove the credential and rotate it immediately.",
    ),
    Rule(
        pattern=r"\bSUBPROCESS_SHELL\b|shell\s*=\s*True",
        severity=Severity.HIGH,
        issue_type=IssueType.SECURITY,
        message="`shell=True` in subprocess call — potential command injection.",
        suggestion="Pass arguments as a list instead of a string, and set shell=False.",
        languages=("python",),
    ),
    Rule(
        pattern=r"\bpickle\.loads?\s*\(",
        severity=Severity.HIGH,
        issue_type=IssueType.SECURITY,
        message="`pickle.load()` on untrusted data can execute arbitrary code.",
        suggestion="Use JSON, msgpack, or other safe serialisation formats for "
        "untrusted input.",
        languages=("python",),
    ),
    Rule(
        pattern=r"\.format\s*\(.*request\.|f\".*\{request\.",
        severity=Severity.HIGH,
        issue_type=IssueType.SECURITY,
        message="Potential SQL/template injection via unescaped user input in format string.",
        suggestion="Use parameterised queries or proper escaping.",
        languages=("python",),
    ),
    # Bugs & Code Quality
    Rule(
        pattern=r"\bprint\s*\(",
        severity=Severity.LOW,
        issue_type=IssueType.MAINTAINABILITY,
        message="Debug `print()` statement found in non-test code.",
        suggestion="Replace with structured logging (e.g. `log.debug(...)`).",
        languages=("python",),
    ),
    Rule(
        pattern=r"\bexcept\s*:",
        severity=Severity.MEDIUM,
        issue_type=IssueType.BUG,
        message="Bare `except:` clause catches ALL exceptions including SystemExit.",
        suggestion="Catch specific exception types (e.g. `except ValueError:`).",
        languages=("python",),
    ),
    Rule(
        pattern=r"#\s*TODO",
        severity=Severity.LOW,
        issue_type=IssueType.MAINTAINABILITY,
        message="TODO comment left in code.",
        suggestion="Resolve before merging or create a tracked issue.",
    ),
    Rule(
        pattern=r"#\s*FIXME",
        severity=Severity.MEDIUM,
        issue_type=IssueType.BUG,
        message="FIXME comment indicates a known bug left unresolved.",
        suggestion="Fix the issue before merging.",
    ),
    Rule(
        pattern=r"#\s*HACK|#\s*XXX",
        severity=Severity.MEDIUM,
        issue_type=IssueType.MAINTAINABILITY,
        message="HACK/XXX comment indicates a workaround or problematic code.",
        suggestion="Refactor the code and remove the comment.",
    ),
    # Performance
    Rule(
        pattern=r"\.objects\.all\(\)\s*$",
        severity=Severity.MEDIUM,
        issue_type=IssueType.PERFORMANCE,
        message="Unfiltered `.objects.all()` query — potential full table scan.",
        suggestion="Add `.filter()` or pagination to avoid loading all records.",
        languages=("python",),
    ),
    Rule(
        pattern=r"time\.sleep\s*\(",
        severity=Severity.MEDIUM,
        issue_type=IssueType.PERFORMANCE,
        message="`time.sleep()` blocks the event loop in async code.",
        suggestion="Use `asyncio.sleep()` in async contexts.",
        languages=("python",),
    ),
    # JavaScript / TypeScript
    Rule(
        pattern=r"\bconsole\.(log|debug|warn|error)\s*\(",
        severity=Severity.LOW,
        issue_type=IssueType.MAINTAINABILITY,
        message="Debug `console.*()` statement found.",
        suggestion="Remove or replace with a proper logger.",
        languages=("javascript", "typescript"),
    ),
    Rule(
        pattern=r"\bvar\s+\w+",
        severity=Severity.LOW,
        issue_type=IssueType.STYLE,
        message="Use of `var` instead of `const` or `let`.",
        suggestion="Replace `var` with `const` (preferred) or `let`.",
        languages=("javascript", "typescript"),
    ),
    Rule(
        pattern=r"dangerouslySetInnerHTML",
        severity=Severity.HIGH,
        issue_type=IssueType.SECURITY,
        message="`dangerouslySetInnerHTML` can lead to XSS vulnerabilities.",
        suggestion="Sanitise HTML or avoid rendering raw HTML from untrusted sources.",
        languages=("javascript", "typescript"),
    ),
]


# ── Analyser ──────────────────────────────────────────────────────────────────


class StaticAnalyzer:
    """Run rule-based static analysis on file diffs.

    Only analyses **added** lines (lines starting with '+' in the diff),
    to avoid flagging pre-existing issues not introduced by this PR.
    """

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self._rules = rules if rules is not None else _RULES

    def analyze_file(self, file_change: FileChange) -> list[CodeIssue]:
        """Analyse a single changed file and return detected issues.

        Args:
            file_change: The changed file including its diff patch.

        Returns:
            List of CodeIssue instances for detected problems.
        """
        if file_change.is_binary or not file_change.patch:
            return []

        issues: list[CodeIssue] = []
        language = file_change.language
        added_lines = self._extract_added_lines(file_change.patch)

        for line_number, line_content in added_lines:
            for rule in self._rules:
                # Skip rules that don't apply to this language
                if rule.languages and language not in rule.languages:
                    continue

                if re.search(rule.pattern, line_content):
                    issues.append(
                        CodeIssue(
                            file=file_change.filename,
                            line=line_number,
                            issue_type=rule.issue_type,
                            severity=rule.severity,
                            message=rule.message,
                            suggestion=rule.suggestion,
                            source="static",
                        )
                    )
                    # One issue per line per rule — don't double-report
                    break

        if issues:
            log.debug(
                "static analysis found issues",
                file=file_change.filename,
                issue_count=len(issues),
            )

        return issues

    def analyze_files(self, file_changes: list[FileChange]) -> list[CodeIssue]:
        """Analyse all files in a PR and return all detected issues.

        Args:
            file_changes: List of changed files.

        Returns:
            Aggregated list of CodeIssue instances across all files.
        """
        all_issues: list[CodeIssue] = []
        for fc in file_changes:
            all_issues.extend(self.analyze_file(fc))

        log.info(
            "static analysis complete",
            files_analyzed=len(file_changes),
            total_issues=len(all_issues),
        )
        return all_issues

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_added_lines(patch: str) -> list[tuple[int, str]]:
        """Extract added lines with their (approximate) line numbers from a diff patch.

        Returns:
            List of (line_number, line_content) tuples for added lines.
        """
        added: list[tuple[int, str]] = []
        current_line = 0

        for raw_line in patch.splitlines():
            # Parse hunk headers like @@ -10,5 +10,7 @@
            hunk_match = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw_line)
            if hunk_match:
                current_line = int(hunk_match.group(1)) - 1
                continue

            if raw_line.startswith("+") and not raw_line.startswith("+++"):
                current_line += 1
                added.append((current_line, raw_line[1:]))  # Strip leading '+'
            elif raw_line.startswith("-"):
                pass  # Removed lines don't advance the new-file line counter
            else:
                current_line += 1

        return added
