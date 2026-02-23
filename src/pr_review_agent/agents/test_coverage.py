"""Test Coverage Agent — third node in the LangGraph pipeline.

Detects new functions and classes introduced in the PR that lack
corresponding test coverage, then generates pytest stubs via LLM.
"""

from __future__ import annotations

import ast
import re
import time
from typing import Any

from pr_review_agent.models.state import (
    AgentState,
    FileChange,
    PipelineStats,
    TestSuggestion,
)
from pr_review_agent.services.llm_service import LLMService
from pr_review_agent.utils.logging import bind_agent_context, get_logger

log = get_logger(__name__)

AGENT_NAME = "test_coverage"

# ── Prompt ────────────────────────────────────────────────────────────────────

_STUB_PROMPT = """You are a senior Python engineer writing pytest unit tests.

Generate a pytest test stub for the following {symbol_type} `{symbol_name}` from `{filename}`.

Source context (relevant added lines):
```python
{source_context}
```

Requirements:
- Use pytest conventions
- Include at least one happy-path test and one edge-case test
- Use `pytest.fixture` for setup where appropriate
- Mock external dependencies with `pytest.mock.patch` or `unittest.mock.MagicMock`
- Add descriptive docstrings to each test function
- Do NOT implement the actual test logic — use `# TODO: implement` placeholders
- Return ONLY the Python code, no explanations

Test file output:
"""


# ── Symbol extraction ─────────────────────────────────────────────────────────


def _extract_added_source(patch: str) -> str:
    """Extract only the added lines from a patch (without the '+' prefix)."""
    lines = []
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            lines.append(line[1:])
    return "\n".join(lines)


def _extract_python_symbols(source: str) -> list[tuple[str, str]]:
    """Parse Python source and return (name, type) for top-level defs.

    Uses AST parsing for accuracy; falls back to regex if syntax errors occur.

    Args:
        source: Python source code string.

    Returns:
        List of (symbol_name, symbol_type) tuples where type is 'function' or 'class'.
    """
    symbols: list[tuple[str, str]] = []

    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                # Skip private/dunder methods and test functions
                if not node.name.startswith("_") and not node.name.startswith("test_"):
                    symbols.append((node.name, "function"))
            elif isinstance(node, ast.ClassDef):
                if not node.name.startswith("_"):
                    symbols.append((node.name, "class"))
    except SyntaxError:
        # Fallback: regex extraction for partial/invalid Python
        for match in re.finditer(r"^(?:async\s+)?def\s+([a-zA-Z][a-zA-Z0-9_]*)", source, re.M):
            name = match.group(1)
            if not name.startswith("test_"):
                symbols.append((name, "function"))
        for match in re.finditer(r"^class\s+([a-zA-Z][a-zA-Z0-9_]*)", source, re.M):
            symbols.append((match.group(1), "class"))

    return symbols


# ── Agent ─────────────────────────────────────────────────────────────────────


class TestCoverageAgent:
    """LangGraph node that identifies untested code and generates pytest stubs.

    Args:
        llm_service: LLM service instance. Created from settings if not provided.
    """

    def __init__(self, llm_service: LLMService | None = None) -> None:
        self._llm = llm_service or LLMService()

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """Detect untested symbols and generate test stubs.

        Args:
            state: Pipeline state with file_changes.

        Returns:
            Partial state update with test_suggestions and updated stats.
        """
        bind_agent_context(AGENT_NAME)
        start = time.perf_counter()

        file_changes: list[FileChange] = state.get("file_changes", [])

        # Find all test files being added/modified in this PR
        test_filenames: set[str] = {fc.filename for fc in file_changes if fc.is_test_file}

        suggestions: list[TestSuggestion] = []

        python_files = [
            fc
            for fc in file_changes
            if fc.language == "python" and not fc.is_test_file and fc.additions > 0
        ]

        log.info(
            "test coverage agent starting",
            python_files=len(python_files),
            test_files_in_pr=len(test_filenames),
        )

        for fc in python_files:
            added_source = _extract_added_source(fc.patch)
            if not added_source.strip():
                continue

            symbols = _extract_python_symbols(added_source)
            if not symbols:
                continue

            # Check if there's a corresponding test file in the PR
            expected_test_path = self._expected_test_path(fc.filename)
            has_test_coverage = expected_test_path in test_filenames

            if has_test_coverage:
                log.debug("test file found in PR", file=fc.filename, test=expected_test_path)
                continue

            log.debug(
                "no test coverage found",
                file=fc.filename,
                symbols=[s[0] for s in symbols],
            )

            for symbol_name, symbol_type in symbols:
                stub = self._generate_stub(
                    filename=fc.filename,
                    symbol_name=symbol_name,
                    symbol_type=symbol_type,
                    source_context=added_source[:2000],  # Limit context size
                )
                if stub:
                    suggestions.append(
                        TestSuggestion(
                            file=fc.filename,
                            symbol_name=symbol_name,
                            symbol_type=symbol_type,
                            test_stub=stub,
                        )
                    )

        elapsed = time.perf_counter() - start
        log.info(
            "test coverage agent complete",
            suggestions_generated=len(suggestions),
            elapsed=round(elapsed, 2),
        )

        existing_stats: PipelineStats = state.get("stats", PipelineStats())
        updated_stats = PipelineStats(
            total_tokens_used=existing_stats.total_tokens_used + self._llm.total_tokens.total,
            llm_calls=existing_stats.llm_calls + self._llm.call_count,
            files_reviewed=existing_stats.files_reviewed,
            issues_found=existing_stats.issues_found,
            test_suggestions_count=existing_stats.test_suggestions_count + len(suggestions),
            duration_seconds=existing_stats.duration_seconds + elapsed,
        )

        return {
            "test_suggestions": suggestions,
            "stats": updated_stats,
        }

    def _generate_stub(
        self,
        filename: str,
        symbol_name: str,
        symbol_type: str,
        source_context: str,
    ) -> str:
        """Use LLM to generate a pytest stub for a symbol.

        Returns the stub code string, or empty string on failure.
        """
        prompt = _STUB_PROMPT.format(
            symbol_type=symbol_type,
            symbol_name=symbol_name,
            filename=filename,
            source_context=source_context,
        )

        try:
            stub = self._llm.invoke_raw(prompt)
            # Strip markdown fences if LLM wrapped the code
            stub = re.sub(r"^```python\s*", "", stub.strip())
            stub = re.sub(r"\s*```$", "", stub.strip())
            return stub
        except Exception as exc:
            log.warning(
                "failed to generate test stub",
                file=filename,
                symbol=symbol_name,
                error=str(exc),
            )
            return ""

    @staticmethod
    def _expected_test_path(filename: str) -> str:
        """Derive the expected test file path for a source file.

        Examples:
            src/myapp/utils.py  ->  tests/unit/test_utils.py
            myapp/services.py   ->  tests/unit/test_services.py
        """
        # Strip src/ prefix if present
        cleaned = re.sub(r"^src/", "", filename)
        # Get just the filename
        base = cleaned.rsplit("/", 1)[-1]
        stem = base.rsplit(".", 1)[0]
        return f"tests/unit/test_{stem}.py"
