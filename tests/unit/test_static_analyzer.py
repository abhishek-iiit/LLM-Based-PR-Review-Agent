"""Tests for the static analysis engine."""

from __future__ import annotations

import pytest

from pr_review_agent.models.state import FileChange, FileStatus, IssueType, Severity
from pr_review_agent.tools.static_analyzer import StaticAnalyzer


@pytest.fixture()
def analyzer() -> StaticAnalyzer:
    return StaticAnalyzer()


def _make_python_change(patch: str, filename: str = "app.py") -> FileChange:
    return FileChange(
        filename=filename,
        status=FileStatus.MODIFIED,
        additions=patch.count("\n+"),
        deletions=0,
        patch=patch,
        language="python",
    )


class TestStaticAnalyzer:
    def test_eval_detected_as_critical(self, analyzer: StaticAnalyzer) -> None:
        fc = _make_python_change("@@ -1,1 +1,2 @@\n+result = eval(user_input)\n")
        issues = analyzer.analyze_file(fc)
        assert any(
            i.severity == Severity.CRITICAL and i.issue_type == IssueType.SECURITY for i in issues
        )

    def test_hardcoded_secret_detected(self, analyzer: StaticAnalyzer) -> None:
        fc = _make_python_change("@@ -0,0 +1,1 @@\n+api_key = 'my-secret-key-here'\n")
        issues = analyzer.analyze_file(fc)
        assert any(i.issue_type == IssueType.SECURITY for i in issues)

    def test_pickle_loads_detected(self, analyzer: StaticAnalyzer) -> None:
        fc = _make_python_change("@@ -0,0 +1,1 @@\n+data = pickle.loads(payload)\n")
        issues = analyzer.analyze_file(fc)
        assert any(i.severity == Severity.HIGH for i in issues)

    def test_print_detected_as_low(self, analyzer: StaticAnalyzer) -> None:
        fc = _make_python_change("@@ -0,0 +1,1 @@\n+print('debug')\n")
        issues = analyzer.analyze_file(fc)
        assert any(i.severity == Severity.LOW for i in issues)

    def test_bare_except_detected(self, analyzer: StaticAnalyzer) -> None:
        fc = _make_python_change("@@ -0,0 +1,2 @@\n+try:\n+    pass\n+except:\n+    pass\n")
        issues = analyzer.analyze_file(fc)
        assert any(i.issue_type == IssueType.BUG for i in issues)

    def test_todo_comment_detected(self, analyzer: StaticAnalyzer) -> None:
        fc = _make_python_change("@@ -0,0 +1,1 @@\n+# TODO: fix this later\n")
        issues = analyzer.analyze_file(fc)
        assert any(i.severity == Severity.LOW for i in issues)

    def test_safe_code_not_flagged(self, analyzer: StaticAnalyzer) -> None:
        fc = _make_python_change(
            "@@ -0,0 +1,5 @@\n"
            "+import os\n"
            "+\n"
            "+def greet(name: str) -> str:\n"
            "+    return f'Hello, {name}'\n"
        )
        issues = analyzer.analyze_file(fc)
        assert issues == []

    def test_binary_file_skipped(self, analyzer: StaticAnalyzer) -> None:
        fc = FileChange(filename="image.png", status=FileStatus.ADDED, patch="", language="unknown")
        issues = analyzer.analyze_file(fc)
        assert issues == []

    def test_removed_lines_not_flagged(self, analyzer: StaticAnalyzer) -> None:
        """Issues on removed lines (starting with '-') should NOT be reported."""
        fc = _make_python_change(
            "@@ -1,2 +1,1 @@\n-result = eval(old_input)  # removed\n+result = safe_parse(input)\n"
        )
        issues = analyzer.analyze_file(fc)
        # Should not flag eval() on the removed line
        eval_issues = [i for i in issues if "eval" in i.message.lower()]
        assert eval_issues == []

    def test_language_scoping_js_rule_not_applied_to_python(self, analyzer: StaticAnalyzer) -> None:
        """console.log should only flag JS/TS files, not Python files."""
        fc = _make_python_change("@@ -0,0 +1,1 @@\n+console.log('test')\n")
        issues = analyzer.analyze_file(fc)
        console_issues = [i for i in issues if "console" in i.message.lower()]
        assert console_issues == []

    def test_console_log_detected_in_js(self, analyzer: StaticAnalyzer) -> None:
        fc = FileChange(
            filename="app.js",
            status=FileStatus.MODIFIED,
            additions=1,
            deletions=0,
            patch="@@ -0,0 +1,1 @@\n+console.log('debug')\n",
            language="javascript",
        )
        issues = analyzer.analyze_file(fc)
        assert any("console" in i.message.lower() for i in issues)

    def test_analyze_files_aggregates_all(self, analyzer: StaticAnalyzer) -> None:
        fc1 = _make_python_change("@@ -0,0 +1,1 @@\n+eval(x)\n", "a.py")
        fc2 = _make_python_change("@@ -0,0 +1,1 @@\n+print('hi')\n", "b.py")
        issues = analyzer.analyze_files([fc1, fc2])
        files_with_issues = {i.file for i in issues}
        assert "a.py" in files_with_issues
        assert "b.py" in files_with_issues

    def test_line_number_extracted_from_hunk(self, analyzer: StaticAnalyzer) -> None:
        fc = _make_python_change(
            "@@ -10,3 +10,4 @@\n existing_line\n+eval(user_data)\n another_line\n"
        )
        issues = analyzer.analyze_file(fc)
        eval_issues = [i for i in issues if "eval" in i.message.lower()]
        assert eval_issues
        assert eval_issues[0].line is not None
