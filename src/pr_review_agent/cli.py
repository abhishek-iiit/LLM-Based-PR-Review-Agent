"""Command-line interface for the PR Review Agent.

Usage:
    pr-review review --repo owner/repo --pr 42
    pr-review review --repo owner/repo --pr 42 --no-post --output-file review.md
    pr-review validate-config
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from pr_review_agent.config.settings import Settings, get_settings
from pr_review_agent.utils.logging import configure_logging, get_logger

# ── Exit codes ────────────────────────────────────────────────────────────────
EXIT_SUCCESS = 0
EXIT_CONFIG_ERROR = 1
EXIT_GITHUB_ERROR = 2
EXIT_LLM_ERROR = 3
EXIT_PIPELINE_ERROR = 4


@click.group()
@click.version_option(version="0.1.0", prog_name="pr-review")
def cli() -> None:
    """Automated GitHub PR review agent powered by Gemini and LangGraph."""


# ── review command ────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--repo",
    required=True,
    help="Full repository name (e.g. owner/repo).",
    metavar="OWNER/REPO",
)
@click.option(
    "--pr",
    "pr_number",
    required=True,
    type=int,
    help="Pull request number.",
    metavar="INT",
)
@click.option(
    "--no-post",
    is_flag=True,
    default=False,
    help="Run the review but do NOT post the comment to GitHub.",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Save the Markdown review to this file (optional).",
    metavar="PATH",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default=None,
    help="Override log level for this run.",
)
def review(
    repo: str,
    pr_number: int,
    no_post: bool,
    output_file: str | None,
    log_level: str | None,
) -> None:
    """Run the full PR review pipeline for a given pull request."""
    try:
        settings = get_settings()
    except Exception as exc:
        click.echo(f"❌ Configuration error: {exc}", err=True)
        sys.exit(EXIT_CONFIG_ERROR)

    effective_log_level = log_level or settings.log_level
    configure_logging(log_level=effective_log_level, log_format="console")
    log = get_logger(__name__)

    # Validate repo format
    if "/" not in repo or repo.count("/") != 1:
        click.echo(
            f"❌ Invalid repo format '{repo}'. Expected 'owner/repo'.", err=True
        )
        sys.exit(EXIT_CONFIG_ERROR)

    click.echo(f"🔍 Reviewing PR #{pr_number} in {repo} …")

    try:
        from pr_review_agent.graph.pipeline import run_pipeline
        from pr_review_agent.services.review_poster import ReviewFormatter, ReviewPoster

        state = run_pipeline(pr_number=pr_number, repo=repo)
    except Exception as exc:
        log.exception("pipeline execution failed")
        click.echo(f"❌ Pipeline failed: {exc}", err=True)
        sys.exit(EXIT_PIPELINE_ERROR)

    if state.get("error"):
        click.echo(f"❌ Pipeline error: {state['error']}", err=True)
        sys.exit(EXIT_PIPELINE_ERROR)

    formatter = ReviewFormatter()
    body = formatter.format_full_review(state)
    body = formatter.truncate(body, settings.github_comment_max_chars)

    # Optionally save to file
    if output_file:
        Path(output_file).write_text(body, encoding="utf-8")
        click.echo(f"💾 Review saved to {output_file}")

    # Post to GitHub
    if not no_post:
        try:
            from pr_review_agent.services.github_service import GitHubService
            from pr_review_agent.services.review_poster import ReviewPoster

            poster = ReviewPoster()
            poster.post_review(state, repo=repo, pr_number=pr_number)
            click.echo(f"✅ Review posted to PR #{pr_number}")
        except Exception as exc:
            log.exception("failed to post review to GitHub")
            click.echo(f"❌ Failed to post review: {exc}", err=True)
            sys.exit(EXIT_GITHUB_ERROR)
    else:
        click.echo("ℹ️  --no-post flag set: skipping GitHub comment.")
        if not output_file:
            # Print to stdout as fallback
            click.echo("\n" + body)

    # Print summary
    stats = state.get("stats")
    if stats:
        click.echo(
            f"\n📊 Summary: {stats.files_reviewed} files reviewed, "
            f"{stats.issues_found} issues found, "
            f"{stats.test_suggestions_count} test gaps, "
            f"{stats.total_tokens_used:,} tokens used."
        )

    sys.exit(EXIT_SUCCESS)


# ── validate-config command ───────────────────────────────────────────────────


@cli.command("validate-config")
def validate_config() -> None:
    """Validate configuration and test API credentials."""
    configure_logging(log_level="WARNING", log_format="console")

    click.echo("Validating configuration…")
    errors: list[str] = []

    # Settings validation
    try:
        settings: Settings = get_settings()
        click.echo("  ✅ Settings loaded")
    except Exception as exc:
        click.echo(f"  ❌ Settings error: {exc}", err=True)
        sys.exit(EXIT_CONFIG_ERROR)

    # Test Gemini connectivity
    try:
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.llm_model)
        model.generate_content("ping", generation_config={"max_output_tokens": 5})
        click.echo("  ✅ Gemini API key valid")
    except Exception as exc:
        errors.append(f"Gemini API: {exc}")
        click.echo(f"  ❌ Gemini API error: {exc}", err=True)

    # Test GitHub token
    try:
        from github import Auth, Github

        g = Github(auth=Auth.Token(settings.github_token))
        user = g.get_user()
        click.echo(f"  ✅ GitHub token valid (authenticated as @{user.login})")
    except Exception as exc:
        errors.append(f"GitHub token: {exc}")
        click.echo(f"  ❌ GitHub token error: {exc}", err=True)

    if errors:
        click.echo(f"\n❌ Validation failed with {len(errors)} error(s).", err=True)
        sys.exit(EXIT_CONFIG_ERROR)

    click.echo("\n✅ All checks passed. Ready to review PRs.")
    sys.exit(EXIT_SUCCESS)


if __name__ == "__main__":
    cli()
