# PR Review Agent

A production-ready automated GitHub pull request review agent built with **LangGraph**, **LangChain**, and **Google Gemini**.

## Architecture

```
PR Fetcher → Code Reviewer → Test Coverage → Doc Summarizer → GitHub Comment
```

Four LangGraph nodes run sequentially, each consuming and enriching a shared `AgentState`:

| Agent | Responsibility |
|---|---|
| **PR Fetcher** | Fetch PR metadata and file diffs from GitHub API |
| **Code Reviewer** | Static analysis + Gemini LLM review for bugs, security, style |
| **Test Coverage** | Detect untested new functions; generate pytest stubs |
| **Doc Summarizer** | Generate structured PR summary with risk assessment |

## Quick Start

### Prerequisites
- Python 3.11+
- Google Gemini API key (get one at https://aistudio.google.com/apikey)
- GitHub personal access token

### Installation

```bash
# Install with pip
pip install -e ".[dev]"

# Copy and fill in environment variables
cp .env.example .env
```

### Run a PR Review

```bash
# Review PR #42 in owner/repo and post results to GitHub
pr-review review --repo owner/repo --pr 42

# Dry run (no GitHub post), save to file
pr-review review --repo owner/repo --pr 42 --no-post --output-file review.md

# Validate your configuration
pr-review validate-config
```

### Run Webhook Server

```bash
uvicorn pr_review_agent.server.app:app --host 0.0.0.0 --port 8080
```

### Docker

```bash
docker compose up -d
```

## Development

```bash
# Run tests
pytest

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Type check
mypy -p pr_review_agent
```

### Pre-commit Hooks

The repo ships with a [pre-commit](https://pre-commit.com/) config that automatically reformats and lints your code on every `git commit`.

**First-time setup** (one-off, after cloning):

```bash
pip install pre-commit
pre-commit install
```

After that, every commit automatically runs:

| Hook | What it does |
|---|---|
| `ruff-format` | Reformats staged files in-place |
| `ruff --fix` | Auto-fixes lint issues; blocks commit if unfixable errors remain |

If a commit is blocked, fix the reported errors, re-stage the files, and commit again:

```bash
# ruff will tell you what's wrong, e.g.:
# src/foo.py:10:5: F401 'os' imported but unused
ruff check --fix src/foo.py
git add src/foo.py
git commit -m "your message"
```

To run the hooks manually against all files at any time:

```bash
pre-commit run --all-files
```

## Configuration

All settings are read from environment variables. See `.env.example` for the full list.

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ | Google Gemini API key |
| `GITHUB_TOKEN` | ✅ | GitHub PAT with `repo` scope |
| `LLM_MODEL` | | Gemini model (default: `gemini-2.0-flash-lite`) |
| `LLM_TEMPERATURE` | | LLM temperature (default: `0.2`) |
| `WEBHOOK_SECRET` | | HMAC secret for webhook validation |
| `PORT` | | Webhook server port (default: `8080`) |
| `LOG_LEVEL` | | Logging level (default: `INFO`) |
| `LOG_FORMAT` | | `json` or `console` (default: `json`) |
