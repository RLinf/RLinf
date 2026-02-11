---
name: review-pr
description: Reviews a pull request by comparing the current branch to main via git diff and verifying compliance with CONTRIBUTING.md. Use when the user asks for a PR review, to review changes before merge, or to check contribution guidelines.
---

# Review PR (Current Branch vs Main)

Reviews the changes introduced in the current branch against `main` and ensures all rules in the project’s [CONTRIBUTING.md](CONTRIBUTING.md) (repository root) are followed.

## 1. Get the diff

Use git to obtain the diff between the current branch and `main`:

```bash
git fetch origin main
git diff origin/main...HEAD --name-only    # list changed files
git diff origin/main...HEAD                # full diff
```

Use `origin/main...HEAD` (three dots) so the diff is the set of commits reachable from HEAD but not from main (what the PR would introduce). Optionally use `git log origin/main..HEAD --oneline` to see commit messages.

## 2. Review against CONTRIBUTING.md

Validate the changed files and diff against the following. Details and examples are in CONTRIBUTING.md; summary below.

### Prime directive

- **User-facing changes** must have **tests** and **documentation**. A reviewer must be able to validate reproducibility.

### Code style and formatting

- **Style**: [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html); be consistent with surrounding code.
- **Lint**: Code should pass pre-commit (e.g. `pre-commit run --all-files`). Flag if new code likely fails lint.
- **Comments & docstrings**: Sufficient comments and docstrings; **public classes and methods** must have docstrings (Google style).
- **Type hints**: Functions/methods should have type hints on parameters; add return type when static analysis cannot deduce it.
- **Error handling**: Assertions and exceptions must have **clear, meaningful messages** (no empty or “xxx != yyy” restatements). Validate inputs/states early (e.g. before division or indexing).
- **Logging**: Use logging (or in Workers: `self.log_info` / `log_warning` / `log_error`) instead of `print`.
- **Config YAML**: Prefer copying existing configs from main as templates; **no calculations or dynamic values** in YAML (do in code, e.g. `config.py`); config fields must be **read-only** in code; avoid cross-field references in YAML when possible.
- **Tests**: New features must include CI tests; large/new dependencies (docker, models, datasets) → note that maintainers may need to be involved.

### Commit messages and sign-off

- Every commit must have a **Signed-off-by** line (e.g. `git commit -s`).
- Commit messages must follow **Conventional Commits**: `<type>(<scope>): <description>` (e.g. `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`).

### PR title and description

- **PR title**: Same format as commit messages: `<type>(<scope>): <description>`.
- **Description**: Must fill at least the **Description** and **Checklist** sections of the PR template; link related issues in Motivation and Context; if the change affects training performance/stability, provide testing results in “How has this been tested?”.

## 3. Output

- List **violations** of CONTRIBUTING.md with file/line or commit reference where possible.
- Call out missing **tests** or **documentation** for user-facing changes.
- Note **suggestions** (e.g. style, clarity, type hints) that are not strict violations.
- If the diff or commit list is large, focus on the most relevant files and the prime directive first.

For a concise checklist derived from CONTRIBUTING.md, see [reference.md](reference.md).
