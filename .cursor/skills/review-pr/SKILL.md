---
name: review-pr
description: Reviews a pull request from a PR URL by directly fetching the URL content (no `gh` dependency) and verifies compliance with CONTRIBUTING.md. Use when the user asks for a PR review, to review changes before merge, or to check contribution guidelines.
---

# Review PR (From PR URL)

Reviews the changes in a specific GitHub pull request and ensures all rules in the project’s [CONTRIBUTING.md](CONTRIBUTING.md) (repository root) are followed.

## 1. Input: PR URL

Require a PR URL (for example: `https://github.com/RLinf/RLinf/pull/123`).

## 2. Fetch PR data directly from URL

Fetch PR details directly from the URL:

- Open/fetch the PR page itself for title, description, and metadata.
- Fetch unified diff via URL forms:
  - `<PR_URL>.diff` (preferred)
  - `<PR_URL>.patch` (fallback)
- If needed, fetch related pages directly from URL for comments/checks.

If the PR page is private and URL fetch is blocked, report that access is unavailable and ask the user to provide exported diff/details.

## 3. Review against CONTRIBUTING.md

Validate the changed files and diff against the following. Details and examples are in CONTRIBUTING.md; summary below.

### Cross-check with RLinf codebase context (required)

Do not review the PR in isolation. In addition to the diff, always map each major change to related RLinf modules and verify integration consistency:

- Identify related call sites/import paths/config/docs/tests in the RLinf codebase.
- Check for likely missing follow-up edits (e.g., moved paths not updated everywhere, stale imports, outdated scripts, registry wiring gaps).
- Check for behavior-level regressions caused by partial refactors (e.g., new package layout but old runtime assumptions).
- If a file is renamed/moved, verify references across code, docs, CI, and tooling.

Report these as findings when you see potential omissions, even if the exact line is outside the PR diff.

### Prime directive

- **User-facing changes** must have **tests** and **documentation**. A reviewer must be able to validate reproducibility.

### Documentation consistency checks (required when docs change)

For docs PRs, or PRs touching `docs/`, always perform a cross-language and style consistency review:

- For paired pages under `docs/source-en/` and `docs/source-zh/` (same topic), verify semantic parity for:
  - setup commands, paths, env vars, and config keys
  - supported models/envs/algorithms and capability claims
  - reported numbers (metrics, table values, dataset sizes, trial counts)
- Flag duplicated, missing, or conflicting paragraphs between EN and ZH pages unless explicitly justified.
- For embodied example pages, cross-check style against sibling docs (for example `opensora.rst`) and flag inconsistent section naming/order, code-block conventions, and result table formatting/link style.
- Highlight terminology/wording mistakes that can mislead users (e.g., duplicated suite names, missing suite names, mismatched metric descriptions).
- Each docs finding must include a concrete suggested wording or structural fix and exact file references.

### Code style and formatting

- **Style**: [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html); be consistent with surrounding code.
- **Lint**: Code should pass pre-commit (e.g. `pre-commit run --all-files`). Flag if new code likely fails lint.
- **Comments & docstrings**: Sufficient comments and docstrings; **public classes and methods** must have docstrings (Google style).
- **Type hints**: Functions/methods should have type hints on parameters; add return type when static analysis cannot deduce it.
- **Error handling**: Assertions and exceptions must have **clear, meaningful messages** (no empty or “xxx != yyy” restatements). Validate inputs/states early (e.g. before division or indexing).
- **Logging**: Use logging (or in Workers: `self.log_info` / `log_warning` / `log_error`) instead of `print`.
- **Config YAML**: Prefer copying existing configs from main as templates; **no calculations or dynamic values** in YAML (do in code, e.g. `config.py`); config fields must be **read-only** in code; avoid cross-field references in YAML when possible.
- **Tests**: New features must include CI tests; large/new dependencies (docker, models, datasets) → note that maintainers may need to be involved.
- **Dependencies/CI integration**: If the PR introduces new dependencies (for example, a new env/model requiring install script updates) or adds new YAML configs that should be exercised in CI, explicitly cross-check install script, Docker, and CI/e2e coverage using the [add-install-docker-ci-e2e skill](../add-install-docker-ci-e2e/SKILL.md). Flag missing installation wiring or missing test coverage.
- **Duplication**: Check for avoidable code duplication across modules/config/docs; suggest extracting shared helpers or reusing existing abstractions when repetition is significant.
- **Implementation simplicity**: If a solution is clearly clumsy or over-complicated, suggest a simpler/more maintainable alternative and explain why it is safer or easier to evolve.
- **No hardcoded paths/hacks**: Flag hardcoded machine-specific paths and heavy hacky workarounds. Prefer config/env-driven paths and robust integration points.

### Commit messages and sign-off

- Every commit must have a **Signed-off-by** line (e.g. `git commit -s`).
- Commit messages must follow **Conventional Commits**: `<type>(<scope>): <description>` (e.g. `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`).

### PR title and description

- **PR title**: Same format as commit messages: `<type>(<scope>): <description>`.
- **Description**: Must fill at least the **Description** and **Checklist** sections of the PR template; link related issues in Motivation and Context; if the change affects training performance/stability, provide testing results in “How has this been tested?”.

## 4. Explain the PR first

Before listing issues, start with a brief explanation of the PR:

- What problem it tries to solve.
- Main change categories (e.g., packaging, docs, CI, refactor).
- Potential impact/risk areas.

Keep this concise (3-6 bullets), then move to findings.

## 5. Output format

- List **violations** of CONTRIBUTING.md with file/line or commit reference where possible.
- Call out missing **tests** or **documentation** for user-facing changes.
- Note **suggestions** (e.g. style, clarity, type hints) that are not strict violations.
- If the diff or commit list is large, focus on the most relevant files and the prime directive first.
- Include the reviewed PR URL in the final report for traceability.
- For every finding, include a concrete **suggested fix**.
- Organize findings as **bullets** ordered by severity (highest first), not as a table.
- Use one concise bullet per finding with this structure:
  - `Severity` + `Area/File`: issue summary
  - `Suggested fix`: concrete action
  - `Reference`: file/line, commit, or diff snippet
- Explicitly label findings discovered via **codebase cross-check** (not only direct diff lines).

For a concise checklist derived from CONTRIBUTING.md, see [reference.md](reference.md).
