# PR Review Checklist (from CONTRIBUTING.md)

Use this alongside the git diff when reviewing. Tick or address each item for the changed files and commits.

## Prime directive
- [ ] User-facing changes have tests
- [ ] User-facing changes have documentation (reproducibility validated by reviewer)

## Code style & formatting
- [ ] Google Python Style Guide; consistent with surrounding code
- [ ] Lint passes (pre-commit)
- [ ] Comments and docstrings present; **public classes/methods** have Google-style docstrings
- [ ] Type hints on function/method parameters; return type when not deducible
- [ ] Assertions/exceptions have clear, meaningful messages; invalid inputs checked early
- [ ] Logging used instead of print; in Workers use `self.log_info` / `log_warning` / `log_error`
- [ ] Config YAML: copied from existing templates; no dynamic values; read-only in code; minimal cross-field refs
- [ ] New features have CI tests; large deps (docker/models/datasets) → maintainer ping noted

## Commits
- [ ] Every commit has Signed-off-by (e.g. `git commit -s`)
- [ ] Commit messages follow Conventional Commits: `<type>(<scope>): <description>`

## PR metadata
- [ ] PR title: `<type>(<scope>): <description>`
- [ ] PR description: Description and Checklist sections filled; issue linked if applicable; testing results if performance/stability impact
