---
name: code
description: "RLinf code, tests, configs, and CI (rlinf/, examples/, tests/, toolkits/, evaluations/, ray_utils/, .github/, any .py file): read before writing or changing them, then check the change. Covers RLinf's design patterns (extension points, containment, ownership, protection code, config), code conventions, unit-test layout and mocks, license headers, and CI path filters. Use before adding or changing a model, env, worker, runner, algorithm, config, or test, and to check such a change."
paths:
  - rlinf/**
  - examples/**
  - tests/**
  - toolkits/**
  - evaluations/**
  - ray_utils/**
  - .github/**
  - "**/*.py"
  - "*.yaml"
  - pyproject.toml
---

# Code

RLinf's design patterns are in [patterns.md](patterns.md), and the rules CI and reviewers enforce are in [AGENTS.md](../../../AGENTS.md). This skill says what to read before you change code and checks the change afterwards. Install, Docker, and e2e wiring for a new model or env belong to the [install](../install/SKILL.md) skill, and docs and README highlights to the [docs](../docs/SKILL.md) skill.

## Write

1. Find the closest current sibling on `main` and start from it: the newest model under `rlinf/models/embodiment/`, env under `rlinf/envs/`, or runner, worker, or algorithm of the same kind. `git log <base> -- <dir>` shows which one is recent, so you copy current conventions rather than legacy boilerplate.
2. Plug into the extension point the component kind already has, from the table in AGENTS.md § Extending RLinf, instead of adding a branch to shared code.
3. Read [patterns.md](patterns.md): its design philosophy first, then the Design, Code, and Tests patterns that apply to the change.
4. Put new tests in the component's existing file in `tests/unit_tests/`, through the public contract, and mock only what lies outside RLinf (patterns.md § Tests).
5. Start a new config YAML from the closest current one in `examples/`, keep values static, and override only what differs from the Hydra `defaults` (patterns.md § Configuration).
6. Add every new `.py`, test, or root YAML file to the path filter in `.github/workflows/ci-tests.yml` whose tests exercise it.
7. Run the check below, and the unit tests of the component you changed (AGENTS.md § Setup and commands).

## Check

```bash
python3 .agents/skills/code/check.py              # files changed against main
python3 .agents/skills/code/check.py FILE ...
```

It prints findings in the shared format of [README.md](../README.md#findings):

| Rule | Severity | Reports |
|---|---|---|
| `code/pre-commit` | major | `pre-commit run --files` fails: Ruff lint and format, and the license header where Ruff applies. Ruff runs with `--fix`, so run the check again after it rewrites a file. |
| `code/init` | major | a directory under `rlinf/` with Python files and no `__init__.py` |
| `code/license` | major or minor | a new Python file without `# Copyright <year> The RLinf Authors.` (major), or with a year other than the current one (minor) |
| `code/ci-filter` | major | a changed file that CI's `filter-coverage-test` would reject |
| `code/test-layout` | minor | a new file in `tests/unit_tests/` |

Then go through what a script cannot decide, for every changed function, compared with its version on `main`:

- [ ] The change extends an existing mechanism; shared code gains no branch that names a model, env, or accelerator (patterns.md § Generalize; don't special-case, § Contain component code).
- [ ] Each resource has one owner, and the fix sits at the layer that owns the invariant (§ Small abstractions with one owner, § Fix at the owning layer).
- [ ] No protection code for states the caller cannot produce, and validation happens once at the boundary (§ No unnecessary protection code).
- [ ] Behavior is preserved on every affected backend, env, model, and task, or the difference is named, documented, and tested (§ Preserve behavior).
- [ ] No new function, field, or config key without a caller or consumer, and no changed line that does not serve the change.
- [ ] Code conventions hold: logging through `self.log_*` or `get_logger()`, `self.torch_platform` instead of `torch.cuda`, type hints and Google-style docstrings on public APIs, short comments, and messages that say what was expected and what was found (§ Code).
- [ ] Tests fail without the change, use real objects, mock only the process edge, and carry `@pytest.mark.placement` when they host parts in scheduler workers (§ Tests).
