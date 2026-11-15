# Mistakes agents make in code

Coding agents repeat a small set of mistakes in RLinf, and reviewers send them back each time. Go through this list before you report a change as done. Each line names the mistake and the fix; the rule behind it is in [patterns.md](patterns.md) or [AGENTS.md](../../../AGENTS.md).

## Design

| Mistake | Fix | Rule |
|---|---|---|
| A branch in shared code that names one model, env, or accelerator (`if model_type == ...`) | Put the behavior in the component, behind the hook or registry its kind already has | patterns.md § Generalize; don't special-case |
| A copy of an existing function or class with a few lines changed | Add a parameter or a hook to the original | § Generalize; don't special-case |
| A new registry, factory, flag, or config key for a single use | Use a direct constructor or the existing extension point | § Small abstractions with one owner |
| Wrappers, managers, and layers that only forward calls | Call the owner directly | § Keep the common path direct |
| A fix at the call site that surfaced the bug | Fix the layer that owns the invariant, so every caller benefits | § Fix at the owning layer |
| Code that writes a user-settable config field, or derives config outside `rlinf/config.py` | Treat user fields as read-only; derive values in `rlinf/config.py` | § Configuration |
| A refactor that changes behavior on a backend, env, or task nobody compared | Compare every affected path with `main`; name, document, and test intended differences | § Preserve behavior |

## Defensive and dead code

| Mistake | Fix | Rule |
|---|---|---|
| `hasattr`, `getattr(x, "y", None)`, `isinstance` ladders, and `None` checks for states the caller never produces | Trust the contract inside RLinf; validate once at the boundary | § No unnecessary protection code |
| `try: ... except Exception:` that logs and continues, bare `except`, `except: pass` | Let the error surface, or handle the specific exception with a clear message | § No unnecessary protection code |
| Retries, sleeps, fallbacks, and compatibility shims for failures nobody demonstrated | Remove them, or add the test that shows the failure | § No unnecessary protection code |
| Unused parameters, fields, helpers, and config keys; commented-out code; leftover debug output | Delete them | patterns.md § Code |
| Reformatting, renames, and drive-by refactors in code the change does not need | Keep the diff to what the task requires | patterns.md § Code |

## Conventions

| Mistake | Fix | Rule |
|---|---|---|
| `print` in product code | `self.log_*` in a worker, `get_logger()` elsewhere | AGENTS.md § Logging and errors |
| `torch.cuda` calls in code that runs on an accelerator | `self.torch_platform` or `Worker.torch_platform` | AGENTS.md § Accelerators |
| Importing a simulator, robot SDK, or model package at module top level | Import it inside the function or branch that uses it | AGENTS.md § Optional dependencies |
| Registering a component in the driver and expecting Ray workers to see it | Register at module import, or through `RLINF_EXT_MODULE` | AGENTS.md § Registration is per process |
| Hardcoded machine paths, GPU indices, hostnames, or credentials | Take them from config or the environment | patterns.md § Code |
| An exception or assert message that only restates the condition | Say what was expected and what was found | AGENTS.md § Logging and errors |
| Comments that narrate the code, repeat the signature, or tell the history of a fix | State the contract, the invariant, or the non-obvious reason, in one sentence | AGENTS.md § Comments and docstrings |
| A public function without type hints or a Google-style docstring | Add them | AGENTS.md § Ruff scopes |
| A new file without `# Copyright <year> The RLinf Authors.`, or a new directory under `rlinf/` without `__init__.py` | Add them; the code check reports both | AGENTS.md § Package layout |

## Accuracy

| Mistake | Fix |
|---|---|
| Calling a function, flag, or config key that does not exist, guessed from a similar library or an older version | Read the code or the installed package before you use it |
| Changing a public name without updating its callers, configs, docs, and tests | Search the whole repository, and update everything in the same change |

## Tests

| Mistake | Fix | Rule |
|---|---|---|
| A new test file named after a fix, model, or symptom | Add the cases to the component's file in `tests/unit_tests/` | patterns.md § Tests |
| Tests that are mostly mock setup, or that only assert a mock was called | Use real objects; mock only vendor SDKs, hardware, and remote services | § Tests |
| A test that passes without the change | Make it fail on `main` first, or delete it | § Tests |
| Weakening an assertion, adding a skip, or editing the expected value so a failing test passes | Fix the code, or explain to the user why the expectation was wrong | AGENTS.md § Working with a human submitter |
| A GPU requirement in a test that needs none | Keep it runnable on CPU | AGENTS.md § Accelerators |

## Process

| Mistake | Fix |
|---|---|
| Starting before reading the closest current sibling on `main` | Read it first and match its shape |
| Reporting done without running anything, or saying tests passed when they were skipped | Run `python3 .agents/skills/route.py check` and the component's tests, and report what ran, on what hardware, and what failed |
| Leaving `TODO(agent)` or placeholder code | Finish it, or stop and ask the user; record known limits in the PR description |
| Committing unrelated files, generated output, or vendored edits | Stage only the change |
| Pushing, opening a PR, or posting a review without being asked | Do it only when the user asks |
