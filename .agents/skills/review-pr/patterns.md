# RLinf code and design patterns

This page is the standard a review measures a PR against. It summarizes the rules in [AGENTS.md](../../../AGENTS.md), [CONTRIBUTING.md](../../../CONTRIBUTING.md), and [docs/STYLE_GUIDE.md](../../../docs/STYLE_GUIDE.md) as patterns you can spot in a diff. Each pattern says what RLinf expects and what to flag. A finding that cites a pattern names the pattern, the code that breaks it, and the version that follows it.

## Design philosophy

RLinf is built from a small number of general abstractions, and everything else is an instance of them. Workers run components, placement maps them to hardware, channels and collectives move data between them, registries and enums select models, envs, and algorithms, and Hydra config declares a run. A new model, env, algorithm, robot, or accelerator plugs into these abstractions; it does not open a new path through the system. The principles below follow from that idea, and every pattern on this page applies one of them. When a diff fits no pattern, judge it against the principles directly.

1. **One general mechanism, with each familiar case as an instance of it.** Extend an existing abstraction until the new case falls out of it, rather than adding a branch beside it. Collocated, disaggregated, and hybrid execution are three placements of the same workers, not three code paths. A feature that needs its own if-else in shared code has not found its abstraction yet. See [Generalize; don't special-case](#generalize-dont-special-case).
2. **Each layer knows only its own concern.** The scheduler places and connects workers without knowing what they compute, a runner drives the algorithm loop without knowing a model's internals, and a model or env owns its specifics without knowing where it runs. Specifics stay inside the component that owns them, and shared code meets them only through an interface, so it never learns a component's name. See [Contain component code](#contain-component-code) and [Fix at the owning layer](#fix-at-the-owning-layer).
3. **Every resource has one owner.** A device, a connection, a process, or a config field is created and released by exactly one object, and everything else borrows it through that owner's interface. In robotics, a `Connection` opens and releases the vendor session, and the parts that ride on it only use it. Explicit ownership and lifecycle make cleanup idempotent and invalid states hard to represent. See [Small abstractions with one owner](#small-abstractions-with-one-owner).
4. **The common path stays direct, and scale is configuration.** A single process with direct calls is the default. More nodes, remote placement, or another execution mode comes from config rather than different code, and the caller uses the same API either way: a robot part placed on another node is still the object the caller holds, now a view of the remote one. See [Keep the common path direct](#keep-the-common-path-direct) and [Configuration](#configuration).
5. **Declare at the edge, trust inside.** Users declare a run in static YAML. RLinf derives the rest once, in `rlinf/config.py`, and validates config, external input, and vendor results at that boundary. Inside, code trusts the contracts of the types it receives and carries no defensive probing. See [No unnecessary protection code](#no-unnecessary-protection-code) and [Optional dependencies and registration](#optional-dependencies-and-registration).
6. **Changes are proven, not asserted.** Behavior is preserved unless a difference is named, documented, and tested, and a refactor is compared with `main` on every affected backend. Tests exercise real objects through public contracts, and mocks stand only at the process edge. See [Preserve behavior](#preserve-behavior) and [Tests](#tests).
7. **Clear code over clever code.** One stable name per concept across code, config, and docs; a direct constructor or method rather than dynamic machinery; comments that state contracts rather than narrate. A good change reads as if it had always been part of the codebase. See [Code](#code) and [Docs](#docs).

## Design

### Generalize; don't special-case

Expect the change to extend an existing mechanism so that the new case falls out of it. RLinf's mechanisms are the registries (`register_model`, `register_advantage`, `register_policy_loss`, `register_reward`), the enums (`SupportedModel`, `SupportedEnvType`) with `get_env_cls()`, the policy hooks on `BasePolicy`, the accelerator classes behind `Worker.torch_platform`, and the config dataclasses built in `rlinf/config.py`.

Flag:

- `if model_type == ...`, `if env_type in (...)`, `if "foo" in name`, or an accelerator check in a runner, worker, scheduler, or shared utility.
- A new flag or config key whose only job is to switch one feature's path on.
- A copy of an existing function or class with a few lines changed for the new case, instead of a parameter or hook on the original.
- Two code paths that do the same thing for two backends.

Ask for a method on the component that owns the behavior, a registry entry, a field on the existing spec or config, or a subclass hook. When a special case is truly unavoidable, it has one owner and one name, and a comment says why.

### Contain component code

Expect each component to live in its own place:

| Component | Home |
|---|---|
| Embodied model | `rlinf/models/embodiment/<model>/`, a `BasePolicy` subclass |
| Simulator | `rlinf/envs/sim/<name>/` |
| Robot or real-world task | `rlinf/robotics/` and `rlinf/envs/real/`, fakes in `tests/robot_mocks/` |
| Accelerator | `rlinf/scheduler/hardware/accelerators/` |
| Install | its `install_<name>_model` or `install_<name>_env` function in `requirements/install.sh` |
| Config | its own YAML under `examples/`, starting from the closest sibling |

Shared files get only the touch points their extension point defines: a registration call, an enum member, a lazy-import branch in `get_env_cls()`, a CI path filter, a docs index entry.

Flag model-specific preprocessing in a generic worker, env-specific logic in a runner, new keys in shared config groups that only one model reads, and helpers named after one model or env in `rlinf/utils/`.

### Small abstractions with one owner

Expect:

- One responsibility and one stable name per class or function, and one name per concept across code, config, and docs.
- Object-oriented code: a function that acts on a class's state is a method of that class, and one that does not use `self` is a `@staticmethod`. Module-level functions belong in utility modules.
- Callers that drive a component through its declared interface. If the interface cannot express what a caller needs, the interface is extended.
- Explicit ownership and lifecycle: one owner per resource, a defined create, start, and stop order, idempotent cleanup, and rollback on partial failure.
- Invalid states that the types rule out: an enum instead of a free string, a required constructor argument instead of an optional one checked later.

Flag:

- Wrapper functions or classes that only forward one call.
- Parallel vocabularies: two names, classes, or config keys for one concept.
- Dynamic machinery where a direct constructor or method works: string dispatch through `getattr`, `setattr` patching, `importlib` tricks, monkey-patches.
- Reaching into private attributes (`obj._driver`, `obj._conn`) or extracting an owner's handle to bypass its interface.

### Fix at the owning layer

Expect a fix in the layer that owns the invariant, so every consumer benefits. Flag a fix at the one call site that surfaced the bug, a `# noqa` that silences a linter instead of fixing what it reports, and a workaround in a model for a bug in shared code.

### No unnecessary protection code

Expect validation at the boundaries, once: user config in `rlinf/config.py` (`build_config`, `validate_cfg`), external input, and vendor SDK results. The message says what was expected and what was found. Inside RLinf, code trusts the contract of the types it receives.

Flag:

- `hasattr`, `getattr(x, "y", None)`, or `isinstance` ladders for attributes and types the caller always provides.
- `if x is None` for values that cannot be `None`, and defaults that hide a missing or wrong config.
- `try: ... except Exception:` that logs and continues in a wrong state, and bare `except` or `except: pass`.
- Retries, sleeps, or fallbacks for failures the PR never demonstrates.

### Keep the common path direct

Expect the local single-process path to stay a direct call. Remote placement, process boundaries, and shared resources appear only when the task needs them, and the API is the same in local and remote configurations.

### Preserve behavior

A refactor keeps behavior unless the PR names, documents, and tests the difference. Compare every affected backend (FSDP, Megatron; SGLang, vLLM; HF), env, model, and task with `main`. Before a field or branch is called dead, check its builders, defaults, serialization, and runtime consumers.

### Configuration

Expect static YAML copied from the closest current sibling on `main`, overriding only what differs from the Hydra `defaults`. Derived values are computed in `rlinf/config.py`. Code never writes a field a user can set. References between fields are rare.

### Optional dependencies and registration

Simulators, robot SDKs, and model-specific packages are imported inside the function or branch that uses them. Registries fill at module import in each process; out-of-tree code registers through `RLINF_EXT_MODULE`.

## Code

- Logging uses `self.log_info`, `self.log_warning`, and `self.log_error` in a `Worker`, and `rlinf.utils.logging.get_logger()` elsewhere. Product code has no `print`.
- Accelerator code uses `self.torch_platform` (or `Worker.torch_platform`), not `torch.cuda`.
- Public APIs have type hints, accepted input types, return types, and short Google-style docstrings.
- Comments are sparse. They state contracts, invariants, ownership, and non-obvious reasons. They do not narrate the code, repeat the signature, or tell the history of a fix.
- Assertion and exception messages say what was expected and what was found.
- New files carry `# Copyright <current year> The RLinf Authors.`; moved or renamed files keep their year, and vendored files keep their upstream header.
- Elegant code reads as if it had always been part of the codebase: names match the neighbors, no scaffolding is left behind, and the diff to shared files is as small as the extension point allows.

## Tests

- `tests/unit_tests/` holds one file per component. New cases go into the component's existing file. Only a new component justifies a new file, never a model, env, fix, symptom, or single function.
- Prefer a few tests that exercise real objects through the public contract. Mock only what lies outside RLinf: vendor SDKs, hardware, and remote services. Robots are the usual case, through the fakes in `tests/robot_mocks/`.
- Flag tests that are mostly mock setup, tests that only prove a mock was called, and tests that no named change would make fail.
- A new model or env comes with an e2e config, an install path, and a CI path filter entry.

## Docs

- Follow [docs/STYLE_GUIDE.md](../../../docs/STYLE_GUIDE.md): the first sentence states the page's purpose, ideas come before names, pages describe what exists now, and English and Chinese change together.
- A feature worth announcing (a new model, env, algorithm, backend, accelerator, robot, or major capability) is highlighted in both READMEs in the same PR: a dated "What's NEW!" entry and an item in the matching support list or hardware table, each linking to its docs page.
- Pages are written for a first-time reader who has not read RLinf's code. They explain what a feature does, how it works, what to prepare, and how to run it, and they show code and paths only where the reader uses them: config examples, config keys, run commands, and, on Extending pages, the files a contributor edits.
- Each language reads as if written in it rather than translated, and sentences, paragraphs, and sections follow one connected line of thought.
- Guide pages are plain and technical: simple task headings, one config example, one worked example, and only the config keys a user writes. They leave out equations and tensor-shape tables.
- Use the [docs-check](../docs-check/SKILL.md) skill for doc-to-code checks and EN/ZH parity, and [refine-docs](../refine-docs/SKILL.md) for structure and voice.
