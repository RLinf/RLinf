# AGENTS.md

This file tells coding agents how to build, test, and change RLinf so that the result passes CI and review. `CLAUDE.md` is a symlink to it. It holds only rules that apply across the repository: the contribution process is in [CONTRIBUTING.md](CONTRIBUTING.md), writing and documentation rules are in [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md), and step-by-step workflows are skills under `.agents/skills/`. When this file disagrees with the code, the code is right; fix this file in the same change.

RLinf is a distributed reinforcement learning framework for embodied and agentic AI. A run starts from one entry script that builds a Ray-backed `Cluster`, places the actor, rollout, env, reward, and agent components on hardware, launches each as a `Worker` group, and hands control to a runner that loops through rollout, reward, advantage, and update. Configuration is Hydra YAML. Training uses FSDP or Megatron; LLM rollout uses SGLang or vLLM. User-facing workflows such as installation, placement, multi-node setup, memory tuning, checkpoint resume, metrics, and evaluation are documented under `docs/source-en/rst_source/` (`start/`, `concepts/`, `guides/`, `extending/`, and `resources/faq.rst`).

## Working with a human submitter

- A human submits and answers for every change. Push, open or edit PRs and issues, and post review comments only when the user asks for that action and has seen the content.
- Before starting a fix or feature, search open PRs and issues for the same work (`gh pr list --repo RLinf/RLinf --search "<keywords>"`) and tell the user about any overlap.
- When a requirement is ambiguous or a constraint blocks the change, stop and ask. Do not leave `TODO(agent)` or similar placeholders in code; record known limitations in the PR description.
- Report results as they are: which tests ran, on what hardware, and which failed or were skipped. State in the PR description that an AI assistant was used.

## Setup and commands

`requirements/install.sh` builds a uv virtual environment (`.venv` unless `--venv` says otherwise) for one target: `embodied`, `agentic`, or `docs`. `--platform` selects a non-NVIDIA accelerator (`amd`, `ascend`, `musa`, `kunlun`, `biren`).

```bash
bash requirements/install.sh embodied --model <model> --env <env>
bash requirements/install.sh agentic --engine sglang            # or vllm
bash requirements/install.sh --no-flash-attn embodied --env dummy  # the unit-test CI environment
bash requirements/install.sh docs --venv .docs-venv
```

Lint runs through pre-commit: Ruff lint and format, plus Conventional Commit and sign-off checks on the commit message.

```bash
pre-commit install --hook-type commit-msg
pre-commit run --all-files
```

CI runs each unit-test file in its own pytest process and stops Ray between files. `ray stop` is machine-wide, so on a shared host check `ray status` for other users' runs first.

```bash
export PYTHONPATH=$(pwd):$(pwd)/tests/unit_tests
pytest tests/unit_tests/test_worker.py; ray stop
pytest --doctest-modules rlinf/scheduler
```

Embodied end-to-end tests need GPUs and model assets. `<config>` is a file name in `tests/e2e_tests/embodied/` without `.yaml`; the optional second argument is the render backend (default `egl`). Use `run_async.sh` for async configs.

```bash
REPO_PATH=$(pwd) bash tests/e2e_tests/embodied/run.sh <config>
```

Docs must build in both languages with no new warnings:

```bash
sphinx-build -b html docs/source-en docs/build/html-en
sphinx-build -b html docs/source-zh docs/build/html-zh
```

## Repository map

- `rlinf/scheduler/` – `Cluster`, `Worker` and `WorkerGroup`, channels, collectives, placement, and the hardware and accelerator layer.
- `rlinf/workers/` – actor (FSDP, Megatron), critic, rollout (HF, SGLang, vLLM), env, reward, inference, agent, and SFT workers.
- `rlinf/runners/` – training and evaluation loops, one per task family (embodied sync and async, reasoning, agent, SFT, offline RL).
- `rlinf/algorithms/` – advantage, loss, and reward registries.
- `rlinf/models/` – model builders (`register_model`, `get_model`); embodied policies live in `models/embodiment/`.
- `rlinf/envs/` – simulators in `sim/<name>/`, real-world envs in `real/`, and `get_env_cls()` in `__init__.py`.
- `rlinf/robotics/` – robots, parts, and adapters behind the real-world envs.
- `rlinf/hybrid_engines/` – FSDP, Megatron, SGLang, and vLLM integration and weight syncing.
- `rlinf/config.py` – `build_config`, `validate_cfg`, and `SupportedModel`.
- `examples/` – entry scripts and Hydra configs: `embodiment/`, `reasoning/`, `agent/`, `sft/`, `offline_rl/`, and others.
- `evaluations/` – standalone embodied evaluation: `bash evaluations/run_eval.sh <benchmark> <config>`.
- `tests/` – `unit_tests/`, `e2e_tests/`, `robot_mocks/` (fake vendor SDKs), and `parity_tests/`.
- `requirements/`, `docker/`, `ray_utils/`, `toolkits/`, and `docs/` (Sphinx, `source-en/` and `source-zh/`).

## Rules CI and reviewers enforce

- **Accelerators.** Unit tests run on CPU, NVIDIA, AMD, Ascend, and MUSA runners. In code that runs on an accelerator, call `self.torch_platform` in a worker (`Worker.torch_platform` elsewhere) rather than `torch.cuda`, and keep tests runnable without a GPU unless the behavior under test needs one. Vendor-specific behavior belongs in `rlinf/scheduler/hardware/accelerators/`.
- **Optional dependencies.** Simulators, robot SDKs, and model-specific packages are not installed in every environment. Import them inside the function or branch that uses them, as `get_env_cls()` does.
- **Registration is per process.** A registry entry made in the driver does not reach Ray workers. In-tree components register at module import; out-of-tree code registers through `RLINF_EXT_MODULE` (see `extending/new_model_fsdp.rst`).
- **Package layout.** Every directory under `rlinf/` has an `__init__.py` (`tests/unit_tests/check_missing_init.py`), and every Python file starts with the Apache-2.0 header `# Copyright <year> The RLinf Authors.` (Ruff `CPY001`).
- **Ruff scopes.** Docstring rules (`D`) apply only to `rlinf/scheduler/`, and annotation rules (`ANN`) only to `rlinf/robotics/` and `rlinf/envs/real/`. Public APIs everywhere still need Google-style docstrings and type hints.
- **Comments and docstrings.** Keep them short and written for someone reading the current code. State the contract, invariants, ownership, and any non-obvious reason; do not narrate the implementation, repeat the signature, or advertise the design. Leave out corner cases a caller never meets and the history of a fix (what broke, how it was found, what was tried); that belongs in the commit message or PR description. A corner case earns a comment only when the code would otherwise look wrong or invite a "simplification" that reintroduces the bug, and then one sentence says why.
- **CI path filters.** `.github/workflows/ci-tests.yml` maps paths to test jobs, and its `filter-coverage-test` job fails when a changed `.py`, test, or root `.yaml` file matches no filter. When you add or move files, add them to the filter whose tests exercise them. The `CI Test` workflow, including unit tests, runs only on non-draft PRs labeled `run-ci`; lint and the PR title check run on every PR.
- **Config YAML.** Values are static: no computed or dynamic values, and as few references to other fields as possible. Code never overwrites a field a user can set; derive values in `rlinf/config.py`. Start a new config from the closest current one on `main`.
- **Logging and errors.** Use `self.log_info`, `self.log_warning`, and `self.log_error` in a `Worker` and `rlinf.utils.logging.get_logger()` elsewhere; never `print`. Assertion and exception messages say what was expected and what was found, not just the failed condition.
- **Multi-node.** Each node exports a unique `RLINF_NODE_RANK` (and optionally `RLINF_COMM_NET_DEVICES`) before `ray start`, because Ray captures the environment when it starts. The entry script runs only on the head node.

## Extending RLinf

Each extension point is a registry or an enum. Register the component, select it in config, and follow the guide (under `docs/source-en/rst_source/`) for the full workflow, including install, Docker, CI, and docs.

| Component | Register with | Selected by | Guide |
|---|---|---|---|
| Advantage | `@register_advantage(name)` in `rlinf/algorithms/registry.py` | `algorithm.adv_type` | `rlinf/algorithms/advantages.py` |
| Policy loss | `@register_policy_loss(name)` in `rlinf/algorithms/registry.py` | `algorithm.loss_type` | `rlinf/algorithms/losses.py` |
| Rule-based reward | `register_reward(name, cls)` in `rlinf/algorithms/rewards/__init__.py` | `reward.reward_type` | existing rewards in `rlinf/algorithms/rewards/` |
| Model | `register_model(model_type, builder, category=...)` in `rlinf/models/__init__.py`; embodied policies subclass `BasePolicy` | `model.model_type` | `extending/new_model_fsdp.rst`, `extending/new_model_megatron.rst` |
| Simulator | a `SupportedEnvType` member and a lazy-import branch in `get_env_cls()`; code in `rlinf/envs/sim/<name>/`; action formatting in `prepare_actions()` in `rlinf/envs/action_utils.py` | `env.train.env_type`, `env.eval.env_type` | `extending/new_env.rst` |
| Robot or real-world task | `rlinf/robotics/` and `rlinf/envs/real/`, with fakes in `tests/robot_mocks/` | `env_type: real` | `extending/new_robot.rst`, `extending/new_task.rst` |
| Task family | a runner in `rlinf/runners/` and an entry script in `examples/` | entry script | `concepts/execution_flow.rst` |

## Tests

- `tests/unit_tests/` has one file per component, named for it (`test_worker.py`, `test_placement.py`, `test_robotics.py`, and so on). Add cases to the file of the component you change. A new file needs a new component; a file named after a fix, symptom, platform, or single function belongs in the component's file.
- Test through the contract a caller uses. A value visible only inside an implementation, such as a unit conversion or a wire format, is tested at the layer that owns it.
- Mock only what lies outside RLinf: vendor SDKs, hardware, and remote services. Robot fakes live in `tests/robot_mocks/`, and e2e configs whose names contain `mock` run real-world envs against them. A test that is mostly mock setup only proves the mocks were called; use a real object, a fake at the process edge, or no test.
- Mark a test that hosts parts in real scheduler workers with `@pytest.mark.placement`.
- Delete a test when you cannot name the change that would make it fail.
- New user-facing behavior needs a unit or e2e test. If the test needs GPUs, assets, or hardware that CI lacks, say so in the PR and ask the maintainers.

## Design and review principles

- Preserve behavior before simplifying. Trace the full call path and compare every affected backend, driver, environment, and task with the baseline; a difference is intentional only when it is named, documented, and tested. Before calling a field vestigial, check its builders, defaults, serialization, and runtime consumers.
- Prefer small, explicit abstractions with one responsibility and one stable name. Avoid parallel vocabularies, convenience APIs that hide ownership, and dynamic machinery that a direct constructor or method can express. Use a registry when independently developed components must extend a central factory.
- Design invalid states out of the API. Make ownership, lifecycle order, partial-failure rollback, and reconnect behavior explicit; give each resource one owner and make cleanup idempotent.
- Keep the common local path direct. Introduce remote placement, process boundaries, and shared resources only when the task needs them, and keep the same API in local and remote configurations. Public names, accepted types, and return types must be discoverable from type hints and docstrings.
- Judge an abstraction by how it composes with existing components, not by its smallest example.
- Review the whole contract, not only the newest diff: every implementation, builder, task, environment, test, and document that participates in it.

## Commits, PRs, and writing

- Commit subjects follow Conventional Commits (`<type>(<scope>): <description>`, imperative, about 72 characters). Every commit carries a `Signed-off-by:` trailer (`git commit -s`), which certifies the [DCO](DCO) for the human author; use that person's Git identity, never an agent's.
- PR titles are stricter than commit subjects: the description after `: ` is at most 50 characters, starts with a lowercase letter, has no trailing period, and is plain ASCII. The `create-pr` skill fills `.github/PULL_REQUEST_TEMPLATE.md` and checks both title and body with `python3 .agents/skills/create-pr/lint_pr.py lint --title "<title>" --body-file <body.md>`. Include the commands you ran and their results; a change that can move training curves needs curves.
- All project writing, including PR descriptions, review comments, and commit messages, follows the voice rules in [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md); its page-structure and RST rules apply to documentation only. Change the English and Chinese docs together.

## Skills

Skills live in `.agents/skills/<name>/SKILL.md`, and `.claude/skills/` holds a symlink to each. Use the one that matches the task:

| Task | Skill |
|---|---|
| Open a PR, or fix a PR title or description | `create-pr` |
| Review a PR against CONTRIBUTING.md | `review-pr` |
| Write or restructure a docs page | `refine-docs` |
| Check docs against the code and EN/ZH parity | `docs-check` |
| Add the example page for a new model or env | `add-example-doc-model-env` |
| Add a publication page | `add-publication-docs` |
| Add install, Docker, CI, and e2e support for a new model or env | `add-install-docker-ci-e2e` |
| Write or review `install.sh` and Dockerfile changes | `install-check` |
| Build a model or env venv and run its e2e test | `test-install` |

## Maintaining agent instructions

- This file is loaded on every agent request. Keep it under 200 lines, and add a rule only when it applies across the repository and an agent would get it wrong without being told.
- Put guidance for one area in that area's docs page, in a skill, or in an `AGENTS.md` inside the directory, with a `CLAUDE.md` symlink beside it.
- Refer to code by symbol or search pattern instead of copying lists that drift, such as supported models, envs, or test files. Check every path, command, and API you add against the code.
- When a new rule supersedes an old one, remove or merge the old one in the same change.
- Add a skill only under `.agents/skills/`, with a folder name that matches the `name` in its `SKILL.md`, and add its symlink in `.claude/skills/`.
