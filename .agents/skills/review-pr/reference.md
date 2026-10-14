# PR Review Checklist

Use alongside the PR diff. **Always cross-reference against `origin/main`** (`git fetch origin main` first; read with `git show origin/main:<path>` or the GitHub `main` URL), not the local working tree. Categories are in priority order — most of the review should be on (a) and (b).

## (a) Correctness & bugs — primary
- [ ] Logic: off-by-one, inverted conditions, wrong default, mutated shared state, missing await/sync
- [ ] Edge cases: empty/None/NaN, single rank, world-size=1, first/last iter, resume-from-checkpoint, eval-only paths
- [ ] Distributed: collectives called on every rank, device placement, deterministic ordering, no races on Ray actors/channels
- [ ] Lifecycle & resources: GPU/file/actor cleanup, no leaks, no double-init/double-free
- [ ] Numerical: dtype, in-place on grad-required tensors, unsafe casts, loss-mask correctness
- [ ] Error handling: meaningful messages, validate at boundaries, no silent except
- [ ] Behavior parity vs `origin/main` on refactored code paths (read both side-by-side)

## (b) Design & pattern consistency vs `origin/main` — primary
- [ ] Closest sibling identified in `origin/main`; new code matches its structure/naming
- [ ] Registry decorators used (`register_advantage` / `register_policy_loss` / `register_reward`)
- [ ] `SupportedModel` / `SupportedEnvType` / `get_env_cls()` / `validate_cfg` updated where needed
- [ ] Worker subclasses `Worker`, uses `self.log_*`, launched via `create_group(...).launch(...)`
- [ ] Embodied policy extends `BasePolicy`; no reimplemented base behavior
- [ ] YAML config copied from a sibling; no dynamic values; fields read-only in code
- [ ] No duplication of helpers already in `rlinf/utils/` (cite the existing helper)
- [ ] Simpler approach used when the codebase already has one
- [ ] No hardcoded machine paths, sleep-based sync, or monkey-patches

## (c) Code ↔ docs consistency
- [ ] If docs changed, follow the [docs-check skill](../docs-check/SKILL.md) (code↔docs cross-check + EN↔ZH parity)
- [ ] Every config key / CLI flag / env var / path / supported name mentioned in changed docs exists in `origin/main` + PR
- [ ] Public-facing additions/renames/removals in code are reflected in BOTH `docs/source-en/` AND `docs/source-zh/`
- [ ] Changed `examples/embodiment/config/*.yaml` model-weight paths (`model_path`/`lora_path`/`backbone_model_path`/`wan_wm_hf_ckpt_path`): basename == repo in the `# https://huggingface.co/...` comment; repo resolves on HF (`/api/models/<org>/<repo>` → 200, non-redirecting `id`); and matches the model the env recipe doc prescribes for that env+suite+model family (e.g. LIBERO spatial/object/goal + π₀ → `RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT`); no base-vs-LoRA mixup or casing drift
- [ ] EN/ZH paired pages agree: commands, paths, keys, claims, numbers, structure
- [ ] No duplicated/missing/conflicting paragraphs between EN and ZH (or justified)
- [ ] Style aligned with sibling docs (section titles/order, code blocks, table/link style)
- [ ] Each docs finding gives concrete wording/structure fix and file references

## (d) Tests & CI
- [ ] If the install script changed (`requirements/install.sh`, `requirements/embodied/`, `docker/Dockerfile`), follow the [install-check skill](../install-check/SKILL.md)
- [ ] User-facing changes have unit or e2e tests
- [ ] New env/model has install-script + Docker stage + CI/e2e coverage (use add-install-docker-ci-e2e)
- [ ] New CI-relevant YAML referenced in the e2e test matrix
- [ ] Large deps (docker/models/datasets) → maintainer ping noted

## (e) Style & metadata — only flag real issues
- [ ] Google Python Style; pre-commit clean
- [ ] Public classes/methods have Google-style docstrings; param type hints; return type when needed
- [ ] Assertions/exceptions have meaningful messages
- [ ] Logging used (no `print`)
- [ ] Newly-added files (`git diff --name-status origin/main...<pr-head>`, status `A`) carry the `# Copyright <YEAR> The RLinf Authors.` header with `<YEAR>` = current year (`date +%Y`); third-party-vendored files keep their upstream copyright line
- [ ] Every commit `Signed-off-by`; Conventional Commits subject
- [ ] PR title in Conventional Commits format; PR Description + Checklist sections filled
