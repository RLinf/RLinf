# Docs Check Reference

Detailed checklists for doc-code cross-check.

---

## Source of truth for models and envs

- **Model types**: Read `SupportedModel` in `rlinf/config.py` – docs must use the string values (e.g. `openpi`, `openvla_oft`, `gr00t`). Do not hardcode lists; verify against the code.
- **Env types**: Read `SupportedEnvType` in `rlinf/envs/__init__.py` – docs must use the string values (e.g. `maniskill`, `libero`). Verify against the code.

---

## Doc layout

Docs use a sidebar-only IA with eight top-level axes (Get Started · Examples ·
Evaluation · Concepts · Guides · Reference · Extending · Resources). Each axis
owns its pages directly; the legacy `tutorials/`, `apis/`, `blog/`, and
`publications/` trees were removed and their pages relocated into the axes.

| Area | EN path | ZH path |
|------|---------|---------|
| Root index | `docs/source-en/index.rst` | `docs/source-zh/index.rst` |
| Get Started | `docs/source-en/rst_source/start/` | `docs/source-zh/rst_source/start/` |
| Examples (embodied) | `docs/source-en/rst_source/examples/embodied/` | `docs/source-zh/rst_source/examples/embodied/` |
| Examples (agentic) | `docs/source-en/rst_source/examples/agentic/` | `docs/source-zh/rst_source/examples/agentic/` |
| Evaluation | `docs/source-en/rst_source/evaluations/` | `docs/source-zh/rst_source/evaluations/` |
| Concepts | `docs/source-en/rst_source/concepts/` | `docs/source-zh/rst_source/concepts/` |
| Guides | `docs/source-en/rst_source/guides/` | `docs/source-zh/rst_source/guides/` |
| Reference (API) | `docs/source-en/rst_source/reference/api/` | `docs/source-zh/rst_source/reference/api/` |
| Reference (algorithms) | `docs/source-en/rst_source/reference/algorithms/` | `docs/source-zh/rst_source/reference/algorithms/` |
| Extending | `docs/source-en/rst_source/extending/` | `docs/source-zh/rst_source/extending/` |
| Resources (blog, publications, release, FAQ) | `docs/source-en/rst_source/resources/` | `docs/source-zh/rst_source/resources/` |

---

## Checklist summary

### Doc vs Code
- [ ] Every config name in docs exists under `examples/embodiment/config/` or `env/`
- [ ] Model types in docs match `SupportedModel` string values in `rlinf/config.py`
- [ ] Env types in docs match `SupportedEnvType` values in `rlinf/envs/__init__.py`
- [ ] Scripts referenced (e.g. `run_embodiment.sh`, `train_embodied_agent.py`) exist
- [ ] Python paths (e.g. `rlinf/models/embodiment/openpi/dataconfig/__init__.py`) exist

### Doc structure
- [ ] Root toctree in EN and ZH matches
- [ ] Category indexes (e.g. embodied/index.rst) list the same toctree entries
- [ ] Every EN RST file has a corresponding ZH file at the same relative path
- [ ] Internal RLinf doc links use `:doc:`/relative links (no hardcoded ReadTheDocs `.../rst_source/...` URLs)

### EN vs ZH
- [ ] Same section headings (translated)
- [ ] Config names, YAML keys, and commands identical in both
- [ ] Technical terms (PPO, GRPO, SFT, model names) consistent
- [ ] Internal and external links correct
- [ ] EN and ZH use equivalent stable internal links for counterpart sections
