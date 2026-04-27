# Docs Check Reference

Detailed checklists for doc-code cross-check.

---

## Source of truth for models and envs

- **Model types**: Read `SupportedModel` in `rlinf/config.py` – docs must use the string values (e.g. `openpi`, `openvla_oft`, `gr00t`). Do not hardcode lists; verify against the code.
- **Env types**: Read `SupportedEnvType` in `rlinf/envs/__init__.py` – docs must use the string values (e.g. `maniskill`, `libero`). Verify against the code.

---

## Doc layout

| Area | EN path | ZH path |
|------|---------|---------|
| Root index | `docs/source-en/index.rst` | `docs/source-zh/index.rst` |
| Start | `docs/source-en/rst_source/start/` | `docs/source-zh/rst_source/start/` |
| Tutorials | `docs/source-en/rst_source/tutorials/` | `docs/source-zh/rst_source/tutorials/` |
| Examples (embodied) | `docs/source-en/rst_source/examples/embodied/` | `docs/source-zh/rst_source/examples/embodied/` |
| Examples (agentic) | `docs/source-en/rst_source/examples/agentic/` | `docs/source-zh/rst_source/examples/agentic/` |
| APIs | `docs/source-en/rst_source/apis/` | `docs/source-zh/rst_source/apis/` |
| FAQ | `docs/source-en/rst_source/faq.rst` | `docs/source-zh/rst_source/faq.rst` |

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
