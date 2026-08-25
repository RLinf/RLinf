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

## The build gate

Both languages are separate Read the Docs projects, configured by
`docs/source-en/.readthedocs.yaml` and `docs/source-zh/.readthedocs.yaml`, and
both set `fail_on_warning: true`. A single Sphinx warning turns
`docs/readthedocs.org:rlinf` or `docs/readthedocs.org:rlinf-cn` red. They build
independently, so "the English build is green" says nothing about the Chinese
one — always reproduce both with `build_docs.py`.

To read a failed Read the Docs check without leaving the terminal:

```bash
gh api repos/RLinf/RLinf/commits/<sha>/status \
  --jq '.statuses[] | "\(.context) \(.state) \(.target_url)"'
```

---

## Inline markup next to CJK punctuation

docutils only recognises inline markup when the *start*-string is preceded by
whitespace, an opener or a delimiter, and the *end*-string is followed by
whitespace, a closer or a delimiter. Full-width punctuation splits across those
classes:

| Character | docutils class | Before markup | After markup |
|-----------|----------------|---------------|--------------|
| `（` `《` `“` | opener | ok | **rejected** |
| `）` `》` `”` | closer | **rejected** | ok |
| `，` `、` `：` `；` `。` `！` `？` `—` | delimiter | ok | ok |
| any CJK ideograph | none | **rejected** | **rejected** |

Two distinct failures follow, and only the first is visible in CI:

1. **Build break.** No later end-string candidate satisfies the rule, so Sphinx
   emits `WARNING: Inline literal start-string without end-string. [docutils]`
   and the CN build fails.

   ```rst
   导出为 ``MUJOCO_EGL_DEVICE_ID``（MuJoCo）和 ``EGL_DEVICE_ID``（其他）。
   ```

2. **Silent mis-render.** A *later* ` `` ` in the same paragraph does satisfy
   the rule, so docutils closes the literal there and swallows the prose in
   between into the code span. No warning, wrong page.

   ```rst
   镜像提供两者：``reason``（SGLang，默认激活）与 ``reason-vllm``。
   ..                 ^ renders as one literal: "reason``（SGLang，默认激活）与 ``reason-vllm"
   ```

   The same happens to `**粗体**` written directly against a Chinese character:
   docutils sees no markup and prints the asterisks.

The fix in every case is an escaped space — a backslash followed by a space —
between the markup and the adjacent character. It suppresses the space in the
output, so Chinese typography is unaffected:

```rst
导出为 ``MUJOCO_EGL_DEVICE_ID``\ （MuJoCo）和 ``EGL_DEVICE_ID``\ （其他）。
CUDA 设备 0 通常\ **并不是** EGL 设备 0。
```

`check_rst_markup.py` reports both classes: `ERROR` for the build break,
`WARNING` for the silent mis-render. Only the first blocks a PR, but a doc PR
touching a Chinese page should leave no new warnings either.

Note that inline markup does **not** nest — `` **stuck in ``reach``** `` is a
strong containing literal backtick characters, not a nested literal, and is
perfectly legal.

---

## Checklist summary

### Build
- [ ] `python3 .cursor/skills/docs-check/build_docs.py` is clean for **both** `en` and `zh`
- [ ] `python3 .cursor/skills/docs-check/check_rst_markup.py` reports no new findings
- [ ] Changed ZH pages have no ` `` ` or `**` touching `（`, `《` or a CJK character

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
