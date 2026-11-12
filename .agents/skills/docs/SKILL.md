---
name: docs
description: "RLinf documentation in docs/source-en, docs/source-zh, README.md, and README.zh-CN.md: write, edit, or check a page in English and Chinese. Covers page types and structure, voice, EN/ZH parity, recipe and publication pages, README highlights, doc-to-code accuracy, and the Read the Docs build. Use before creating or editing any docs page or README, including doc updates made as part of a code change, and to check docs."
paths:
  - docs/**
  - README.md
  - README.zh-CN.md
---

# Docs

The rules for RLinf docs are in [docs/STYLE_GUIDE.md](../../../docs/STYLE_GUIDE.md). This skill says which parts of the guide apply to the page you are writing, and its check runs the build and the mechanical checks. When this file and the guide disagree, the guide is right; fix this file in the same change.

## Write

1. Read the guide's Voice and tone and Write for a first-time reader sections, then the rules for the page type below.
2. Open both files. Every page exists at `docs/source-en/<path>` and `docs/source-zh/<path>`; write both in the same pass, each in its own language (EN ↔ ZH parity, Writing the Chinese pages).
3. Classify the page and open a current page of the same type to copy its structure:

   | Type | Guide sections | Model page |
   |---|---|---|
   | index | Section and subsection index pages, Navigation labels | the gallery index the page sits in |
   | recipe (an env, model, algorithm, SFT, or robot example) | Example / recipe page requirements, Page anatomy | `examples/embodied/opensora.rst` |
   | concept or guide | Explanatory pages, Build a continuous article | a page in the same axis |
   | extending | Guide the reader from use to internals | `extending/new_env.rst` |
   | reference | Technical accuracy is part of the prose | a page in the same reference section |
   | publication | Publication pages | `resources/publications/rlinf_user.rst` |

4. Check every name, command, config key, and path the page uses against the code (Technical accuracy is part of the prose). Model and env names are the string values of `SupportedModel` in `rlinf/config.py` and `SupportedEnvType` in `rlinf/envs/__init__.py`.
5. Place the page in one axis and one gallery index (Information architecture), and add it to the EN and ZH toctrees together.
6. Highlight a feature worth announcing in both READMEs (README highlights).
7. Run the check below.

### A recipe page for a new model or env

- Write `docs/source-en/rst_source/examples/embodied/<name>.rst` and the same path under `docs/source-zh/`. Agentic recipes live under `examples/agentic/`, system demos under `examples/system/`.
- Register the page in the gallery index that owns it (Information architecture, Category ownership): `simulators_index.rst`, `real_world_index.rst`, `vla_wam_index.rst`, `sft_index.rst`, or `methods_index.rst` under `examples/`, which list it as `embodied/<name>`. Add its card and its entry in the hidden toctree, in both languages.
- When the env has a benchmark evaluation flow, link it from the matching page under `evaluations/` (Evaluation ownership).
- A recipe that runs on a non-CUDA accelerator follows Non-CUDA hardware sections.

### A publication page

Write `resources/publications/<slug>.rst` in both languages, with a lowercase slug joined by underscores (`rlinf_vla`). Follow Publication pages, then add a row to the `list-table` in `publications/index.rst` and the slug to its hidden toctree, in both languages, at the position the list should show it.

## Check

```bash
python3 .agents/skills/docs/check.py              # docs changed against main
python3 .agents/skills/docs/check.py FILE ...     # specific pages
python3 .agents/skills/docs/check.py --no-build   # skip the Sphinx build while iterating
```

Page findings count only on the lines the change adds, because many existing pages predate rules such as the one on Chinese line wrapping; `--all-lines` reports every line of the selected pages, for a cleanup pass. The build always covers both trees. The build and the markup check run in the docs virtualenv: `$DOCS_VENV`, or `<repo>/.docs-venv`, which the script creates from `requirements/docs/requirements.txt` on first use. Findings use the shared format of [README.md](../README.md#findings):

| Rule | Severity | Reports |
|---|---|---|
| `docs/build` | critical | a Sphinx warning in either language; both Read the Docs projects build with `fail_on_warning`, independently ([reference.md](reference.md#the-build-gate)) |
| `docs/markup` | critical or major | inline markup next to CJK text that breaks the build (critical) or silently renders wrong (major) ([reference.md](reference.md#inline-markup-next-to-cjk-punctuation)) |
| `docs/symbol` | major | a name in code font, on a Concepts, Extending, or Reference page, that nothing in the repository defines |
| `docs/rtd-link` | major | a hardcoded Read the Docs link to an RLinf page instead of `:doc:` or `:ref:` |
| `docs/parity` | major or minor | a page without its counterpart in the other language, or toctrees that differ (major); a page or README changed while its counterpart did not (minor) |
| `docs/zh-hard-wrap` | minor | Chinese prose wrapped inside a paragraph or list item, which renders as a space |

Then read every changed page in full, in both languages, as a reader new to RLinf who has not seen its code, and check:

- [ ] Docs to code: every config key, CLI flag, env var, path, function or class name, and model or env name exists on `main` plus the change.
- [ ] Code to docs: a change that adds, removes, or renames a public config key, model, env, runner, script, env var, or feature updates the pages that describe it. Name the pages under `docs/source-en/` and `docs/source-zh/` that still need it.
- [ ] Parity: paired pages agree on commands, paths, env vars, config keys, supported models, envs, and algorithms, capability claims, reported numbers, and section order.
- [ ] README highlight for a feature worth announcing (README highlights).
- [ ] Model and dataset references in changed docs and configs pass [reference.md § Asset references](reference.md#asset-references).
- [ ] The page has the structure of its type, compared with the model page's section names and order, and sits in the right place in the tree.
- [ ] None of the conditions in the guide's Review gate holds: a reader learns what the feature does, how it works, what to prepare, and how to run it; the lead states the page's purpose; sections follow one line of thought; each language reads as written in it.
- [ ] Every figure and media URL returns 200 directly (Images and media).

Report every changed page with its type and its result, pass or the findings, so the reader sees which pages were checked. A page fails as a whole (major) when a first-time reader cannot tell what the feature does or how to run it, it explains code layout instead of behavior, it has no lead or no connected line of thought, it has the wrong structure for its type or the wrong place in the tree, or the Chinese page tracks the English one sentence for sentence. A line or two of wording is minor. Each finding quotes the line, names the guide section it breaks, and gives the rewrite.
