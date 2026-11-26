---
name: docs
description: "RLinf documentation in docs/source-en, docs/source-zh, README.md, and README.zh-CN.md: write, edit, or check a page in English and Chinese. Holds the writing rules split by page type (recipe, guide, concept, extending, reference, index, publication) and by language, the mistakes agents make in docs, templates, and the check for the build, markup, names, EN/ZH parity, and writing. Use before creating or editing any docs page or README, including doc updates made as part of a code change, and to check docs."
paths:
  - docs/**
  - README.md
  - README.zh-CN.md
---

# Docs

RLinf's documentation rules live in [rules/](rules/), split so that a writer reads only what the page in hand needs: the rules for every page, the rules for its type, and the rules for each language. This file says which to read, how to write a page, and how to check it.

## Write

1. **Classify the page** by where it lives ([rules/base.md](rules/base.md) has the table), and open a current page of the same type in the same section, or its template in [templates/](templates/).
2. **Read** [rules/base.md](rules/base.md), the file for the page type, [rules/english.md](rules/english.md) and [rules/chinese.md](rules/chinese.md) for the languages you write, and [rules/glossary.md](rules/glossary.md) for terms.
3. **Check the facts** the page will state against the code: every command, config key, path, and model or env name.
4. **Plan the page**: its type, its reader, the result the reader reaches, and one line per section saying what it answers.
5. **Write the English and the Chinese page from the plan**, each in its own language. Write the Chinese from the plan, not by translating the English sentences.
6. **Place the page** in one gallery or section index and in both toctrees ([rules/index.md](rules/index.md)), and add a README highlight for a feature worth announcing ([rules/readme.md](rules/readme.md)).
7. **Go through [rules/mistakes.md](rules/mistakes.md).**
8. **Run a simplicity and language pass.** Give a fresh agent, which has seen nothing else, only the changed pages, [rules/base.md](rules/base.md), and the language file for each. Ask it first to cut anything the reader does not need for the main path (long-winded explanations of details, redundant paragraphs, repeated facts, restated YAML defaults, defensive caveats, code citations, complex commands, dense inline parameter dumps, and repeated negative-and-positive framing such as "this is not A, not B, but C"), and then to rewrite any remaining sentence that reads as translated, padded, or built on code names. Apply the cuts and rewrites that hold. A writer cannot hear the length or cadence of their own prose.
9. **Run the check** below and fix what it reports.

### A recipe page for a new model or env

- Write `docs/source-en/rst_source/examples/embodied/<name>.rst` and the same path under `docs/source-zh/`, from [templates/recipe.rst](templates/recipe.rst) and [templates/recipe.zh.rst](templates/recipe.zh.rst). Agentic recipes live under `examples/agentic/`, system demos under `examples/system/`.
- Register the page in the gallery index that owns it ([rules/index.md § Where a page goes](rules/index.md#where-a-page-goes)), with its card and its entry in the hidden toctree, in both languages.
- When the env has a benchmark evaluation flow, link it from the matching page under `evaluations/`.

### A publication page

Write `resources/publications/<slug>.rst` in both languages from [templates/publication.rst](templates/publication.rst), following [rules/publication.md](rules/publication.md), then add its row to the `list-table` in `publications/index.rst` and its slug to the hidden toctree, in both languages.

## Check

```bash
python3 .agents/skills/docs/check.py              # docs changed against main
python3 .agents/skills/docs/check.py FILE ...     # specific pages
python3 .agents/skills/docs/check.py --no-build   # skip the Sphinx build while iterating
```

Page findings count only on the lines the change adds, because many existing pages predate the current rules; `--all-lines` reports every line of the selected pages, for a cleanup pass. The build always covers both trees. The build and the markup check run in the docs virtualenv: `$DOCS_VENV`, or `<repo>/.docs-venv`, which the script creates from `requirements/docs/requirements.txt` on first use. Findings use the shared format of [../README.md](../README.md#findings):

| Rule | Severity | Reports |
|---|---|---|
| `docs/build` | critical | a Sphinx warning in either language; both Read the Docs projects build with `fail_on_warning`, independently ([reference.md](reference.md#the-build-gate)) |
| `docs/markup` | critical or major | inline markup next to CJK text that breaks the build (critical) or silently renders wrong (major) ([reference.md](reference.md#inline-markup-next-to-cjk-punctuation)) |
| `docs/symbol` | major | a name in code font, on a Concepts, Extending, or Reference page, that nothing in the repository defines |
| `docs/code-name` | major or minor | RLinf's internal classes, functions, or source paths in code font on a recipe or guide page (major) or a concept page (minor) |
| `docs/math` | major | an equation on a recipe or guide page |
| `docs/structure` | major | an Agent & LLM recipe (or new embodied/eval recipe) missing `Overview`/`概览` or `Setup and Run`/`安装与运行`, using legacy `一键上手`, keeping legacy split H2s (`Run It`/`运行`, `Run on Different Hardware Backends`/`在不同硬件后端上运行`), or keeping generic theory sections (`Algorithm`, `Model Architecture`, `GRPO 原理`, `技术细节`, etc.) |
| `docs/rtd-link` | major | a hardcoded Read the Docs link to an RLinf page instead of `:doc:` or `:ref:` |
| `docs/image-url` | major | a broken external figure/image URL (HTTP 4xx/5xx such as 404 on `RLinf/misc`) or a redirecting `github.com/<org>/<repo>/raw/` URL instead of `raw.githubusercontent.com/...` |
| `docs/parity` | major or minor | a page, `conf.py`, or `_static` CSS/JS without its counterpart, differing toctrees, or diverged shared CSS/JS (`custom.css` / `sidebar-nav.js`) (major); a page or README changed while its counterpart did not (minor) |
| `docs/heading` | minor | a code identifier in a heading outside Reference and Extending |
| `docs/heading-length` | minor | an overly long heading (> 48 EN chars or > 20 ZH CJK-equivalent chars) |
| `docs/en-style` | minor | a word or phrase that [rules/english.md](rules/english.md) cuts |
| `docs/zh-style` | minor | 您, 「」, 值得注意的是, 进行, 当……时, 不是/并非……而是……, half-width punctuation in a Chinese sentence, or no space between Chinese and English or digits ([rules/chinese.md](rules/chinese.md)) |
| `docs/zh-term` | minor | translated RL/system terms (`优势函数`, `策略损失`, `损失函数`, `规则奖励`, `奖励模型`, `新增环境`) in Chinese headings ([rules/glossary.md](rules/glossary.md)) |
| `docs/zh-hard-wrap` | minor | Chinese prose wrapped inside a paragraph or list item, which renders as a space |
| `docs/command-placeholder` | major | an `install.sh` or `hf download` command containing unrunnable `<...>` placeholders, or `/path/to/...` placeholders inside recipe/eval bash commands |
| `docs/command-launcher` | major | bare `python examples/embodiment/train_embodied_agent.py` in a recipe instead of `bash examples/embodiment/run_embodiment.sh <config>` |
| `docs/command-path` | minor | hardcoded root-level `/data/...` path in a recipe/eval bash command (prefer repo-relative `./data/...`) |
| `docs/command-legacy` | major | `huggingface-cli download`, `pip install huggingface-hub`, `git lfs`, `git clone https://huggingface.co/`, or `CUDA_VISIBLE_DEVICES` (on recipes, eval guides, and `guides/`) |
| `docs/command-cd` | major | `cd` immediately before `hf download`, or `--local-dir .` / `--local-dir /path/to/...` |
| `docs/command-tabset` | major | an `install.sh` or `hf download` block on `examples/` or `evaluations/` not wrapped in `.. tab-set::`, or a hollow `.. tab-set::` that wraps only install while leaving launch commands outside |
| `docs/duplicate-commands` | minor | identical multi-line setup or launch bash blocks duplicated across multiple changed same-language recipe pages (link to the primary recipe or extract a `_*.rst` partial) |
| `docs/command-bloat` | major | mega bash blocks combining `git clone` + launch script, >= 16 command lines outside `.. dropdown::`, shell argument arrays, or >= 6 Hydra overrides on one launch line |
| `docs/nested-tabset` | major | nested `.. tab-set::` directives (stacked tab bars); flatten into one single-level `.. tab-set::` |
| `docs/tab-title` | major | `.. tab-item::` title > 48 chars or concatenating `Model + Benchmark` (e.g. `OpenPI + RoboCasa`) |
| `docs/tab-parity` | major | paired EN/ZH pages with mismatched `.. tab-set::` or `.. tab-item::` counts |
| `docs/mirror-comment` | major | commented `--use-mirror`, `HF_ENDPOINT`, `hf-mirror`, or `gh-proxy` lines inside bash blocks |
| `docs/dir-tree` | major | decorative ASCII directory trees (`├──` / `└──`) outside `.. dropdown::` |
| `docs/boilerplate-yaml` | major | stale `rollout.model.model_path` alongside `actor.model.model_path`, or repeated `logger_backends:` / `video_base_dir:` YAML blocks |
| `docs/long-yaml` | minor | a YAML block with ≥ 20 lines outside `.. dropdown::` on `start/`, `examples/`, `evaluations/`, or `guides/` pages |
| `docs/yaml-edit-friction` | minor | a recipe telling readers to manually edit `model_path` / `train_data_paths` in YAML without showing a copy-pasteable Hydra CLI override |
| `docs/eval-command` | major | `env.eval.total_num_envs=10` (crashes on 4/8 GPUs), passing a training-only YAML to `evaluations/run_eval.sh`, or claiming arbitrary training configs work via eval fallback |
| `docs/multinode-trap` | major | launching a YAML whose `cluster.num_nodes > 1` without stating the node count or giving a `cluster.num_nodes=1` quick-start override |
| `docs/local-asset` | major | a new image/diagram file added under `docs/**/_static/` instead of hosted in `RLinf/misc` under `pic/` |
| `docs/no-todo` | major | unfinished `TODO`, `TBD`, or `FIXME` marker in published docs (prose, table cells, or comments) |
| `docs/jargon` | major | testing jargon (`冒烟测试` / `smoke test` / `smoke check`) or informal ZH shorthand (`采数` / `数采` / `遥操`) |
| `docs/long-paragraph` | minor | wall-of-text paragraph (> 120 EN words or > 260 ZH chars) on a recipe or guide page |
| `docs/inline-code-density` | minor | single prose paragraph packing >= 6 inline code literals (`` ``...`` ``) on a recipe or guide page; simplify or move into a table/dropdown |
| `docs/raw-html-grid` | major | `.. raw:: html` navigation card grid or hardcoded `color: blue` (use `sphinx-design` `.. grid::`) |
| `docs/index-style` | major | non-hidden toctree on an index page, missing cards/table routing, or hidden toctree entry omitted from the page body |
| `docs/path-exists` | major | repository file path (`rlinf/`, `examples/`, `evaluations/`, `toolkits/`, `requirements/`) cited in prose or commands that does not exist on disk |
| `docs/extending-no-pytest` | major | boilerplate `tests/unit_tests/` or `pytest` unit-test section in `extending/` (use standalone local CPU verification instead) |
| `docs/extending-structure` | major | `extending/` page missing top At-a-Glance `.. list-table::` or runnable verification step |
| `docs/guide-quality` | major | ultra-thin stub guide (< 40 non-empty lines) in `guides/` |
| `docs/concept-purity` | major | `.. autoclass::` / `.. autofunction::`, ultra-thin stub (< 40 lines), or PR migration history in `concepts/` |
| `docs/algo-keys` | major | YAML key on `reference/algorithms/` that does not exist in `rlinf/` / `examples/` / `evaluations/`, or missing runnable config/recipe link |
| `docs/latex-sanity` | major | broken LaTeX subscript omission (`\mathbb{E}{`, `\pi{\theta}`) or orphan paper equation tags (`\tag{8}`) in `reference/` |
| `docs/sidebar-grouping` | major | raw leaf pages dumped into grouped top-level chapters (`Embodied AI`, `Agent & LLM`, `Guides`) or >5 entries in `Get Started` on root `index.rst` |
| `docs/sidebar-depth` | major | nested `.. toctree::` sub-index folders inside `examples/agentic/` (each Agent & LLM project must be a single-page recipe with tabs like Embodied AI) |
| `docs/alpha-order` | major | unsorted simulator, real-world robot, model, RL method, Agent & LLM sub-category, algorithm reference, Python API, or publication toctrees (`simulators_index`, `vla_wam_index`, `sft_index`, `world_model_index`, `video_models`, `methods_index`, `agentic/{math_reasoning,agents}/index`, `reference/{algorithms,api}/index`, `resources/publications/index`, and root `Real-World Robots`) |
| `docs/sibling-alignment` | major | Agent & LLM recipe (or new embodied/eval recipe) missing Overview `.. grid::` + `Prerequisites` header, `start/installation` link, `.. tab-set::`, or `Visualization and Results` / `可视化与结果` |
| `docs/box-table` | major | misaligned Unicode box-drawing table rows (e.g. `Metric Table`) whose display width differs from the top border |
| `docs/heading-consistency` | major | redundant `.. contents::` (wraps headings in underlined `a.toc-backref` links) or EN/ZH heading-level sequence mismatch |

Then read every changed page in full, in both languages, as a reader new to RLinf who has not seen its code, and inspect every command block (the script above checks build, markup, and prose, not command quality or workflow design):

- [ ] The checklist in the page type's rules file holds.
- [ ] **Readability & conciseness**: apply [rules/base.md](rules/base.md#prefer-simplicity-and-readability-over-verbosity). Flag unreadable/stiff prose, machine-translated or sentence-by-sentence Chinese, long-winded detail explanations, walls of prose, internal code citations, repeated facts, restated YAML defaults, repeated negative-and-positive framing ("not A, not B, but C" / "不是……而是……"), and pages bloatier than a clean `main` sibling covering the same scope.
- [ ] **Commands & sibling alignment (recipes/guides)**: compare against the closest model- or domain-family siblings on `main` (not only the gallery). Shared setup links to existing family pages or partials; commands stay minimal and YAML-first (no shell plumbing, long CLI overrides, inline prep scripts, or env-var workarounds).
- [ ] **Docs ↔ code accuracy**: every config key, flag, env var, path, and model/env name matches `main` plus the change; public additions, renames, or removals update all affected EN/ZH pages.
- [ ] **Cross-references**: a sentence that sends the reader to another page for a step ("as in", "see") points to a page that covers that step for the same model, config, and workflow. Open the target page; a link that resolves is not enough.
- [ ] **Code examples**: code the reader is meant to copy (a subclass, a hook, a config snippet) runs as shown. When it needs a cluster, trace each of its branches against the code of the change, including the error and early-exit branches, and confirm each one ends in a state RLinf accepts.
- [ ] **Parity, assets & README**: paired EN/ZH pages agree on sections, commands, keys, claims, and numbers; changed assets pass [reference.md § Asset references](reference.md#asset-references); noteworthy features get a README highlight ([rules/readme.md](rules/readme.md)).
- [ ] None of the mistakes in [rules/mistakes.md](rules/mistakes.md) remains.

Report each changed page with its type and result (pass or findings). Rate a page `major` when it fails any of the bullets above as a whole (hard to follow or run, bloated/unscannable, machine-translated Chinese, cites internals instead of behavior, complex/non-standard commands, duplicated setup, or broken parity/placement); rate isolated wording issues `minor`. Quote the line, name the rule, and give the rewrite.
