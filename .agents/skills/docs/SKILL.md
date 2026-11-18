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
8. **Run a language pass.** Give a fresh agent, which has seen nothing else, only the changed pages and the language file for each, and ask it to mark every sentence that reads as translated, padded, or built on code names, with its rewrite. Apply the rewrites that hold. A writer cannot hear the cadence of their own prose.
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
| `docs/structure` | major | a new gallery recipe without Installation and Run It, or with a generic Algorithm or Environment section |
| `docs/rtd-link` | major | a hardcoded Read the Docs link to an RLinf page instead of `:doc:` or `:ref:` |
| `docs/parity` | major or minor | a page without its counterpart, or toctrees that differ (major); a page or README changed while its counterpart did not (minor) |
| `docs/heading` | minor | a code identifier in a heading outside Reference and Extending |
| `docs/en-style` | minor | a word or phrase that [rules/english.md](rules/english.md) cuts |
| `docs/zh-style` | minor | 您, 「」, 值得注意的是, 进行, 当……时, half-width punctuation in a Chinese sentence, or no space between Chinese and English or digits ([rules/chinese.md](rules/chinese.md)) |
| `docs/zh-hard-wrap` | minor | Chinese prose wrapped inside a paragraph or list item, which renders as a space |

Then read every changed page in full, in both languages, as a reader new to RLinf who has not seen its code, and check:

- [ ] The checklist in the page type's rules file holds.
- [ ] Docs to code: every config key, CLI flag, env var, path, and model or env name exists on `main` plus the change.
- [ ] Code to docs: a change that adds, removes, or renames a public config key, model, env, runner, script, env var, or feature updates the pages that describe it. Name the pages under `docs/source-en/` and `docs/source-zh/` that still need it.
- [ ] Parity: paired pages agree on commands, paths, env vars, config keys, supported models, envs, and algorithms, capability claims, reported numbers, and section order.
- [ ] README highlight for a feature worth announcing ([rules/readme.md](rules/readme.md)).
- [ ] Model and dataset references in changed docs and configs pass [reference.md § Asset references](reference.md#asset-references).
- [ ] None of the mistakes in [rules/mistakes.md](rules/mistakes.md) remains, and each page reads as written in its own language.

Report every changed page with its type and its result, pass or the findings, so the reader sees which pages were checked. A page fails as a whole (major) when a first-time reader cannot tell what the feature does or how to run it, it explains code instead of behavior, it breaks the rules of its type, it sits in the wrong place in the tree, or the Chinese page tracks the English one sentence for sentence. A line or two of wording is minor. Each finding quotes the line, names the rule it breaks, and gives the rewrite.
