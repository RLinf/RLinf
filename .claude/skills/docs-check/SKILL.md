---
name: docs-check
description: Cross-checks RLinf documentation against code and other docs, including English-Chinese parity checks. Use when adding or editing docs, reviewing doc PRs, validating commands/config keys/model-env names, or ensuring EN and ZH docs stay consistent.
---

# Docs Check

## Quick Start

Use this skill when documentation changes may introduce mismatches with:

- Code and config source of truth
- Other existing docs in the same section
- English and corresponding Chinese docs

Always read `reference.md` first, then run the workflow below.

Two harnesses in this folder do the mechanical part; run them before reasoning
about content:

```bash
# Does the page still build? Both Read the Docs projects use fail_on_warning.
python3 .claude/skills/docs-check/build_docs.py            # en + zh

# Does CJK punctuation break inline markup, loudly or silently?
python3 .claude/skills/docs-check/check_rst_markup.py      # needs docutils
```

## Inputs

Collect these inputs before reviewing:

- Changed doc files (or target docs to validate)
- Corresponding EN and ZH files for the same topic
- Related code/config files referenced by the docs

If scope is unclear, default to checking:

- `docs/source-en/` and `docs/source-zh/` counterparts
- `rlinf/config.py` (`SupportedModel`)
- `rlinf/envs/__init__.py` (`SupportedEnvType`)
- Referenced scripts under `examples/`, `toolkits/`, `ray_utils/`, and `requirements/`

## Workflow

1. Read `reference.md` and extract the relevant checklist items.
2. Run the build harness for **both** languages and fix every warning:
   - `python3 .claude/skills/docs-check/build_docs.py` (add `--lang zh` to
     iterate faster on a Chinese-only failure).
   - The two Read the Docs projects build independently with
     `fail_on_warning: true`, so an English-clean page can still turn
     `docs/readthedocs.org:rlinf-cn` red. Never conclude "docs are fine"
     from one language.
   - Then `python3 .claude/skills/docs-check/check_rst_markup.py` for the
     inline-markup defects that render wrong *without* warning.
3. Verify doc-to-code correctness:
   - Commands exist and are runnable in principle.
   - Script/module paths in docs exist.
   - Config keys and values match real code/config names.
   - Model/env names match `SupportedModel` and `SupportedEnvType` string values.
4. Verify doc-to-doc consistency within one language:
   - Terminology is consistent across start/tutorials/examples/API pages.
   - New page is linked in the correct index/toctree.
   - No conflicting instructions between related pages.
   - Internal doc links use stable `:doc:`/relative links, not hardcoded ReadTheDocs URLs.
5. Verify EN-ZH parity:
   - Same topic coverage and section structure.
   - Same commands, config keys, and model/env identifiers.
   - Translations preserve technical meaning (do not rename code symbols).
   - Corresponding EN/ZH pages use equivalent stable internal links.
6. Report findings with severity and concrete fixes.

## Severity Rules

- `Critical`: A Sphinx warning in either language — Read the Docs fails the
  build, so the page does not ship at all.
- `Critical`: Wrong command/path/key/value that can break user workflow.
- `Major`: Inconsistent docs that likely mislead users.
- `Minor`: Wording/terminology drift without immediate breakage.

Prefer actionable findings with exact file paths and corrected values.

Hardcoded ReadTheDocs links to RLinf docs should be reported as at least `Major`.

## Output Format

Use this format when reporting results:

```markdown
## Docs Check Findings

- Critical: <issue>, in `<path>`
  - Why: <impact>
  - Fix: <specific correction>

- Major: <issue>, in `<path>`
  - Why: <impact>
  - Fix: <specific correction>

- Minor: <issue>, in `<path>`
  - Why: <impact>
  - Fix: <specific correction>

## Verified

- <what was checked and confirmed>
```

If no issues are found, explicitly state:

`No doc-code or EN-ZH consistency issues found in checked scope.`

## Guardrails

- Do not invent model/env/config names; verify against source files.
- Do not change code to match incorrect docs unless explicitly requested.
- Keep EN and ZH technical tokens identical where applicable (paths, CLI flags, keys, enum values).
- When uncertain, flag as an assumption and request confirmation.
- Do not keep RLinf internal links as hardcoded `readthedocs.io/.../rst_source/...` URLs; convert to `:doc:` or relative internal links.

## Quick Detection

Use this regex scan to detect unstable hardcoded RLinf docs links:

- `readthedocs\.io/(en|zh-cn)/latest/rst_source/`

Chinese pages: `` `` `` or `**` sitting directly against `（`, `《` or a CJK
character is the single most common way to break `rlinf-cn`. Grep for
```` ``（ ```` and ```` **（ ```` as a first pass, then run
`check_rst_markup.py` for the full rule.

## Additional Resource

- Detailed checklist and paths: [reference.md](reference.md)
- Local Read the Docs gate for both languages: [build_docs.py](build_docs.py)
- Inline-markup checker: [check_rst_markup.py](check_rst_markup.py)
