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
2. Verify doc-to-code correctness:
   - Commands exist and are runnable in principle.
   - Script/module paths in docs exist.
   - Config keys and values match real code/config names.
   - Model/env names match `SupportedModel` and `SupportedEnvType` string values.
3. Verify doc-to-doc consistency within one language:
   - Terminology is consistent across start/tutorials/examples/API pages.
   - New page is linked in the correct index/toctree.
   - No conflicting instructions between related pages.
   - Internal doc links use stable `:doc:`/relative links, not hardcoded ReadTheDocs URLs.
4. Verify EN-ZH parity:
   - Same topic coverage and section structure.
   - Same commands, config keys, and model/env identifiers.
   - Translations preserve technical meaning (do not rename code symbols).
   - Corresponding EN/ZH pages use equivalent stable internal links.
5. Report findings with severity and concrete fixes.

## Severity Rules

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

## Additional Resource

- Detailed checklist and paths: [reference.md](reference.md)
