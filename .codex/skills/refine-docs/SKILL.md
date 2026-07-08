---
name: refine-docs
description: Refine, rewrite, or write an RLinf documentation page or section so it conforms to the docs style guide (docs/STYLE_GUIDE.md) — voice, information architecture, card/table index pages, the recipe-page template, navigation labels, reuse, and EN/ZH parity. Use when improving an existing doc page, drafting a new one, or doing a style/structure pass. For pure doc-to-code and EN/ZH correctness checks, use the docs-check skill instead (the two are complementary).
---

# Refine RLinf Docs

Bring a documentation page (or a whole section) up to the RLinf documentation
**style guide**. Use this when writing a new page, rewriting an existing one, or
doing a style/structure pass.

The single source of truth is **`docs/STYLE_GUIDE.md`** — read it first and apply
it. This skill is the operating procedure; the style guide holds the exact rules.
When the two disagree, the style guide wins.

## When to use

- Editing/improving an existing page, or drafting a new one.
- A "make this page match our docs style" / "clean up these docs" request.
- Pair with the **docs-check** skill: `refine-docs` covers voice, structure, and
  style; `docs-check` covers facts, code references, and EN/ZH parity.

## Workflow

1. **Read `docs/STYLE_GUIDE.md`.** It is authoritative.
2. **Find both language files.** Every page exists at `docs/source-en/...` and
   `docs/source-zh/...`. Refine **both in the same pass** and keep them in parity.
3. **Classify the page**, then apply the matching part of the guide:
   - **Landing / section / sub-section index** → cards or tables + a `:hidden:`
     toctree; one-line outcome; routed list. Never body bullet lists.
   - **Recipe / example page** (env, model, algorithm, SFT, robot) → the page
     anatomy: figure + intro → `Overview` (4 aligned cards) → `Tasks` +
     `Observation and Action` tables → `Installation` → `Download the Model` →
     `Run It` → `Visualization and Results`. Use the standard section names and
     the aligned card schema for that gallery subsection.
   - **Concept / guide / reference / extending prose page** → outcome-first intro,
     short sections, link out instead of inlining reference material.
4. **Apply the voice rules** to every paragraph: second person, imperative,
   outcome first, no throat-clearing, short sentences, annotate non-trivial
   commands ("What this does: 1… 2…").
5. **Fix structure and labels:** Title Case headings + standard names, one H1 per
   page, bare nav captions, cards/tables instead of bullet walls, footguns in a
   `warning`, correct axis ownership/placement.
6. **De-duplicate:** link to the canonical Reference / Evaluation page instead of
   re-explaining; if identical prose/commands repeat across 3+ pages, extract an
   underscore include partial (`_name.rst`).
7. **Keep EN ↔ ZH parity:** same structure and (translated) headings; identical,
   untranslated code identifiers (config keys, CLI flags, env/model names); stable
   `:doc:` / `:ref:` links (no hardcoded ReadTheDocs URLs); never glue `**bold**`
   directly between CJK characters.
8. **Verify (the gate)** — see below.

## Quick checklists by page type

**Any page**
- [ ] Opens with the outcome, second person, no throat-clearing.
- [ ] One H1; Title Case headings; standard section names where applicable.
- [ ] EN and ZH updated together; code tokens identical; no CJK-glued `**bold**`.
- [ ] Reference material linked, not inlined; repeated blocks factored into partials.

**Index / landing page**
- [ ] Body uses a card grid or `list-table`, not bullets or bare `:doc:` lists.
- [ ] `.. toctree::` is `:hidden:` and drives nav/order.
- [ ] One-line purpose ("Pick this when…") before the cards/tables.

**Recipe / example page**
- [ ] Credited figure + one-paragraph intro.
- [ ] `Overview` card grid (`.. grid:: 2 4 4 4`) with the gallery's aligned schema.
- [ ] `Tasks` and `Observation and Action` are `list-table`s.
- [ ] No "Env type" card, no generic "Algorithm" section, no boilerplate VLA intro.
- [ ] Metrics/eval linked out; only "watch `env/success_once`" + a results table stay.
- [ ] Shared install / model-path tails come from `_setup_common.rst` / `_model_path.rst`.

## Gate

- Build both trees with **zero new warnings**:
  `/opt/venv/docs/bin/sphinx-build -b html docs/source-en /tmp/build-en` and the
  same for `docs/source-zh`.
- Run the **`docs-check`** skill (doc-to-code correctness + EN/ZH parity).
- Confirm: no new bullet-list index pages, no throat-clearing intros, and no
  literal `**` leaking into built ZH pages from CJK-glued bold.
