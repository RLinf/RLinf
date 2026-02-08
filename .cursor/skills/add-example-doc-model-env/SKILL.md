---
name: add-example-doc-model-env
description: "Adds example documentation for a new model or environment in RLinf (RST pages in the docs gallery for both English and Chinese). Use when adding a new embodied or reasoning example, or new benchmark (e.g. LIBERO, ManiSkill)."
---

# Add Example Doc to a Model or Environment

Use this skill when adding example documentation for a new **model** (e.g. π₀, GR00T, OpenVLA) or **environment** (e.g. LIBERO, ManiSkill, MetaWorld) in RLinf. Documentation is added for both **English** and **Chinese**.

---

## Steps

1. **Create the English RST file**  
   Path: `docs/source-en/rst_source/examples/<name>.rst` (e.g. `dexbotic.rst`, `libero.rst`).  
   Follow the structure of existing examples (see [reference.md](reference.md)).

2. **Register in the English example index**  
   Edit `docs/source-en/rst_source/examples/index.rst`:
   - Add an entry in the `.. toctree::` at the bottom (e.g. `dexbotic`).
   - Optionally add a gallery card in the right category (Embodied / Reasoning / Agent): same HTML block pattern as existing cards (image, link to the new doc, short title + description).

3. **If it is an embodied evaluation environment**  
   In `docs/source-en/rst_source/start/vla-eval.rst`, add a line in the “List of currently supported evaluation environments”:
   - `:doc:\`Display Name <../examples/<name>\``

4. **Create the Chinese RST file**  
   Path: `docs/source-zh/rst_source/examples/<name>.rst`.  
   Mirror the English content (same structure and sections). Use existing pairs (e.g. `libero.rst` in both `source-en` and `source-zh`) as reference.

5. **Register in the Chinese example index**  
   Edit `docs/source-zh/rst_source/examples/index.rst`: add the same toctree entry and, if added for English, a matching gallery card. If the project has a Chinese vla-eval or start page that lists environments, add the new example there too.

6. **Update README.md**  
   In the "What's NEW!" section at the top, add a new dated bullet (e.g. `- [YYYY/MM] 🔥 ... Doc: [Display Title](https://rlinf.readthedocs.io/en/latest/rst_source/examples/<name>.html).`). If the example is a simulator, model, or feature that appears in the Key Features table, add a corresponding list item in the right column (see [reference.md](reference.md)).

7. **Update README.zh-CN.md**  
   In the "最新动态" section, add the same news item in Chinese with the doc link using `/zh-cn/` in the URL. If you added a feature list entry in README.md, add the same entry in README.zh-CN.md (Chinese display text, `zh-cn` in the link).

### RST structure (concise)

- Title (overbar length matches title).
- Optional HuggingFace icon block (copy from libero.rst).
- Short intro (what this example does, which model + env).
- **Environment**: env name, task, observation/action space, task description format, data shapes.
- **Algorithm**: PPO/GRPO/etc. and model architecture notes.
- **Dependency Installation**: clone, install (Docker or pip).
- **Quick Start**: exact commands and key YAML/config snippets.
- **Evaluation** (if applicable): eval command and config notes.

Use existing examples (e.g. `libero.rst`, `pi0.rst`) as templates; see [reference.md](reference.md) for a minimal template.

---

## Checklist

- [ ] English RST created: `docs/source-en/rst_source/examples/<name>.rst`.
- [ ] English index updated: `docs/source-en/rst_source/examples/index.rst` (toctree; optional gallery card).
- [ ] If embodied eval env: `docs/source-en/rst_source/start/vla-eval.rst` updated.
- [ ] Chinese RST created: `docs/source-zh/rst_source/examples/<name>.rst`.
- [ ] Chinese index updated: `docs/source-zh/rst_source/examples/index.rst` (toctree; gallery card if added for EN).
- [ ] If embodied eval env: update Chinese start/vla-eval page if it lists environments.
- [ ] README.md updated: new bullet in "What's NEW!" and, if applicable, entry in Key Features table.
- [ ] README.zh-CN.md updated: new bullet in "最新动态" and, if applicable, entry in 核心特性 table.
