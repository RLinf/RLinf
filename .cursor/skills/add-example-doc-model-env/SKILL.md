---
name: add-example-doc-model-env
description: "Adds example documentation for a new model or environment in RLinf (RST pages in the docs gallery for both English and Chinese). Use when adding a new embodied or reasoning example, or new benchmark (e.g. LIBERO, ManiSkill)."
---

# Add Example Doc to a Model or Environment

Use this skill when adding example documentation for a new **model** (e.g. π₀, GR00T, OpenVLA) or **environment** (e.g. LIBERO, ManiSkill, MetaWorld) in RLinf. Documentation is added for both **English** and **Chinese**.

---

## Steps

1. **Create the English RST file**  
   Examples are now grouped by **category**:
   - `embodied/` – embodied RL/VLA examples (e.g. ManiSkill, LIBERO, Dexbotic, π₀, OpenSora)
   - `agentic/` – agent / tool-use / coder / math reasoning examples (e.g. SearchR1, coding_online_rl, reasoning)
   - `system/` – placement, scheduling, system demos  
   Path pattern: `docs/source-en/rst_source/examples/<category>/<name>.rst`  
   - Example (embodied): `docs/source-en/rst_source/examples/embodied/dexbotic.rst`  
   - Example (agentic): `docs/source-en/rst_source/examples/agentic/searchr1.rst`  
   Follow the structure of existing examples in the same category (see [reference.md](reference.md)).

2. **Register in the English category index**  
   Each category has its own index file (e.g. `docs/source-en/rst_source/examples/embodied/index.rst`).  
   - Edit `docs/source-en/rst_source/examples/<category>/index.rst`.
   - Add an entry for `<name>` in the hidden `.. toctree::` at the bottom (e.g. `dexbotic`).
   - Optionally add a gallery card in the correct section using the same HTML block pattern as existing cards (image, hyperlink to the rendered doc, short title + description).  
   **Note:** The top-level `examples/index.rst` now only links to the four category indexes and usually does not need to be changed when adding a single example.

3. **If it is an embodied evaluation environment**  
   In `docs/source-en/rst_source/start/vla-eval.rst`, add a line in the “List of currently supported evaluation environments”:
   - For embodied examples: `:doc:\`Display Name <../examples/embodied/<name>\``
   - For other categories, follow the existing pattern in that file and mirror the relative path used there.

4. **Create the Chinese RST file**  
   Use the same `<category>` and `<name>` as in English:  
   Path: `docs/source-zh/rst_source/examples/<category>/<name>.rst`.  
   Mirror the English content (same structure and sections). Use existing EN/ZH pairs under the same category (e.g. `embodied/libero.rst` in both `source-en` and `source-zh`) as reference.

5. **Register in the Chinese category index**  
   Edit `docs/source-zh/rst_source/examples/<category>/index.rst`:  
   - Add the same `<name>` entry to the hidden `.. toctree::`.  
   - If you added a gallery card in the English index, add a matching Chinese gallery card here, following existing HTML patterns.  
   If the Chinese docs have a vla-eval or start page that lists environments, add the new example using the same relative path pattern as in English (typically `../examples/embodied/<name>` for embodied.

6. **Update README.md**  
   In the "What's NEW!" section at the top, add a new dated bullet, with the documentation link pointing to the correct **category** path, e.g.:  
   - `- [YYYY/MM] 🔥 ... Doc: [Display Title](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/<name>.html).`  
   If the example is a simulator, model, or feature that appears in the Key Features table, add a corresponding list item in the right column using the same category path (see [reference.md](reference.md)).

7. **Update README.zh-CN.md**  
   In the "最新动态" section, add the same news item in Chinese with the doc link using `/zh-cn/` and the correct category path, for example:  
   - `https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/<name>.html`  
   If you added a feature list entry in README.md, add the same entry in README.zh-CN.md (Chinese display text, `zh-cn` in the link, and the same `<category>` segment).

### RST structure (concise)

- Title (overbar length matches title).
- Optional HuggingFace icon block (copy from libero.rst).
- Short intro (what this example does, which model + env).
- **Environment**: env name, task, observation/action space, task description format, data shapes.
- **Algorithm**: PPO/GRPO/etc. and model architecture notes.
- **Dependency Installation**: clone, install (Docker or pip).
- **Quick Start**: exact commands and key YAML/config snippets.
- **Evaluation** (if applicable): eval command and config notes.

Use existing examples in the same category (e.g. `embodied/libero.rst`, `embodied/pi0.rst`, `agentic/searchr1.rst`) as templates; see [reference.md](reference.md) for a minimal template.

---

## Checklist

- [ ] English RST created: `docs/source-en/rst_source/examples/<category>/<name>.rst`.
- [ ] English category index updated: `docs/source-en/rst_source/examples/<category>/index.rst` (toctree; optional gallery card).
- [ ] If embodied eval env: `docs/source-en/rst_source/start/vla-eval.rst` updated with `../examples/embodied/<name>`.
- [ ] Chinese RST created: `docs/source-zh/rst_source/examples/<category>/<name>.rst`.
- [ ] Chinese category index updated: `docs/source-zh/rst_source/examples/<category>/index.rst` (toctree; gallery card if added for EN).
- [ ] If embodied eval env: Chinese vla-eval/start page updated if it lists environments (use the same relative path pattern as EN).
- [ ] README.md updated: new bullet in "What's NEW!" and, if applicable, entry in Key Features table (using the correct category path).
- [ ] README.zh-CN.md updated: new bullet in "最新动态" and, if applicable, entry in 核心特性 table (using the correct category path).
