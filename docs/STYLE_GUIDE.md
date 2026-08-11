# RLinf Documentation Style Guide

The standing contract for **writing and refining** every page in
`docs/source-en` and `docs/source-zh`. It applies to new pages and to edits of
existing ones. The goal is task-first, scannable, consistent docs at the
[LeRobot](https://huggingface.co/docs/lerobot/en/index) /
[Ray](https://docs.ray.io/en/latest/index.html) standard.

This guide is the single source of truth for RLinf documentation. To apply it to
a page, use the **`refine-docs`** skill; to validate doc-to-code and EN/ZH
correctness, use the **`docs-check`** skill.

## Voice and tone

- **Second person, imperative.** "You'll fine-tune…", "Run the script", "Set `cluster.num_nodes`".
  Never "RLinf provides a comprehensive guide to launching and managing…".
- **Outcome first.** Open every page and section with what the reader gets, then how.
- **No throat-clearing.** Cut "This section provides a comprehensive guide to … within the RLinf framework, focusing on…". Start with the verb or the result.
- **Short sentences.** One idea each. Prefer lists, cards, and tables to run-on paragraphs.
- **Annotate commands.** After any non-trivial command, say what it does ("What this does: 1… 2…") and point to where to configure it further.

## Information architecture

RLinf docs are organized into eight single-purpose top-level axes, in this order:

**Get Started · Examples · Evaluation · Guides · Concepts · Reference · Extending · Resources**

Every page belongs to exactly **one** axis. Place it by the reader's starting
question, not by the team that owns the feature.

| Axis | Owns |
|---|---|
| **Get Started** | Install, quickstarts, requirements, cheat sheet. |
| **Examples** | The recipe galleries (simulators, robots, models, SFT, algorithms, agents, systems). |
| **Evaluation** | Eval onboarding, benchmark eval guides, eval CLI / config / results reference. |
| **Guides** | Operational how-tos: configure, launch & scale, data & checkpoints, performance, hardware backends, agent workflows. |
| **Concepts** | The mental model: execution flow, workers, channels, cluster, placement, execution modes, replay buffer. |
| **Reference** | Exact specs: APIs, algorithm specs, configuration keys & metrics, evaluation reference. |
| **Extending** | Contributor how-tos: new env / model / SFT, advanced integrations. |
| **Resources** | Why RLinf, blog, publications, release notes, FAQ. |

**Information ownership.** A page is owned by the section where readers look for
that task. Do not make `Concepts` point at broad aggregate pages that also own
Guides, Reference, or Extending content. If conceptual content is needed from a
mixed page, move or copy that concept into a dedicated Concepts page and link
operational/reference pages directly from their owning sections.

**Category ownership (Examples).** Place a page by the reader's starting point:
simulators / benchmarks → `simulators_index`; physical hardware →
`real_world_index`; model families and policy classes (including lightweight
policies such as ``MLP``) → `vla_wam_index` (Models); training recipes /
algorithms → `methods_index`; SFT-only workflows → `sft_index`. Do not duplicate
the same page in multiple gallery indexes.

**Evaluation ownership.** Evaluation is a first-class top-level section, not an
Examples subsection. `rst_source/evaluations/get_started/` owns eval onboarding,
`guides/` owns benchmark-specific eval workflows, and `reference/` owns eval
CLI/config/results reference. Training example pages may include training-time
validation and compact results, but standalone benchmark eval setup,
`run_eval.sh` usage, and result interpretation link to Evaluation instead of
duplicating it.

**Robots / Franka hierarchy.** Franka belongs under the Robots gallery, not as a
top-level Examples category. The Robots toctree links ``Franka <embodied/franka>``
so clicking **Franka** opens the base Real-World RL page, which owns the nested
Franka variant toctree (``Reward Model``, ``ZED + Robotiq``, ``GELLO``,
``Dual-Arm``, ``Dexterous Hand``, ``Pi0 SFT``, ``HG-DAgger``).

**URLs are versioned.** Moving a page changes its URL; since the docs are
versioned, accept the breakage and do not add redirects. Use stable
`:doc:` / `:ref:` cross-references internally so in-tree links survive moves.

**Sidebar grouping.** When a section's sidebar grows long and flat, group its
child pages under small, intent-based sub-indexes instead of adding more
top-level entries. Make the immediate sidebar children the group names and keep
individual articles one level deeper; give each group a short landing page with
cards or `list-table`s (not prose). Preserve page filenames when regrouping to
avoid link churn, and update both EN and ZH toctrees in the same change. The
established groupings:

- **Guides:** Configure · Launch & Scale · Data & Checkpoints · Performance · Hardware Backends · Agent Workflows.
- **Reference:** API · Algorithms · Configuration · Evaluation Reference.
- **Concepts:** Execution Model · Scheduling Model.
- **Extending:** keep the primary add-component pages (New Environment, New Model with FSDP, New Model with Megatron, New SFT Model) as immediate children; group only advanced topics under Advanced Integrations (Megatron-Bridge, weight synchronization, reward-model workflow).
- **Examples** keeps its own gallery category structure — do not regroup it.

## Landing page and section intros

- **Landing (`index.rst`)** is a task router, not a feature wall: a one-line
  value proposition, a centered hero (logo + welcoming title + subtitle), CTA
  cards (Get Started · Install · Examples · Evaluation), a "Choose Your Path"
  card grid, and a short "Why RLinf" teaser. The full feature/benchmark pitch
  lives on a dedicated **Why RLinf** page under Resources.
- **Get Started landing** is install → one copy-paste hello-world run →
  requirements → "What's Next" routing. Do not bury a marketing block here.
- **Section / sub-section landings** lead with a one-line purpose ("Pick this
  when…") and route via cards or tables — never prose walls.

## Section and subsection index pages

Every section and subsection landing — the root `index.rst`, each top-level axis
index, and every gallery / sub-index under them — organizes its contents with
**cards or tables, never bullet lists**.

- Open with a one-line outcome, then route with a `sphinx-design` card grid
  (`.. grid::` + `grid-item-card` using `:link:` / `:link-type: doc`), or a
  `list-table` when columns carry information (e.g. *Page · What you get*).
- Keep the page's `.. toctree::` **`:hidden:`** — it drives the sidebar nav and
  page order, while the visible body presents the same entries as cards/tables.
  Do **not** render child pages as `-` bullets or as a bare `:doc:`-per-line
  bullet list in the body.
- `examples/index.rst` is the reference implementation (category card grid).

## Navigation labels

Toctree entry captions (what shows in the left "Section Navigation") must be the
**bare name** — no "Benchmark", "Benchmarks", "Models", "World Model",
"Simulation Platform", "RL with …", "Training", "评测平台", "仿真平台", "模型"
prefixes/suffixes. Use an explicit caption in the toctree:
``LIBERO <embodied/libero>``, ``MLP <embodied/mlp>``, ``π₀ / π₀.₅ <embodied/pi0>``.
The page **H1 title** may stay descriptive (e.g. "RL with LIBERO Benchmarks");
only the nav caption is shortened. This applies to **every** gallery.

**Top-level gallery category captions** (in `examples/index.rst`) are a single
word — the category, not a sentence:

| Index page | Nav caption (EN) | Nav caption (ZH) |
|---|---|---|
| `simulators_index` | Simulators | 模拟器 |
| `real_world_index` | Robots | 真机 |
| `vla_wam_index` | Models | 模型 |
| `sft_index` | SFT | SFT |
| `methods_index` | Algorithms | 算法 |
| `agentic/index` | Agents | 智能体 |
| `system/index` | Systems | 系统 |

The category index pages keep their descriptive H1 (e.g. "Algorithms for
Embodiment"); only the `examples/index.rst` toctree caption is the one-word form.

**Global navigation (sidebar-only).** The top bar is intentionally removed; place
the logo/title, search field, and a compact utility row (version selector +
repository link with a live GitHub star count) in the left sidebar, followed by
the global section navigation. The **Ask AI** button is a floating action button
pinned to the bottom-right of the viewport (not in the sidebar). The sidebar must
show all eight axes from every page, including the home index. All top-level
sections expand to their immediate children by default (`js/sidebar-nav.js`),
while deeper sub-trees stay collapsed. Keep `navbar_start: []`,
`navbar_center: []`, `navbar_end: []`, `html_sidebars` ordered as `sidebar-brand`,
`search-field`, `sidebar-tools`, `global-sidebar-nav`, `collapse_navigation: False`,
`show_nav_level: 1`, and `navigation_depth: 5` unless an IA change intentionally
revises the global contract. The header band is collapsed to zero height
(`--pst-header-height: 0`) so the sidebar starts at the top edge. The root
`index.rst` toctree stays hidden so navigation lives in the sidebar, not the page
body.

## Example / recipe page requirements

Every benchmark (env) or model example page must:

1. **Open with a figure + intro.** Lead with the upstream benchmark/model figure
   (credited) and one paragraph on what it is and how RLinf uses it — like the
   [LeRobot benchmark pages](https://huggingface.co/docs/lerobot/en/libero).
2. **Put the benchmark facts inside Overview as tables.** Under the card grid, add
   two H3 subsections — `Tasks` (always a `list-table`, never a bullet list) and
   `Observation and Action` (a `list-table` of observation/action/reward/prompt).
   There is no separate "Tasks and Environment" section.
3. **Overview = 4 aligned cards.** Use `.. grid:: 2 4 4 4` (see anatomy). Cards
   must align within each gallery subsection, with the exact same card titles and
   order in every page in that subsection. On an **env** page the **Models** card
   lists *every* model supported on that env and the **Algorithms** card lists
   *every* algorithm.
4. **No "Env type" card** — it carries too little information; put the env-type
   string in prose or the overview table instead.
5. **No generic "Algorithm" section** and **no boilerplate VLA intro** ("This
   section provides a comprehensive guide…", "Visual Understanding / Language
   Comprehension…"). Algorithm definitions live in Reference, not on every recipe
   page.
6. **Cards or tables, not bullet walls.** Replace bullet lists of
   specs/metrics/perturbations with cards or `list-table`s.
7. **Don't explain metrics or evaluation per page.** Link to the shared
   :doc:`Training metrics <reference/metrics>` page for training logs and to the
   unified Evaluation section for benchmark / standalone eval workflows. Keep only
   the page-specific "watch `env/success_once`" pointer and the results table.
8. **Name the card-grid section "Overview".** On a single-recipe page the card
   grid lives under an `Overview` heading right after the intro. On a multi-recipe
   page (e.g. LIBERO), there's no page-level `Overview`; instead each recipe family
   is its own section with a **descriptive, parallel** name (e.g. `Standard LIBERO
   Suites` / `LIBERO-Pro & LIBERO-Plus Suites`) and its own card grid. Don't repeat
   the H1 in a subtitle, and give any `:ref:` that points at a renamed section
   explicit link text so it still reads right.

### Page anatomy (recipe / example pages)

```rst
RL with <Name> Benchmarks            ← descriptive H1; nav caption is just "<Name>"
=========================

.. figure:: <upstream figure URL>
   :align: center
   :width: 90%

   <caption with image credit>

<One paragraph: what the benchmark/model is and how RLinf uses it.>

Overview                             ← cards + the benchmark facts (no "Tasks and Environment" title)
--------

<one-line outcome>.

.. grid:: 2 4 4 4                    ← 4 aligned cards (a 12-col grid aligns cleanly only
   :gutter: 2                          for 1/2/3/4/6 — avoid 5; push overflow to prose)

   .. grid-item-card:: Models        ← list EVERY model supported on this env
      :text-align: center
      <list>
   .. grid-item-card:: Algorithms    ← list EVERY algorithm supported
   .. grid-item-card:: Tasks
   .. grid-item-card:: Hardware

| **You'll do:** install → download model → launch → watch ``<metric>``.
| **Prerequisites:** :doc:`Installation <…>` · <other prereqs>.

Tasks                                ← H3, always a TABLE (never a bullet list)
~~~~~
.. list-table::   (benchmark suites: Suite · config id · Tasks · Focus;
                   multi-task envs: Category · Task · Description)

Observation and Action               ← H3
~~~~~~~~~~~~~~~~~~~~~~~
.. list-table::   (Observation · Action · Reward · Task prompt)

Installation              → .. include:: _setup_common.rst + recipe-specific tag / --env
Download the Model        → recipe-specific download + .. include:: _model_path.rst
Run It                    → command + "What this command does" + "Configure further" admonition
Visualization and Results → TensorBoard / video / logger + link to Training metrics;
                            link to Evaluation for standalone eval; results as a TABLE
```

## Headings and admonitions

- **Title Case for all headings**, consistent: `Run It`, `Download the Model`,
  `Visualization and Results` (lowercase only articles/short
  prepositions/conjunctions: a, an, the, and, or, of, to, in, on, with, vs).
- **One H1 per page.** A page with two top-level (`===`) headings breaks title
  resolution and the sidebar caption; demote the second to a subsection.
- **Standard section names** (use these exact names so pages match):
  `Overview` (the card grid + the `Tasks` and `Observation and Action`
  subsections) · `Installation` · `Download the Model` (and `Download the Assets`
  if needed) · `Run It` · `Visualization and Results`.
- **Align pages within each gallery subsection.** Simulator / benchmark pages use
  `Overview` → `Tasks` → `Observation and Action` → `Installation` → optional
  download sections → `Run It` → `Visualization and Results`. Model pages use the
  same overview table pattern, including lightweight policies such as ``MLP``.
  Within one subsection, overview cards and tables must use the same fields. Card
  schemas are: Simulators and Robots use `Models`, `Algorithms`, `Tasks`,
  `Hardware`; Models use `Environments`, `Algorithms`, `Tasks`, `Hardware`;
  Algorithms use `Algorithm`, `Models`, `Environments / Data`, `Training`; SFT uses
  `Models`, `Methods`, `Data`, `Hardware` (translated in ZH). Models pages use
  `Tasks` columns `Environment`, `Task / Suite`, `Config / Weights`, `Focus`, and
  `Observation and Action` rows `Observation`, `Action`, `Reward`, `Prompt`
  (translated in ZH, technical row names unchanged). Omit `Download the Model` only
  when there is no checkpoint to download. Algorithm pages use `Overview` with cards
  and a task/config table, then a method-specific `How <Method> Works` / `Pipeline`
  section before setup and commands. SFT pages use `Overview`, then dataset/model
  preparation sections, then `Installation`, `Run It`, and `Visualization and
  Results` where applicable. Robots pages may keep hardware/safety workflow
  sections but still start with `Overview`, `Tasks`, and `Observation and Action`.
- **Overview** uses a `sphinx-design` card grid (`.. grid:: 2 4 4 4` +
  `grid-item-card`), not a `tip` admonition.
- `note` = side info · `warning` = footguns (OOM, `MUJOCO_GL`, `RLINF_NODE_RANK`
  ordering, multi-node gotchas). Put footguns in a `warning`, not prose.

## Reuse

- **Link, don't inline** reference material (full config tables, the complete
  metrics list, placement theory, standalone evaluation workflows). Each page does
  one job and links to the canonical Reference or Evaluation page.
- **Shared partials** live as underscore-prefixed files (`_setup_common.rst`,
  `_model_path.rst`). They are excluded from the build
  (`exclude_patterns = ["**/_*.rst"]`) and pulled in with `.. include:: _name.rst`.
  Substitutions don't work inside code blocks, so partials hold only the
  *identical* prose/code; recipe-specific tokens stay on the page.
- **Don't copy-paste across pages.** If the same command block, YAML snippet, or
  paragraph appears on three or more pages, extract it into a partial or link to a
  single canonical page.

## Images and media

- **Verify every image/media URL resolves (HTTP 200) before committing.** Broken
  images are a recurring problem — check the figure, every `<img>`/`<source>` in
  `raw:: html` blocks, and result images. Quick scan:

  ```bash
  grep -rhoE '(\.\. (figure|image):: |src=")https?://[^ "<>]+' source-en source-zh \
    | sed -E 's/^\.\. (figure|image):: //; s/^src="//' \
    | grep -iE '\.(png|jpg|jpeg|gif|svg|mp4|webm)$' | sort -u \
    | while read -r u; do echo "$(curl -s -o /dev/null -w '%{http_code}' -L "$u")  $u"; done \
    | grep -v '^200 '
  ```

- **Use a direct host URL, not a redirecting one.** Prefer
  `https://raw.githubusercontent.com/<org>/<repo>/<branch>/<path>` (or the gh-pages
  site). Avoid `https://github.com/<org>/<repo>/raw/...` — it 301/302-redirects
  through `text/html` responses that browsers don't reliably render as an `<img>`.
- **Watch for repo renames.** A 301 on the `raw` path means the org/repo moved
  (e.g. `haosulab/ManiSkill` → `mani-skill/ManiSkill`); point at the current name.
- **Confirm the exact path.** RLinf assets live in `RLinf/misc` under `pic/` *and*
  subfolders (e.g. `pic/rlinf-vla/…`, `pic/release_0.2/…`) — a wrong subfolder 404s.
- **Prefer a static image over a large animated GIF** for page figures.
- **Make figure captions specific to the page.** Not "Robot setup used by this
  RLinf recipe" but "GELLO joint-level teleoperation device used to collect Franka
  demonstrations." For RLinf-owned images, omit image-credit suffixes.
- A content image gets a white background in dark mode from the theme; for logos
  or diagrams that should stay transparent, override with
  `background-color: transparent !important`.

## EN ↔ ZH parity

- Land every change in **both** trees in the same pass; keep the same file set and
  toctree targets.
- **Code identifiers are sacred** — never translate config keys, env-type strings,
  CLI flags, script names, or model names in ZH.
- Headings are translated; internal links use stable `:doc:` / `:ref:` (no
  hardcoded ReadTheDocs URLs).
- **ZH:** don't put `**bold**` directly between CJK characters — docutils won't
  render it and a literal `**` leaks into the page. Use a space-bounded boundary,
  Chinese quotes, or drop the emphasis.

## Review gate

After each change:

- `sphinx-build` both trees with **zero new warnings**:
  `/opt/venv/docs/bin/sphinx-build -b html docs/source-en /tmp/build-en` and the
  same for `source-zh`.
- Run the **`docs-check`** skill (doc-to-code correctness + EN/ZH parity).
- Confirm no new bullet-list index pages, no throat-clearing intros, and no
  `**bold**` glued between CJK characters.
