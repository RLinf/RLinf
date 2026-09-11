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

These apply to every page:

- **Second person, imperative.** "You'll fine-tune…", "Run the script", "Set `cluster.num_nodes`".
  Never "RLinf provides a comprehensive guide to launching and managing…".
- **State the page's purpose first.** The first prose sentence after the title,
  or after a leading figure, must say directly what the page explains, enables,
  routes, or lets the reader look up. Do not begin with background and wait until
  a later paragraph to reveal why the page exists.
- **Outcome first.** After stating the page's purpose, explain what the reader
  gets and how. Open each section with the result it establishes.
- **No throat-clearing.** Cut "This section provides a comprehensive guide to … within the RLinf framework, focusing on…". Start with the verb or the result.
- **Annotate commands.** After any non-trivial command, say what it does ("What this does: 1… 2…") and point to where to configure it further.
- **Name what you mean.** Prefer `Robot.connect`, `PartGroup`, `Placement` to "the
  common layer", "the robotics machinery", "the rest of the system".
- **Avoid the usual tells:** "it is worth noting", "simply", "seamlessly",
  "powerful", "leverage", "robust", "in order to", "a wide range of", and
  paragraphs that all open the same way.

### Explain before naming

Use this basic language flow in documentation, review comments, and design
conversations. The page-layout rules in this guide are docs-specific, but this
order applies to any explanation:

1. Start from the concrete situation or question the reader recognizes.
2. Explain the idea in ordinary language and say what distinction matters.
3. Introduce the exact class, method, field, or config name.
4. Show one example that connects the name back to the idea.
5. Add edge cases only after the normal path is clear.

A heading, card title, or opening sentence must make sense before the reader
knows the implementation. Do not introduce an unexplained API term in a heading
and define it below. API and reference pages may use an identifier as a heading
when that identifier is the reader's lookup target; elsewhere, introduce the
term in prose first or write the heading in task-oriented language.

This is progressive disclosure at sentence level. It is not a reason to hide
precise names: once the idea is clear, use the real identifier consistently so
readers can search for it in code.

### Guide the reader from use to internals

Progressive disclosure also determines the order of a whole page. Begin with the
path most readers need and introduce system details only when they explain the
next task. For an extensible system, use this sequence when it fits the topic:

1. Use an existing component through the public API.
2. Add one component in the normal local configuration.
3. Compose it with components that already exist.
4. Show the remote or distributed form without changing the caller's mental
   model.
5. Explain ownership, placement, connection management, or scheduler internals.

Do not use an internal class hierarchy as the teaching outline. Related API
terms such as ``exports`` and ``children`` may appear in code, but prose must
first explain what each collection contains, how the two differ, and what type
the corresponding method returns. Readers should never need to reverse-engineer
a title or example before they can understand the distinction it is meant to
teach.

Examples should close the loop from declaration to use. When a page adds an
extensible component, show how it combines with an existing component, how the
application reads or controls it, and where it enters a real task or environment.
An isolated class definition demonstrates syntax, not composability.

### Technical accuracy is part of the prose

Documentation is a user-facing description of the code's contract. Check public
signatures, accepted input types, concrete return types, lifecycle behavior,
configuration names, and call sites before describing them. If a constructor
accepts two related types, explain what each type represents and why both forms
are valid. Do not infer behavior from a class or field name, and do not describe
an intended design when the implementation still behaves differently.

For a behavior-preserving refactor, compare every affected path with the
baseline and state the intentional differences. Update examples and conceptual
pages in the same change when a public name, type relationship, ownership rule,
or lifecycle changes.

### Explanatory pages: Concepts and Guides

Recipes, index pages, and reference tables should stay terse — a reader scanning
for a command wants the command. Concepts and Guides are different: they exist to
make someone understand a design, and terse prose actively fails at that. Write
them like a colleague explaining the system at a whiteboard to an engineer who
has to use it — concrete, willing to say why a choice was made, willing to name a
sharp edge, and not selling anything.

- **Concise, not clipped.** The rule that goes wrong most often, because "cut
  throat-clearing" reads as licence to write telegraphically. A page where every
  sentence starts cold and stops the instant the fact lands reads like a spec
  sheet, not an explanation. Vary sentence length. Let a sentence finish its
  thought instead of ending at the first period that would parse, and carry one
  paragraph into the next with a real transition instead of restarting from zero.
- **Take the reader with you.** Guiding words earn their place here: "Let's build
  a Franka from the ground up", "Say your gripper hangs off the arm's own
  connection", "Now that the parts are declared…", "We'll come back to placement
  below". A few per page give it a spine; one in every paragraph becomes its own
  tic.
- **Vary paragraph shape.** If every paragraph is three sentences with the same
  rhythm, the page sounds machine-written even when no individual sentence is
  wrong. Some paragraphs are a single sentence; some run five. A code block that
  explains itself may need a line of setup and nothing after it.
- **Explain the mechanism, skip the pitch.** Don't end paragraph after paragraph
  with a sentence whose only job is to say the design is good — "This keeps the
  common behavior in one place", "That split prevents details from leaking",
  "You get concurrency without having to coordinate it". A reader who has just
  seen the mechanism can see the benefit. Two or three per page, where the payoff
  genuinely isn't obvious from the mechanism.
- **Say the thing; don't announce it.** "`build` never mentions the gripper"
  beats "Notice that `build` never mentions the gripper".

### Build a continuous article

Every article needs a continuous line of thought, not a sequence of locally
correct paragraphs. Its introduction, sections, examples, and transitions must
let a reader follow one question from the page title to the final result. The
amount of prose varies by page type: an index may establish its purpose in one
sentence before routing through cards, and a reference page may lead with what
can be looked up and how entries are organized. Neither is exempt from having a
clear lead and a deliberate order.

- **Give the page a lead.** Begin with a direct statement of what the page does.
  The rest of the opening should establish the reader's situation, the result
  the page will help them reach, the boundary of the topic, and the order in
  which the page develops it. A list of features or implementation areas is not
  a lead. A reader should be able to predict why the second section follows the
  first.
- **Give every section a lead.** Open a section by connecting it to the state
  established above and naming the one question the section resolves. Do not
  begin abruptly with a code block, table, API name, or isolated fact. The lead
  should add direction, not merely repeat the heading.
- **Make paragraphs depend on one another.** Develop the section as a chain:
  establish the distinction, introduce the relevant API, show it in a complete
  example, interpret the result, then carry that result into the next concern.
  Reordering the paragraphs should change the explanation; if it does not, the
  section is probably a fact list.
- **Explain an interface in call order.** When a section teaches a workflow,
  account for each public operation the example relies on, in the order a caller
  uses it. State what the operation accepts or returns, why it is needed at that
  point, and how its result feeds the next call. Do not place several unfamiliar
  methods in one example and explain only the most interesting two.
- **Frame and interpret examples.** Before a code block, state the concrete
  result it demonstrates. After it, explain the important return values,
  ownership or lifecycle effects, and the next decision the reader can now
  make. Avoid line-by-line narration, but never leave the example to carry the
  conceptual transition by itself.
- **Close the local loop.** End a section with the established result or the
  condition that motivates the next section when that relationship is not
  already obvious. Use a real dependency between ideas rather than a generic
  transition such as "Next, we discuss...".

Before accepting any article, read only its introduction and the opening
paragraph of each section. They should form a coherent outline on their own.
Then read the full page and verify that each paragraph advances that outline.
When the article teaches an interface, also verify that every API used in the
primary example is explained and that every code block advances the same
narrative.

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
- **Concepts:** Execution · Scheduling.
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

### Writing the Chinese pages

Parity covers structure and technical content, not sentence order. A ZH
paragraph that reads better with a different number of sentences than its EN
counterpart is correct, not a defect — mirroring the English clause by clause is
exactly what makes a page read as a translation. Write the Chinese as Chinese:
decide what the paragraph needs to say, then say it the way a Chinese engineer
would say it to a colleague.

Conventions, on every page:

- 标题后的第一句应直接说明本页介绍什么、帮助读者完成什么，或可供查阅什么。若页面以图片开头，则从图片后的第一句开始遵循此要求。不要先铺陈背景，到后文才交代页面用途。
- 全角标点：，。、；：（）「」。中文句子里不要混用半角逗号句号。
- 中文与英文、数字之间空一格，例如「在 node_rank 指定的节点上」。
- 中文正文不要按列宽手动换行。每个段落或列表项的正文在 RST 源文件中保持一行；reStructuredText 会把段内换行渲染成空格，在两个汉字之间留下不自然的间隔。标题、directive、表格和代码块所需的结构换行不受此规则影响。
- 英文技术名词保留原文，不要硬译；术语前后统一。强化学习中的 policy 统一写作 policy，不译成“策略”；描述通用决策方法时仍可使用“策略”，例如“placement 策略”。
- 自然不等于口语化。正文应采用清晰、克制的书面技术表达，避免“看看长什么样”“等需要时再看”“不用跟着改”等聊天式说法；同时避免“本文旨在”“进行相关操作”等公文腔。
- 不要滥用「的」；少用「进行/实现/提供/负责/使得」这类空动词。
- 被动改主动；长定语从句拆成短句：中文靠短句和动词推进，不靠从句堆叠。
- 开发者日常使用的英文词不必硬译，例如 policy、key、value、mapping、endpoint、worker、binding、
  wrapper、mock SDK、contract、shape、schema 和 API。先用中文解释它在当前场景中的作用，
  再保留代码里能搜索到的名称。
- Robotics 文档中，表示机械臂、夹爪、相机等 ``RobotPart`` 时统一使用「零部件」。不要将所有 part 都机械地翻译为「零部件」：``action_parts`` 中的 part 按语境写成「动作项」或「对应的机器人动作」。介绍组合模型时，优先说明「零部件的名称、层级和访问路径」，避免「按名称组织的部件树」「对外公开的部件树」等抽象且冗长的表达；API 名 ``parts`` 和 ``children`` 保留原文，分别解释为 connection 支持的零部件和组合中的直接下一级。
- 标题只写读者已经理解的任务或概念。不要先在标题中抛出 ``exports``、``children`` 一类
  实现名，再到正文里补定义。
- 少用「本节将」「接下来」「此时」「因此」「这样就能」「值得注意的是」串联每一段。
  这些词有明确作用时可以用，但不能代替真正的上下文和过渡。

On Concepts and Guides pages, where the English follows the explanatory rules
above, the Chinese needs the same treatment in its own idiom:

- 用「我们」「先……再……」「不妨」「接下来」把读者带着走，段落之间要有过渡。
- 去掉翻译腔和英文语序，例如「这就是全部的接入工作」「它由 X 负责读取」。
- 破折号「——」少用，多用分句、冒号或直接断句。

## Review gate

After each change:

- `sphinx-build` both trees with **zero new warnings**:
  `/opt/venv/docs/bin/sphinx-build -b html docs/source-en /tmp/build-en` and the
  same for `source-zh`.
- Run the **`docs-check`** skill (doc-to-code correctness + EN/ZH parity).
- Confirm no new bullet-list index pages, no throat-clearing intros, and no
  `**bold**` glued between CJK characters.

For every article, also read the page top to bottom in both languages before
merging, and rewrite if any of these are true:

- Most paragraphs are the same length and end in a sentence about why the design
  is good.
- Sentences begin cold and stop the moment the fact lands, with no transition
  between paragraphs or sections.
- The ZH page tracks the EN sentence for sentence.
- The page introduction does not establish a result, scope, and reading order.
- The first prose sentence does not state the page's purpose directly.
- A section can be moved elsewhere without changing the surrounding
  explanation, or begins with code or an API name before stating why it is
  needed.
- The primary example calls public methods that the surrounding prose never
  explains, or explains them in an order unrelated to the workflow.

Cadence is the hardest thing to hear in your own prose, so a second pass by a
different writer — human or model — catches what a self-review will not. Give
that reviewer specific examples of what reads wrong, not a general request to
improve the writing; a vague brief comes back with the same cadence reworded.
