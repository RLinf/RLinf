# Index pages, navigation, and where a page goes

An index page routes readers to other pages; it does not explain anything itself. This file also says where a new page belongs and how the sidebar presents it.

## Where a page goes

The docs have eight top-level axes, in this order: **Get Started · Examples · Evaluation · Guides · Concepts · Reference · Extending · Resources**. Every page belongs to exactly one, chosen by the reader's starting question rather than by the team that owns the feature.

| Axis | Owns |
|---|---|
| Get Started | install, quickstarts, requirements, cheat sheet |
| Examples | the recipe galleries: simulators, robots, models, SFT, algorithms, agents, systems |
| Evaluation | eval onboarding, benchmark eval guides, eval CLI, config, and results reference |
| Guides | operational how-tos: configure, launch and scale, data and checkpoints, performance, agent workflows |
| Concepts | the mental model: execution flow, workers, channels, cluster, placement, execution modes, replay buffer |
| Reference | exact specs: APIs, algorithm specs, configuration keys and metrics, evaluation reference |
| Extending | contributor how-tos: new env, model, or SFT model, advanced integrations |
| Resources | why RLinf, blog, publications, release notes, FAQ |

- **Information ownership.** A page is owned by the section where readers look for its task. When a mixed page holds a concept that Concepts needs, move the concept to a dedicated Concepts page and link the operational and reference parts from their own sections.
- **Examples categories.** Simulators and benchmarks go in `simulators_index`, physical hardware in `real_world_index`, model families and policy classes (including lightweight policies such as MLP) in `vla_wam_index`, training recipes and algorithms in `methods_index`, and SFT-only workflows in `sft_index`. A page appears in one gallery index only.
- **Hardware setup.** Model example pages own their backend-specific install and launch steps ([recipe.md § Run on Different Hardware Backends](recipe.md#run-on-different-hardware-backends)) and list supported backends in the Hardware card. The README support matrix, the Models gallery, the install guide, and simulator pages link to those sections. Hardware e2e coverage is distinct from installer options: support applies to a model, env, and backend combination.
- **Evaluation.** `evaluations/get_started/` owns eval onboarding, `guides/` benchmark-specific eval workflows, and `reference/` the eval CLI, config, and results. Training pages may show training-time validation and compact results, and link to Evaluation for the rest.
- **Robots and Franka.** Franka sits under the Robots gallery. The Robots toctree links `Franka <embodied/franka>`, the base Real-World RL page, which owns the nested Franka variants (Reward Model, ZED + Robotiq, GELLO, Dual-Arm, Dexterous Hand, Pi0 SFT, HG-DAgger).
- **Sidebar grouping.** When a section's sidebar grows long and flat, group its children under small intent-based sub-indexes, each a short landing page of cards or tables. Keep file names when regrouping, and update the EN and ZH toctrees together. The groupings are: Guides (Configure · Launch & Scale · Data & Checkpoints · Performance · Agent Workflows), Reference (API · Algorithms · Configuration · Evaluation Reference), Concepts (Execution · Scheduling), and Extending (the add-component pages as direct children, with Megatron-Bridge, weight synchronization, and the reward-model workflow under Advanced Integrations). Examples keeps its gallery categories.

## Landing and index pages

- **The landing page (`index.rst`)** routes tasks: a one-line value proposition, a centered hero (logo, title, subtitle), cards for Get Started, Install, Examples, and Evaluation, a "Choose Your Path" card grid, and a short "Why RLinf" teaser. The full pitch lives on the Why RLinf page under Resources.
- **The Get Started landing** is install, one copy-paste first run, requirements, and "What's Next" routing.
- **Every section and sub-section index** opens with one line saying when to pick it ("Pick this when…"), then routes with a `sphinx-design` card grid (`.. grid::` with `grid-item-card`, `:link:`, `:link-type: doc`) or a `list-table` when columns carry information. Never bullet lists or one `:doc:` per line in the body. `examples/index.rst` is the model.
- **The toctree is `:hidden:`.** It drives the sidebar and page order; the body shows the same entries as cards or tables.

## Navigation labels

- **Toctree captions are bare names**, with no "Benchmark", "Models", "World Model", "Simulation Platform", "RL with …", "Training", 评测平台, 仿真平台, or 模型 around them: `LIBERO <embodied/libero>`, `MLP <embodied/mlp>`, `π₀ / π₀.₅ <embodied/pi0>`. The page's H1 may stay descriptive ("RL with LIBERO Benchmarks").
- **Gallery category captions** in `examples/index.rst` are one word; the category pages keep descriptive H1s.

  | Index page | EN caption | ZH caption |
  |---|---|---|
  | `simulators_index` | Simulators | 模拟器 |
  | `real_world_index` | Robots | 真机 |
  | `vla_wam_index` | Models | 模型 |
  | `sft_index` | SFT | SFT |
  | `methods_index` | Algorithms | 算法 |
  | `agentic/index` | Agents | 智能体 |
  | `system/index` | Systems | 系统 |

- Pages and captions about adding tasks that run on physical hardware say "New Real-World Tasks" and 新增真机任务.

## Global navigation

The top bar is removed. The left sidebar holds the logo and title, the search field, a compact row with the version selector and the repository link with its live star count, and then the section navigation, which shows all eight axes from every page. Top-level sections expand to their direct children (`js/sidebar-nav.js`), and deeper trees stay collapsed. The Ask AI button floats at the bottom right of the viewport. Keep `navbar_start: []`, `navbar_center: []`, `navbar_end: []`, `html_sidebars` ordered as `sidebar-brand`, `search-field`, `sidebar-tools`, `global-sidebar-nav`, `collapse_navigation: False`, `show_nav_level: 1`, `navigation_depth: 5`, and `--pst-header-height: 0` unless a layout change revises them on purpose. The root `index.rst` toctree stays hidden.
