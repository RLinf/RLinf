# Index pages, navigation, and where a page goes

An index page routes readers to other pages; it does not explain anything itself. This file also says where a new page belongs and how the sidebar presents it.

## Where a page goes

The docs have eight top-level chapters in the sidebar, in this order: **Get Started · Embodied AI · Real-World Robots · Agent & LLM · Guides · Extending · Concepts & Reference · Resources**. Every page belongs to exactly one, chosen by the reader's starting question rather than by the team that owns the feature.

| Chapter | Owns |
|---|---|
| Get Started | installation & hardware requirements, first training run, tour of RLinf, FAQ |
| Embodied AI | overview, simulators, embodied models, SFT & imitation, RL recipes, world models, embodied evaluation (quick tour, benchmark guides, eval reference) |
| Real-World Robots | overview, shared real-world workflows (multi-node robot launch, teleoperation, data collection), and per-robot bring-up / SFT / RL pages |
| Agent & LLM | overview, Reasoning (math GRPO/PPO, Moonlight MoE, Qwen3-VL Geo3K), Agentic RL (Search-R1, WideSeek-R1, rStar2, AgentLightning, Coding Online RL), Image & Video Generation |
| Guides | workflow-ordered operational how-tos: Configure & Run, Scale to a Cluster, Profile & Speed Up, Rollout Backends |
| Extending | goal-grouped contributor how-tos: Add a Model (all model paths & weight sync), Add an Env or Robot, Add an Algorithm, Add a Reward, Add an Agent, Add a Hardware Backend |
| Concepts & Reference | concepts (overview, execution, scheduling, robotics), algorithm specs, config & metrics, Python API |
| Resources | why RLinf, release notes, blog, publications, contributor policy |

- **Information ownership.** A page is owned by the section where readers look for its task. When a mixed page holds a concept that Concepts needs, move the concept to a dedicated Concepts page and link the operational and reference parts from their own sections.
- **Examples categories.** Simulators and benchmarks go in `simulators_index`, physical hardware in `real_world_index`, model families and policy classes (including lightweight policies such as MLP) in `vla_wam_index`, training recipes and algorithms in `methods_index`, and SFT-only workflows in `sft_index`. A page appears in one gallery index only.
- **Hardware setup.** Model example pages own their backend-specific install and launch steps ([recipe.md § Run on Different Hardware Backends](recipe.md#run-on-different-hardware-backends)) and list supported backends in the Hardware card. The README support matrix, the Models gallery, the install guide, and simulator pages link to those sections. Hardware e2e coverage is distinct from installer options: support applies to a model, env, and backend combination.
- **Evaluation.** The sidebar has 4 items: `Overview` (`evaluations/index`), `Quick Tour` (`evaluations/get_started/quick_tour`), `Benchmarks` (`evaluations/guides/index`), and `Eval Reference` (`evaluations/reference/index`), with `start/installation.rst` as the single installation hub. `evaluations/guides/` owns benchmark-specific eval workflows, and `evaluations/reference/` owns the eval CLI, config, models, and results. Training pages may show training-time validation and compact results, and link to Evaluation for the rest.
- **Robots and Franka.** Franka sits under the Robots gallery. The Robots toctree links `Franka <embodied/franka>`, the base Real-World RL page, which owns the nested Franka variants (Reward Model, ZED + Robotiq, GELLO, Dual-Arm, Dexterous Hand, Pi0 SFT, HG-DAgger).
- **Sidebar grouping.** When a section's sidebar grows long and flat, group its children under small intent-based sub-indexes, each a short landing page of cards or tables. Keep file names when regrouping, and update the EN and ZH toctrees together. The groupings are:
  - **Embodied AI → Evaluation**: `Quick Tour · Benchmarks · Eval Reference` (`start/installation.rst` is the single installation hub).
  - **Real-World Robots → Workflows**: `Multi-Node Robot Launch · Teleoperation · Data Collection`.
  - **Guides**: `Overview · Configure & Run · Scale to a Cluster · Profile & Speed Up · Rollout Backends`.
  - **Extending**: `Overview · Add a Model · Add an Env or Robot · Add an Algorithm · Add a Reward · Add an Agent · Add a Hardware Backend`.
  - **Concepts & Reference**: `Concepts` (`Overview · Execution · Scheduling · Robotics`), `Algorithms`, `Config & Metrics` (owns `Training Configuration`, `Training Metrics`, and the 3 full schema reference pages: `basic_config`, `embodiment_config`, `agentic_config`), and `Python API`.

## Landing and index pages

- **The landing page (`index.rst`)** routes tasks: a one-line value proposition, a centered hero (logo, title, subtitle), cards for Get Started, Install, Examples, and Evaluation, a "Choose Your Path" card grid, and a short "Why RLinf" teaser. The full pitch lives on the Why RLinf page under Resources.
- **The Get Started landing** is install, one copy-paste first run, requirements, and "What's Next" routing.
- **Every section and sub-section index** opens with one line saying when to pick it ("Pick this when…"), then routes with a `sphinx-design` card grid (`.. grid::` with `grid-item-card`, `:link:`, `:link-type: doc`) or a `list-table` when columns carry information. Never use bullet lists, one `:doc:` per line, or `.. raw:: html` card grids with hardcoded inline styles (`color: blue`). `examples/real_world_index.rst` and `examples/agentic/index.rst` are the models.
- **The toctree is `:hidden:`.** It drives the sidebar and page order; every entry in the hidden toctree must appear in the page's cards or table (`docs/index-style`).

## Navigation labels

- **Sidebar labels match target page H1s (`docs/sidebar-title-match`).** On Extending, Guides, Concepts, and Config & Metrics toctrees, every explicit `Label <target>` entry (except `Overview` / `总览`) matches the target page's H1 verbatim. On gallery and Reference (Algorithms / Python API) toctrees, sidebar labels use concise bare names (`LIBERO`, `PPO`, `Actor`) that appear inside the target page's H1 (`LIBERO Evaluation`, `Proximal Policy Optimization (PPO)`, `Actor Interface`).
- **Keep sibling titles parallel.** Sibling entries under the same parent must follow the same grammatical pattern and naming symmetry—for example:
  - **Add a Reward / 新增 Reward**: `Rule-Based Reward` · `Model-Based Reward` (`基于规则的 Reward` · `基于模型的 Reward`)
  - **Add an Agent / 新增 Agent**: `White-Box Agent` · `Black-Box Agent` (`白盒 Agent` · `黑盒 Agent`)
  - **Add an Env or Robot / 新增 Env 与真机**: `Simulator Env` · `Real-World Hardware` · `Real-World Task` (`仿真 Env` · `真机硬件` · `真机任务`)
  - **Add a Model / 新增模型**: `<Domain> Model (<Backend>)` (`具身 RL 模型（FSDP）` · `具身 SFT 模型（FSDP）` · `具身 Rollout 模型（SGLang）` · `LLM 模型（Megatron-Bridge）` · `LLM 模型（手动 Megatron）`)
- **Toctree captions on recipe galleries are bare names**, with no "Benchmark", "Models", "World Model", "Simulation Platform", "RL with …", "Training", 评测平台, 仿真平台, or 模型 around them: `LIBERO <embodied/libero>`, `MLP <embodied/mlp>`, `π₀ / π₀.₅ <embodied/pi0>`.
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

The top bar is removed. The left sidebar holds the logo and title, the search field, a compact row with the version selector and the repository link with its live star count, and then the section navigation, which shows all eight axes from every page. Expanded sidebar branches persist in `sessionStorage` (`js/sidebar-nav.js`) so multiple sections can remain open simultaneously across page navigations. The Ask AI button floats at the bottom right of the viewport. Keep `navbar_start: []`, `navbar_center: []`, `navbar_end: []`, `html_sidebars` ordered as `sidebar-brand`, `search-field`, `sidebar-tools`, `global-sidebar-nav`, `collapse_navigation: False`, `show_nav_level: 1`, `navigation_depth: 4`, and `--pst-header-height: 0` unless a layout change revises them on purpose. The root `index.rst` toctree stays hidden.
