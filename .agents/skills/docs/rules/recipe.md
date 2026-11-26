# Recipe pages

A recipe takes the reader from a fresh machine to a working run of one model, env, algorithm, or robot setup. The Examples galleries, the Get Started quickstarts, and the Evaluation get-started pages are recipes. The reader wants to run the thing and see the result, not learn how RLinf works inside, so a recipe is clean, short between commands, and complete enough to follow top to bottom without guessing.

## What a recipe contains

1. **What runs and what you get.** One paragraph after the title, or after a credited figure of the upstream benchmark or model: what it is, what this page runs with RLinf, and the result the reader reaches.
2. **What to prepare.** Hardware, installation, model weights, datasets, and accounts, each with the command or link that provides it.
3. **One unified `Setup and Run` (`安装与运行`) `.. tab-set::` per model, suite, or hardware backend.** Instead of splitting Installation, Download, and Run It into separate sections with repeated tab bars, group the complete end-to-end flow inside ONE top-level `.. tab-set::` (even when there is only one option):
   - Each `.. tab-item::` contains **1. Install** (`git clone` + `install.sh` + `source .venv`), **2. Download** (all companion assets + model weights together), **3. Launch** (only required paths + one directly copy-pasteable launch line, with optional tuning folded into `.. dropdown:: Advanced configuration (optional)`), and optionally that choice's individual training curve.
   - **Hardware backends**: When a recipe supports multiple accelerators (NVIDIA CUDA, AMD ROCm, Huawei Ascend CANN, Moore Threads MUSA), place the `.. _<recipe>-hardware:` anchor right above `Setup and Run` and include hardware backends as **flat tabs in the same single-level `.. tab-set::`** using synced keys (`:sync: cuda`, `:sync: rocm`, `:sync: ascend`, `:sync: musa`) — never nest `.. tab-set::` inside another `.. tab-set::`.
   - **Commands**: Every `install.sh` and `hf download` command must be directly copy-pasteable with zero `<...>` placeholders, zero `cd` before `hf download`, and `--local-dir ./<name>` (or repo-relative asset path). Split **1. Install**, **2. Download**, and **3. Launch** into separate short bash blocks (never combine `git clone` and training/eval launch commands in one mega block), prefer Hydra CLI overrides (such as `actor.model.model_path=...`) over asking readers to edit tracked YAML files, and state multi-node defaults explicitly (or give a `cluster.num_nodes=1` quick-start override) whenever the launched YAML sets `cluster.num_nodes > 1`. Keep `--use-mirror` and `HF_ENDPOINT` mentioned once via `_setup_common.rst` right above the tab-set, and point to [Docker Hub](https://hub.docker.com/r/rlinf/rlinf) for prebuilt images. Omit `pip install huggingface-hub`, `git lfs clone`, and ASCII directory trees of downloaded files.
   - **Tab titles**: Keep `.. tab-item::` titles short (<= 48 chars), consistent across pages, and free of `Model + Benchmark` concatenation (write `OpenPI`, not `OpenPI + RoboCasa`). Keep EN and ZH tab-set and tab-item counts identical (`docs/tab-parity`).
   - **Sequential pipelines / real-robot guides**: Multi-phase workflows (hardware wiring → collection → SFT → RL → deployment) keep sequential section headings, with `.. tab-set::` inside each setup/download step.
   - **Visualization and Results**: Keep shared cross-model benchmark comparison tables outside the tabs under `Visualization and Results`; link to `:doc:`Training metrics <../../reference/metrics>`` and mention TensorBoard / `save_video` in one short line instead of repeating `runner.logger` or `video_cfg` YAML blocks.
4. **The result.** A table of what the run reaches, with the config or command that reproduces it, and where to watch progress (for example `env/success_once` in TensorBoard).
5. **Problems readers hit.** Each titled by the symptom or the error text, with the fix, when such problems are known.
6. **Where to go next.** Links to the metrics page, to Evaluation for standalone evaluation, and to the concept page behind the feature.

## What a recipe leaves out

| Leave out | Where it belongs |
|---|---|
| How RLinf works inside: workers, channels, placement, the training loop | a Concepts page, linked once |
| Class, function, module, and file names the reader never touches | nowhere |
| Algorithm definitions, losses, and equations | Reference |
| Every config key | Reference; the recipe shows only the keys the reader changes |
| What each training metric means; standalone evaluation | the Training metrics page; the Evaluation section |
| Alternative ways to do the same step | nowhere; give the one path that works best |
| Shell argument arrays, long CLI override lists, inline prep scripts, or external env-var workarounds for settings RLinf manages in YAML | the recipe YAML, or a helper script in the repository |
| Duplicated install or download steps already owned by a family setup page or partial | link to the shared page with `:doc:` / `:ref:`, or include the partial |
| Boilerplate model introductions ("Visual Understanding, Language Comprehension…") | nowhere |

Between two commands, write only what the reader needs to run the next one: what it does and why now. When a step needs background, link to the page that explains it.

## Gallery example pages

A page in an Examples gallery follows [templates/recipe.rst](../templates/recipe.rst) (Chinese: [templates/recipe.zh.rst](../templates/recipe.zh.rst)) and matches the other pages of its gallery:

- **Overview** is a card grid of four aligned cards (`.. grid:: 2 4 4 4`), with the same card titles in the same order on every page of the gallery: Simulators and Robots use Models, Algorithms, Tasks, Hardware; Models use Environments, Algorithms, Tasks, Hardware; Algorithms use Algorithm, Models, Environments / Data, Training; SFT uses Models, Methods, Data, Hardware. On an env page, the Models card lists every model supported on that env and the Algorithms card every algorithm. There is no "Env type" card; the env-type string goes in the prose or a table.
- **Tasks** and **Observation and Action** are H3 `list-table`s under Overview, never bullet lists. Models pages use the columns Environment, Task / Suite, Config / Weights, Focus, and the rows Observation, Action, Reward, Prompt.
- **Section names** are standard: `Overview` (`概览`), `Setup and Run` (`安装与运行`, containing Install, Download / Prepare, and Launch inside one flat `.. tab-set::`), and `Visualization and Results` (`可视化与结果`). Do not split separate `Installation`, `Download the Model`, `Run It`, or `Run on Different Hardware Backends` H2 sections alongside `Setup and Run`. Algorithm pages may add a short `How <Method> Works` section before `Setup and Run` that summarizes the loop in a few sentences and links to Reference for the math. Robot pages may add hardware and safety sections after `Observation and Action`.
- **A multi-recipe page** (LIBERO) has no page-level Overview; each recipe family gets its own section with a descriptive, parallel name and its own card grid. Give a `:ref:` to a renamed section explicit link text.
- **Shared steps** reuse what already exists: link to a model- or domain-family setup page (such as `video_models.rst` for shared diffusion installation and checkpoints) with `:doc:` or `:ref:`, or include partials (`_setup_common.rst` for installation and `_model_path.rst` for the model path), keeping only recipe-specific steps on the page.

## Multi-Hardware Backends

A recipe that runs on more than one accelerator places `.. _<recipe>-hardware:` directly above `Setup and Run` (`安装与运行`) and adds each backend as a flat `.. tab-item::` inside the same single-level `.. tab-set::` using synced keys (`:sync: cuda`, `:sync: rocm`, `:sync: ascend`, `:sync: musa`), never as a separate `Run on Different Hardware Backends` H2 section or nested tab-set:

- Name each tab for the vendor stack (`NVIDIA CUDA`, `AMD ROCm`, `Huawei Ascend CANN`, `Moore Threads MUSA`), appending `(Experimental)` / `（实验性）` when convergence is unverified.
- Inside each tab, keep the same numbered flow (**1. Install** → **2. Download / Prepare** → **3. Launch**), noting platform-specific prerequisites, flags, and limits.
- Container and install blocks shared across recipes live in partials named `_<platform>_<env>.rst`; the model, config, and checkpoint stay on the page.
- State the hardware the page was verified on, and name what is untested.
- Every platform with a tab appears in that recipe's row of the `vla_wam_index` hardware table, and every platform in the row has a tab.

## Checklist

- [ ] A reader can follow the steps top to bottom on the stated hardware without fighting complex shell glue, and reach the stated result.
- [ ] Every command is standard and minimal: paths, placement, and defaults live in the recipe YAML, launch uses RLinf's standard script without unnecessary wrappers or long CLI override lists, and every non-obvious step says how to tell it worked.
- [ ] Shared setup (install, base checkpoints, common datasets) links to the family setup page or includes a partial instead of duplicating commands.
- [ ] No internals, no unexplained code names, no equations, and one path per step.
- [ ] The page matches its gallery's section names, card titles, and table columns, and aligns its user workflow (setup, config, launch) with its closest model- or domain-family siblings on `main`.
- [ ] Metrics and standalone evaluation are linked, not explained.
