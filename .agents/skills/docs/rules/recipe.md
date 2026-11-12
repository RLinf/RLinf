# Recipe pages

A recipe takes the reader from a fresh machine to a working run of one model, env, algorithm, or robot setup. The Examples galleries, the Get Started quickstarts, and the Evaluation get-started pages are recipes. The reader wants to run the thing and see the result, not learn how RLinf works inside, so a recipe is clean, short between commands, and complete enough to follow top to bottom without guessing.

## What a recipe contains

1. **What runs and what you get.** One paragraph after the title, or after a credited figure of the upstream benchmark or model: what it is, what this page runs with RLinf, and the result the reader reaches.
2. **What to prepare.** Hardware, installation, model weights, datasets, and accounts, each with the command or link that provides it.
3. **The steps, in order.** Each step's heading names the action (Installation, Download the Model, Run It). Each command is introduced by one sentence on what it does, followed by the keys or flags the reader may change, and by the expected output or the sign that the step worked.
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
| Boilerplate model introductions ("Visual Understanding, Language Comprehension…") | nowhere |

Between two commands, write only what the reader needs to run the next one: what it does and why now. When a step needs background, link to the page that explains it.

## Gallery example pages

A page in an Examples gallery follows [templates/recipe.rst](../templates/recipe.rst) (Chinese: [templates/recipe.zh.rst](../templates/recipe.zh.rst)) and matches the other pages of its gallery:

- **Overview** is a card grid of four aligned cards (`.. grid:: 2 4 4 4`), with the same card titles in the same order on every page of the gallery: Simulators and Robots use Models, Algorithms, Tasks, Hardware; Models use Environments, Algorithms, Tasks, Hardware; Algorithms use Algorithm, Models, Environments / Data, Training; SFT uses Models, Methods, Data, Hardware. On an env page, the Models card lists every model supported on that env and the Algorithms card every algorithm. There is no "Env type" card; the env-type string goes in the prose or a table.
- **Tasks** and **Observation and Action** are H3 `list-table`s under Overview, never bullet lists. Models pages use the columns Environment, Task / Suite, Config / Weights, Focus, and the rows Observation, Action, Reward, Prompt.
- **Section names** are standard: Overview, Installation, Download the Model (and Download the Assets when needed), Run It, Visualization and Results. Omit Download the Model only when there is no checkpoint. Algorithm pages add a short How <Method> Works section before setup, which says what the method does in a few sentences and links to Reference for the math. SFT pages put dataset and model preparation before Installation. Robot pages may add hardware and safety sections after Observation and Action.
- **A multi-recipe page** (LIBERO) has no page-level Overview; each recipe family gets its own section with a descriptive, parallel name and its own card grid. Give a `:ref:` to a renamed section explicit link text.
- **Shared steps** come from partials: `_setup_common.rst` for installation and `_model_path.rst` for the model path.

## Run on Different Hardware Backends

A recipe that runs on more than one accelerator keeps the non-CUDA platforms in one section after the CUDA steps, named Run on Different Hardware Backends, with a `_<recipe>-hardware` anchor that the Hardware card and the support table link to. Inside it:

- One H3 per platform, named for the vendor stack: AMD ROCm, Huawei Ascend CANN, Moore Threads MUSA.
- Each covers the same four things in order: how to start the container or install natively, what the platform changes about the run (device flags, rendering, simulator backend), the launch command, and the limits a reader would otherwise find at runtime.
- Container and install blocks are partials named `_<platform>_<env>.rst`; the model, config, and checkpoint stay on the page.
- State the hardware the page was verified on, and name what is untested.
- Every platform with a subsection appears in that recipe's row of the `vla_wam_index` hardware table, and every platform in the row has a subsection.

## Checklist

- [ ] A reader can copy the commands top to bottom on the stated hardware and reach the stated result.
- [ ] Every command is introduced by what it does, and every non-obvious step says how to tell it worked.
- [ ] No internals, no unexplained code names, no equations, and one path per step.
- [ ] The page has its gallery's section names, card titles, and table columns, in the same order as its sibling pages.
- [ ] Metrics and standalone evaluation are linked, not explained.
