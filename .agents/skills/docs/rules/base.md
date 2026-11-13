# Rules for every page

Every RLinf docs page follows this file, whatever its type. Before you write, read this file, the file for the page's type, and the file for each language you write in ([english.md](english.md), [chinese.md](chinese.md)). Before you finish, go through [mistakes.md](mistakes.md).

| Page type | Where it lives | Rules |
|---|---|---|
| Recipe: run one model, env, algorithm, or robot setup | `examples/`, `start/`, `evaluations/get_started/` | [recipe.md](recipe.md) |
| Guide: do one task with RLinf | `guides/`, `evaluations/guides/` | [guide.md](guide.md) |
| Concept: how one part of RLinf works | `concepts/` | [concept.md](concept.md) |
| Extending: add a component to RLinf | `extending/` | [extending.md](extending.md) |
| Reference: facts to look up | `reference/`, `evaluations/reference/` | [reference.md](reference.md) |
| Index: route readers to other pages | every `index.rst` and `*_index.rst` | [index.md](index.md) |
| Publication | `resources/publications/` | [publication.md](publication.md) |
| README highlight | `README.md`, `README.zh-CN.md` | [readme.md](readme.md) |

Other Resources pages (FAQ, blog, release notes) follow this file and the language files. A FAQ entry states the symptom first, then the fix.

## Write about what the reader does, not about the code

The reader is an engineer who uses RLinf and has not read its source. Nobody can recite the class names, function names, or file paths of a codebase they did not write, so a sentence built on them explains nothing to that reader.

- Describe what the reader sees and controls: the parts of a run, what they configure, what happens when they launch it, and what result to expect.
- Name a piece of code only when the reader types, runs, opens, or edits it: a command, a config key they set, a path they pass, and, on Extending pages, the files and APIs a contributor writes against. Leave out every other class, function, module, and file, however precise it would make the sentence.
- When a concept page must name an API, explain the idea in plain words first, then give the name once, as the term the reader can search for.
- Never explain a mechanism by touring the repository or listing the modules a call passes through.

Rewrite such a sentence around the behavior instead of trimming it:

- Before: "`EnvWorker.interact` calls `EnvManager.step`, which sends the observation through `Channel.put` to `MultiStepRolloutWorker`."
- After: "Each env worker steps its simulators and sends the observations to the rollout workers, which reply with the next actions."

## Lead with the purpose

- The first sentence after the title, or after a leading figure, says what the page lets the reader do, understand, or look up. Background comes after it, never before.
- Describe what exists now. Leave out the history of a change and sentences whose only content is that something is not needed, not included, or no longer used; state the requirement positively. Keep a negative only when it answers a decision the reader is making ("the controller computer needs no GPU").

## Be accurate

- Check every name, command, config key, path, and value against the code before you write it. Model and env names are the string values of `SupportedModel` in `rlinf/config.py` and `SupportedEnvType` in `rlinf/envs/__init__.py`.
- Read the code instead of guessing behavior from a name, and describe what the code does today, not what a design intends.
- When a change alters a public name, a type relationship, or a lifecycle, update every page that describes it in the same change.
- Use one term per concept across the docs ([glossary.md](glossary.md)).

## Structure

- One H1 per page. Headings are Title Case (lowercase only a, an, the, and, or, of, to, in, on, with, vs), never skip a level, and use the standard section names of the page type.
- A heading names a task or an idea the reader already understands, not an unexplained code identifier. Reference pages may use identifiers as headings.
- Explain in prose; use lists for steps and enumerations, and tables for specs, comparisons, and results (`list-table`). Avoid bullet walls with bold labels.
- Put footguns in a `warning` (out-of-memory settings, `MUJOCO_GL`, the `RLINF_NODE_RANK` ordering, multi-node pitfalls) and side information in a `note`.

## Commands and code blocks

- Before a command, say in one sentence what it does. After a non-trivial one, say what it does in order and where to configure it further.
- Show the expected output in its own block when it tells the reader the step worked.
- Write commands without a `$` prompt. Write placeholders as `<name>` and explain each one.
- Show only the config keys the reader sets, with the rest of the file left to its defaults.

## Links and reuse

- Link to reference material instead of copying it: full config tables, the metrics list, standalone evaluation workflows. Each page does one job.
- A block repeated on three or more pages lives in an underscore partial (`_name.rst`, excluded from the build) pulled in with `.. include::`. Substitutions do not work inside code blocks, so a partial holds only identical text.
- Link inside the docs with `:doc:` and `:ref:`, never a hardcoded Read the Docs URL. Link text names the target. URLs are versioned, so moving a page changes its URL; do not add redirects.

## Images and media

- Every figure and media URL returns HTTP 200 directly. Use `https://raw.githubusercontent.com/<org>/<repo>/<branch>/<path>`, not a redirecting `github.com/<org>/<repo>/raw/...` URL, and point at a repository's current name after a rename. RLinf's images live in `RLinf/misc` under `pic/` and its subfolders.
- Prefer a static image to a large GIF. A caption says what this figure shows on this page, and RLinf-owned images carry no credit suffix.
- A content image gets a white background in dark mode; a logo or diagram that should stay transparent overrides it with `background-color: transparent !important`.

## Two languages

- Every page exists at the same path under `docs/source-en/` and `docs/source-zh/`. Change both in the same pass, and keep the same sections, commands, config keys, numbers, and links.
- Write each language from the page plan, in its own idiom; the sentences need not correspond.
- Never translate an identifier: config keys, env and model names, CLI flags, script names.
