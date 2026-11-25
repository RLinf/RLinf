# Extending pages

An Extending page shows a contributor how to add a component to RLinf: a model, simulator env, real-world robot or task, reward model, algorithm, agent loop, or accelerator backend. The reader writes code, so this is the one page type where code names belong in the text, limited to the files and public APIs the contributor creates, calls, implements, or registers. Explain the way a concept page does ([concept.md](concept.md)): idea first, then the name.

## What an Extending page contains

- **What you add and what works afterwards**, in one short, plain-language opening paragraph (lead with what kind of component this is and a concrete example—no dense implementation recap).
- **An At-a-Glance summary table** (`.. list-table::`) near the top of the page (within the first 60 lines) listing the component, extension point, and role.
- **When showing a runner/orchestration diagram**, highlight only the user-implemented workers, interfaces, and connecting channels; keep unrelated built-in components (such as Rollout, Actor, or Session Server) as compact boxes without expanding their internals, and do not repeat the diagram in a long prose paragraph underneath.
- **Consistent component-section order**: for each component the contributor implements (for example `ToolWorker`, `AgentLoopWorker`, `BasePolicy`), structure the section in three concise subsections:
  1. **Interfaces to Implement** — a short `.. list-table::` of required vs. optional methods (plain-language role, no dense variable/signature walls).
  2. **Implementation Example** — one minimal, runnable code block.
  3. **Call Flow** — 3–4 short plain-language steps showing how RLinf calls those methods end to end.
- **What to prepare**: a working install, upstream package, or weights/assets.
- **One minimal, runnable example that closes the loop**: the new component, its registration (including `RLINF_EXT_MODULE` out-of-tree option when applicable), the config that selects it, and the launch command. An isolated class definition shows syntax, not how the component fits.
- **A standalone local CPU verification snippet**: a short Python command the contributor can run locally (without GPU or hardware) to verify registration, shapes, or step/forward contracts directly.
- **Keep `Extending` sub-indexes self-contained**: every primary path on an `extending/**/index.rst` page links to a guide under `extending/`, not across chapters into `guides/`.

## What an Extending page leaves out

- **Boilerplate `tests/unit_tests/` and `pytest` sections.** Extending guides teach contributors how to build and verify their extension locally, not how to wire repository CI tests; keep unit-test conventions in `AGENTS.md`.
- Internals the contributor does not touch, such as scheduler or placement internals; link to the concept page.
- Base-class methods the contributor does not implement.
- Install, Docker, and CI wiring for a new model or env; link to the contributing guide instead of repeating it.

## Checklist

- [ ] Opens with a clear At-a-Glance `.. list-table::` near the top of the page.
- [ ] Every code name on the page is one the contributor writes, calls, implements, or registers, and exists in the codebase (`docs/symbol`).
- [ ] The example runs end to end: component, registration, config, command.
- [ ] Ends with a standalone local CPU verification snippet (no `tests/unit_tests/` or `pytest` boilerplate).
