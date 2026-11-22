# Extending pages

An Extending page shows a contributor how to add a component to RLinf: a model, simulator env, real-world robot or task, reward model, algorithm, agent loop, or accelerator backend. The reader writes code, so this is the one page type where code names belong in the text, limited to the files and public APIs the contributor creates, calls, implements, or registers. Explain the way a concept page does ([concept.md](concept.md)): idea first, then the name.

## What an Extending page contains

- **What you add and what works afterwards**, in one opening paragraph.
- **An At-a-Glance summary table** (`.. list-table::`) near the top of the page (within the first 60 lines) listing the extension point, base class or decorator, config selector, and files touched.
- **What to prepare**: a working install, upstream package, or weights/assets.
- **One minimal, runnable example that closes the loop**: the new component, its registration (including `RLINF_EXT_MODULE` out-of-tree option when applicable), the config that selects it, and the launch command. An isolated class definition shows syntax, not how the component fits.
- **Each public call the example uses**, in call order: what it takes, what it returns, and why it is needed at that point.
- **A standalone local CPU verification snippet**: a short Python command the contributor can run locally (without GPU or hardware) to verify registration, shapes, or step/forward contracts directly.

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
