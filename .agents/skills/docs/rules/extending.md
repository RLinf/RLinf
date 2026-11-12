# Extending pages

An Extending page shows a contributor how to add a component to RLinf: an env, a model, a robot, a real-world task, an SFT model, a reward. The reader writes code, so this is the one page type where code names belong in the text, but only the ones the contributor creates, calls, implements, or registers. Explain the way a concept page does ([concept.md](concept.md)): idea first, then the name.

## What an Extending page contains

- **What you add and what works afterwards**, in one paragraph.
- **What to prepare**: a working install, the upstream package, weights or assets.
- **The files you create or edit**, and the extension point the component plugs into (the registry, enum, or base class), in the order you touch them.
- **One example that closes the loop**: the new component, its registration, the config that selects it, and the command that runs it with components that already exist. An isolated class definition shows syntax, not how the component fits.
- **Each public call the example uses**, in the order the contributor makes it: what it takes, what it returns, and why it is needed at that point. Do not leave several methods in the example unexplained.
- **How to test it**: the unit test file the cases go into and the command that runs an end-to-end check.

## What an Extending page leaves out

- Internals the contributor does not touch, such as scheduler or placement internals; link to the concept page.
- Base-class methods the contributor does not implement.
- Install, Docker, and CI wiring for a new model or env, which is a maintainer workflow; link to the contributing guide instead of repeating it.

## Checklist

- [ ] Every code name on the page is one the contributor writes, calls, implements, or registers.
- [ ] The example runs end to end: component, registration, config, command.
- [ ] Every public call in the example is explained in call order.
