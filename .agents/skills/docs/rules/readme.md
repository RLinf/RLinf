# README highlights

A feature worth announcing (a new model, env, algorithm, rollout or training backend, accelerator, robot, or major capability) is highlighted in `README.md` and `README.zh-CN.md` in the same change. Bug fixes, refactors with no user-visible change, and internal tooling get no entry.

- Add a dated entry at the top of "What's NEW!" (`最新动态`), in the form `- [YYYY/MM] 🔥 RLinf supports <feature> <with what, where>. Doc: [<page title>](<docs link>).` Write the Chinese entry as Chinese ([chinese.md](chinese.md)) rather than as a translation of the English one.
- Add an item to the matching support list (the embodied Simulators, Models, or Algorithms list, the Hardware Support table, or the Agentic AI table), above its "More..." (`更多...`) item when there is one.
- Link to the rendered page: `https://rlinf.readthedocs.io/en/latest/rst_source/<path>.html` from `README.md`, and the `zh-cn` URL from `README.zh-CN.md`. READMEs use these absolute links; docs pages use `:doc:` and `:ref:`.
- An entry says what the reader can now do, in plain words, with no sales language and no internal names.
