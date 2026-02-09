# Reference: Example RST Template and Index Snippets

## Minimal RST template for a new example

```rst
RL with [Benchmark Name] / RL on [Model Name]
=============================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document provides a guide to [model] training in the [environment] using RLinf.

Environment
-----------

**[Environment Name]**

- **Environment**: [one line]
- **Task**: [one line]
- **Observation**: [image size, state dims]
- **Action Space**: [dim and description]

**Task Description Format**

.. code-block:: text

   [In/Out format if applicable]

**Data Structure**

- **Images**: shape ``[batch_size, H, W, 3]``
- **Task Descriptions**: Natural-language instructions
- **Actions** / **Rewards**: as applicable

Algorithm
---------

**Core Algorithm Components**

1. **PPO** (or GRPO, etc.) — bullet points
2. **Model** — short note

Dependency Installation
-----------------------

1. Clone and install (see existing examples for Docker vs pip blocks).

Quick Start
-----------

**Training**

.. code-block:: bash

   bash examples/embodiment/train_embodiment.sh <config_name>

**Key config**: mention main YAML and important keys (e.g. ``rollout.model.model_path``, ``env.*``).

Evaluation (optional)
---------------------

.. code-block:: bash

   bash examples/embodiment/eval_embodiment.sh <eval_config_name>
```

## Index toctree entry

In `docs/source-en/rst_source/examples/index.rst`, add the module name (no `.rst`) to the toctree:

```rst
.. toctree::
   :hidden:
   :maxdepth: 2

   ...
   <name>   # e.g. dexbotic
```

## Gallery card (optional)

Same HTML block pattern as existing cards; one card:

```html
<div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
  <img src="https://..." style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
  <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
    <a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/<name>.html" target="_blank" style="text-decoration: underline; color: blue;">
      <b>Display Title</b>
    </a><br>
    Short description
  </p>
</div>
```

Place inside an existing `.. raw:: html` block in the right section (Embodied Intelligence / Reasoning / Agent).

## README updates (required)

Update both READMEs so the new example is visible on the repo front page.

**What's NEW! / 最新动态** — Add a new bullet at the **top** of the list (below the section heading):

- **README.md** (English):  
  `- [YYYY/MM] 🔥 RLinf supports ... [Project Name](https://github.com/...). Doc: [Display Title](https://rlinf.readthedocs.io/en/latest/rst_source/examples/<name>.html).`
- **README.zh-CN.md** (Chinese):  
  Same content in Chinese; doc URL must use `https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/<name>.html`.

Use the current year/month for the date. Match the tone of existing bullets (e.g. "RLinf supports reinforcement learning fine-tuning for [X]" / "基于[X]的强化学习微调已经上线！文档：...").

**Key Features / 核心特性 table** — If the example is a simulator, model, or real-world platform, add it to the matching list in the Embodied AI / 具身智能 table:

- **Simulators / 模拟器**: `<li><a href=".../examples/<name>.html">Display Name</a> ✅</li>` (README.md uses `/en/`, README.zh-CN.md uses `/zh-cn/` in the base URL).
- **Models / 模型** (VLA, World Model, etc.): add under the appropriate sub-list.
- **Real-world / 真机**: add if the example is real-robot.

Place the new `<li>...</li>` above the "More..." / "更多..." item when present.

## vla-eval.rst environment list entry

```rst
:doc:`Display Name <../examples/<name>>`
```

## Chinese documentation (required)

Add a matching RST and index entry for Chinese:

- **RST**: `docs/source-zh/rst_source/examples/<name>.rst` — same structure as the English file; mirror sections and adapt links (e.g. `/en/` → `/zh/` in doc URLs if applicable).
- **Index**: `docs/source-zh/rst_source/examples/index.rst` — add `<name>` to the toctree; add a gallery card in the same category if one was added for English.
- **Start/vla-eval**: If the example is an embodied eval environment, check `docs/source-zh/rst_source/start/` for a vla-eval or equivalent page and add the new example to the list.

Use existing en/zh pairs (e.g. `docs/source-en/rst_source/examples/libero.rst` and `docs/source-zh/rst_source/examples/libero.rst`) as reference.

## Existing examples to copy from

- **English**: `docs/source-en/rst_source/examples/` — e.g. `libero.rst`, `maniskill.rst`, `pi0.rst`, `gr00t.rst`
- **Chinese**: `docs/source-zh/rst_source/examples/` — same filenames; use as pairs with the English versions
