---
name: add-publication-docs
description: Adds a new publication page to the RLinf Sphinx docs (EN + ZH) and wires it into the Publications index/toctree and navbar. Use when the user asks to add a publication/paper page, convert a paper README/markdown into docs, or update the publications listing.
---

# Add a Publication to RLinf Docs

This skill adds **one new publication** (paper/tech report/system note) into the RLinf documentation, in **both English and Chinese**, following the existing Publications section conventions.

## What to create

- **EN page**: `docs/source-en/rst_source/publications/<slug>.rst`
- **ZH page**: `docs/source-zh/rst_source/publications/<slug>.rst`
- **Index wiring**:
  - `docs/source-en/rst_source/publications/index.rst`
  - `docs/source-zh/rst_source/publications/index.rst`

Where `<slug>` is lowercase with underscores (match current style, e.g. `rlinf_vla`, `rlinf_user`).

## Required structure (match current pages)

For both EN and ZH pages, keep the same section order as existing publication pages:

1. **Title** (paper title / report title)
2. **Link line** (pick one):
   - EN: `**Paper:** ...` (and optionally `| **Models:** ...`)
   - ZH: `**论文：** ...` (and optionally `| **模型：** ...`)
   - For non-paper pages, use `**Documentation:** ...` / `**文档：** ...`
3. `Overview` / `概述`
4. `Results` / `结果` (tables/figures)
5. `Quickstart` / `快速开始` (links only)
6. `Citation` / `引用` (BibTeX, if applicable)

## Quickstart rules (important)

- **Do not re-iterate full instructions** in publication pages.
- **Quickstart must contain exactly one link** and it must point to the **corresponding existing example page** (no extra links).
  - Embodied: `:doc:\`../examples/embodied/<benchmark_or_platform>\``
  - Agentic/reasoning: `:doc:\`../examples/agentic/<task>\``
  - Real-world: `:doc:\`../examples/embodied/franka\``
- If there is no suitable example page yet, create that example page first; do not add installation / generic guides to Quickstart.

## Tables & figures rules

- Prefer `.. list-table::` for tables.
- If you use `:widths:`, **the number of widths must equal the number of columns**.
- External figures are allowed (e.g. GitHub raw URLs) via:

```rst
.. image:: https://example.com/fig.png
   :alt: caption
   :align: center
```

- If you need side-by-side images, `.. raw:: html` is acceptable (follow existing RLinf pages).

## Wire the publication into the index (order matters)

In both EN and ZH `publications/index.rst`:

- Add `<slug>` under the `.. toctree::` block **in the exact order you want it shown**.
- Add a bullet entry that matches the same order and naming.

Sphinx displays pages in the **toctree listing order**.

## Checklist

- [ ] Add EN publication page under `docs/source-en/.../publications/`
- [ ] Add ZH publication page under `docs/source-zh/.../publications/`
- [ ] Update EN/ZH `publications/index.rst` toctree + bullets (desired order)
- [ ] Ensure Quickstart is exactly one link to the corresponding example page
- [ ] Validate list-table `:widths:` counts match column counts

## Example (minimal new publication page)

```rst
My Paper Title
==============

**Paper:** `arXiv:XXXX.XXXXX <https://arxiv.org/abs/XXXX.XXXXX>`__

Overview
--------

One-paragraph summary.

Results
-------

.. list-table:: Main results
   :header-rows: 1
   :widths: 40 20 20

   * - Setting
     - Metric A
     - Metric B
   * - Method
     - 1.23
     - 4.56

Quickstart
----------

- :doc:`../examples/embodied/<benchmark_or_platform>`

Citation
--------

.. code-block:: bibtex

   @article{...}
```

