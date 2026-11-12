# Publication pages

A publication page under `resources/publications/` summarizes one paper or technical report and sends the reader to the recipe that reproduces it. Start from [templates/publication.rst](../templates/publication.rst). Both languages have these sections, in this order:

1. The paper title as the H1.
2. One link line: `**Paper:** <link>` (`**论文：**`), optionally followed by `| **Models:** <link>` (`| **模型：**`). A page about something other than a paper uses `**Documentation:**` (`**文档：**`).
3. Overview (概述): what the work changes or which problem it solves, before any project-specific name.
4. Results (结果): tables as `list-table`, with as many `:widths:` values as columns, and figures.
5. Quickstart (快速开始): exactly one link, to the example page that runs the work. When that page does not exist yet, write it first; the publication page does not repeat install or run steps.
6. Citation (引用): the BibTeX, when the work has one.

The publications index lists each page as a row of its `list-table` and in its hidden toctree, in the order the list shows. The slug is lowercase with underscores (`rlinf_vla`).
