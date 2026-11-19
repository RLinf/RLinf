# Reference pages

A reference page lists facts a reader looks up: config keys, API signatures, algorithm definitions, metrics, evaluation options. Pages under Reference and Evaluation reference are reference pages. The reader arrives with a name and wants its meaning, so the page is austere, complete, and organized like the thing it describes.

- **Lead** with what can be looked up and how the entries are ordered.
- **Mirror the structure** of the code or config it describes, with the same fields for every entry: name, type, default, meaning, and an example value when it helps.
- **Identifiers may be headings** here, since they are the reader's lookup target.
- **Generate from docstrings** (autodoc) where an API is documented, instead of copying signatures by hand.
- **Equations and tensor shapes belong here**, each with one sentence on what it computes.
- **Leave out tutorials and narrative**; link to the guides and concept pages that use each entry.
