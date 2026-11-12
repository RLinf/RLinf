# Guide pages

A guide shows a reader who already runs RLinf how to do one task: place components on GPUs, scale to several nodes, mix heterogeneous envs, resume from a checkpoint, collect data. Pages under Guides and Evaluation guides are guides. The reader has a goal and a running setup; the guide gets them to the goal in plain, technical language and stops there.

## What a guide contains

- **A title that names the task**: Configure Heterogeneous Envs, Resume from a Checkpoint, Run on Multiple Nodes. Section headings are tasks too, in the reader's words.
- **When you need it.** One paragraph on the situation that calls for this task and what changes once it is done.
- **What must already be in place**, with links.
- **How it works, briefly.** When the reader cannot set the options sensibly without knowing the mechanism, explain it in a few plain sentences: what runs where, and why a rule exists. Link to the concept page for the full picture.
- **The steps**, in order, one action each, written as instructions. Put a choice as "To get X, set Y".
- **One config example** that contains only the keys the reader writes, and **one worked example** that shows the effect, for instance a small table of which GPU runs which component.
- **Caveats** after the steps, with footguns in a `warning`.
- **Links** to the reference for every key and to related guides.

## What a guide leaves out

| Leave out | Where it belongs |
|---|---|
| Equations, tensor shapes, and derivations | Reference |
| The implementation: classes, functions, modules, files | nowhere |
| Every option and every validation rule | Reference |
| The history of the feature | the PR description or release notes |
| A tour of the alternatives | one recommended way, with the others linked if they matter |

## Tone

Plain and technical. Each sentence says one thing in full words, without telegraphic fragments and without the narrative cadence of a concept page. A guide reads like a colleague's instructions: what to set, why, and what you will see.

## Checklist

- [ ] The title and headings name tasks, and none is a code identifier.
- [ ] The config example shows only keys the reader writes, and the worked example shows their effect.
- [ ] No equations, tensor-shape tables, or internal names.
- [ ] The reader can finish the task without opening the source code.
