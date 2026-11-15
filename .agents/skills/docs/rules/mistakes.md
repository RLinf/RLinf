# Mistakes agents make in docs and prose

Agents that write docs repeat the same mistakes, and the reader pays for each one. Go through this list before you finish a page, a README entry, a PR description, or a review comment. Each line names the mistake and the fix; the rule behind it is in the file named at the end of the section.

## What goes on the page

| Mistake | Fix |
|---|---|
| Naming classes, functions, modules, and files the reader never touches, or explaining a mechanism by touring the source tree | Describe what the reader sees and controls; name code only when the reader types, runs, opens, or edits it |
| Explaining RLinf's internals on a recipe or guide page | Link to the concept page once and stay on the task |
| Documenting every config key, option, and alternative | Show the keys the reader writes and the one path that works best; link to Reference for the rest |
| Equations and tensor-shape tables on recipe or guide pages | Move them to Reference |
| Telling the history of a change ("previously", "no longer", "after the refactor") | Describe what exists now; history goes in the PR description |
| Sentences that only say what is not needed ("ROS is not required") | State the requirement positively |
| Only the happy path: no prerequisites, no expected output, no sign that a step worked | Add what to prepare, the output to expect, and the known problems with their fixes |
| Claims with no evidence ("significantly faster", "state-of-the-art") | Give the number and its source, or cut the claim |
| Copying a block that another page already has | Link to that page, or move the block into a partial |
| Creating a new page where an existing one should grow, or listing a page in two galleries | Search the docs first; extend the page that owns the topic |
| Mixing page types: a recipe that turns into a tutorial on internals, a concept page full of procedures | Split by type and link between the parts |

Rules: [base.md](base.md), the page type's file.

## Accuracy

| Mistake | Fix |
|---|---|
| Invented config keys, flags, scripts, paths, or model and env names, often guessed from similar projects | Copy every name from the code or from a run you made; the docs check reports names the repository does not define on concept, extending, and reference pages, so check the others by hand |
| Behavior inferred from a name, or the intended design described instead of the code | Read the code that runs |
| Commands nobody ran, and output nobody saw | Run the command, and show only output you observed |
| Values copied from an older sibling page: model repos, config names, metrics | Check them against the code and [../reference.md § Asset references](../reference.md#asset-references) |
| Changing one language only | Change both pages in the same pass |

## Structure

| Mistake | Fix |
|---|---|
| An opening that restates the title or clears its throat ("This page provides a comprehensive guide to…") | The first sentence says what the reader can do or learn |
| A closing summary that repeats the page ("In summary, …") | End with where to go next, or just end |
| Bullet walls with bold labels, and bold everywhere | Explain in prose; keep lists for steps and enumerations |
| Headings named after code identifiers, or headings that need the body to make sense | Name the task or the idea |
| Sections that open with a code block, a table, or an API name | Open by saying what the section answers and how it follows the previous one |
| Paragraphs of the same length that each end by praising the design | Vary the shape; cut the praise |
| Index pages made of bullet lists | Cards or a `list-table`, with a hidden toctree |

Rules: [base.md](base.md), [index.md](index.md), [concept.md](concept.md).

## English

Filler (simply, note that, in order to), sales words (seamlessly, robust, leverage), "not just X but Y", trailing "-ing" benefit clauses, lists padded to three, hedging ("you may want to consider"), synonyms that rotate for one thing, and calques from Chinese ("video memory", "carry out training"). Rules and replacements: [english.md](english.md).

## Chinese

| Mistake | Fix |
|---|---|
| Translating the English draft sentence by sentence | Write from the page plan in Chinese order; split and merge sentences freely |
| 被 sentences, 进行 + noun, 当……时, 对于……来说, 通过……的方式, 允许你…… | Make the actor the subject and use the verb directly |
| 值得注意的是, 首先/其次/最后 on every sentence, 不是……而是…… | State the fact; let the order of events connect the sentences |
| 您, 「」, half-width punctuation in Chinese sentences, missing spaces next to English or digits | 你 or no subject, “”, full-width punctuation, a space between Chinese and English or digits |
| Hard-wrapped Chinese paragraphs, and markup glued to Chinese characters | One source line per paragraph; escaped spaces around markup |
| Chatty phrasing (看看长什么样) or bureaucratic phrasing (本文旨在, 进行相关操作) | Restrained written technical Chinese |
| Translating code names or rendering policy as 策略 | Keep identifiers and the terms in [glossary.md](glossary.md) |

Rules and examples: [chinese.md](chinese.md).

## Process

| Mistake | Fix |
|---|---|
| Building or checking one language only | Run `python3 .agents/skills/docs/check.py`, which builds both |
| Hardcoded Read the Docs links inside the docs | `:doc:` and `:ref:` |
| Leftovers: TODO markers, `[your model]`, "I hope this helps", notes to the user | Remove them before you finish |
| Emoji in headings or prose | None; the README news entry keeps its 🔥 |
| Reviewing your own prose in the context that wrote it | Run the language pass in a fresh agent that sees only the page and the language rules |

## Prose outside the docs

PR descriptions, review comments, commit messages, and replies to the user fail in the same ways. Lead with what changed or what is wrong in words the reader knows, and name a file, class, or function only where the reader needs it to find the change. Report what ran and what it showed, including failures, rather than asserting that things work. Cut the history of how you got there, the hedging, and the summary that repeats the message.
