# Concept pages

A concept page explains one part of how RLinf works to a reader who meets it for the first time: what it is, which problem it solves, how it behaves, and how it fits into a run. Pages under Concepts are concept pages, and Extending pages use the same way of explaining ([extending.md](extending.md)). The reader wants to understand, not to finish a task, so the page is a continuous explanation in clear prose, not a list of facts or a tour of the code.

## What a concept page contains

- **A lead.** One or two sentences that say what the thing is and which problem it solves, in words a newcomer understands. Then the scope of the page and the order in which it explains the topic.
- **The explanation, in the reader's terms.** Start from a situation the reader recognizes, explain the idea in ordinary words and the distinction that matters, and only then give the name they can search for, once. Edge cases come after the normal case is clear.
- **The normal use first.** Move from using the part as it is, to adding to it, to combining it with other parts, to running it distributed, and stop before internals the reader never needs.
- **One example.** A short config or code example that shows the idea in use, introduced by the result it demonstrates and followed by what the reader should notice.
- **The reasons.** Why the design is the way it is, where that is not obvious, and its limits.
- **Where to go next.** Links to the guides that use it and to the reference that lists its options.

## What a concept page leaves out

| Leave out | Where it belongs |
|---|---|
| Step-by-step procedures and command sequences | a guide or recipe, linked |
| Option and key tables | Reference |
| The class hierarchy as the outline of the page | nowhere; follow the reader's questions |
| Code names beyond the few terms the reader searches for | nowhere; describe the behavior |
| Several unfamiliar methods in one example | a smaller example, or an Extending page |

## Build a continuous article

- **The page has a lead, and so does every section.** A section opens by connecting to what came before and naming the question it answers, not with a code block, a table, or an API name.
- **Paragraphs form a chain:** the distinction, then the mechanism, then the example, then what it means, then the next concern. If the paragraphs can be reordered without changing the explanation, the section is a fact list; rewrite it.
- **Sentences follow from one another**, as cause, consequence, contrast, or example. A sentence that could be dropped or moved unnoticed is filler.
- **Examples are framed:** before a code block, the result it shows; after it, what changed and what the reader can now decide. Do not narrate it line by line.
- **The outline test.** Read the introduction and the first paragraph of each section alone. They should form a coherent outline, and the full page should only fill it in.

## Cadence

- **Concise, not clipped.** Let a sentence finish its thought. A page where every sentence starts cold and stops at the first period reads like a spec sheet.
- **Take the reader with you.** A few guiding phrases per page ("Say your gripper hangs off the arm's own connection", "Now that the parts are declared…") give the page a spine; one in every paragraph becomes a tic.
- **Vary paragraph shape.** Some paragraphs are one sentence and some are five. Uniform three-sentence paragraphs sound machine-written.
- **Explain the mechanism, skip the pitch.** Do not end paragraphs with a sentence that says the design is good; the reader who saw the mechanism sees the benefit.
- **Say the thing; don't announce it.** "The builder never mentions the gripper", not "Notice that the builder never mentions the gripper".

## Checklist

- [ ] The lead says what the thing is and which problem it solves, before any code name.
- [ ] Every name the page uses is explained in words first, and the page uses as few as the reader needs to search the code.
- [ ] The outline test passes, and no section is a reorderable list of facts.
- [ ] No procedures, option tables, or tours of the source tree.
