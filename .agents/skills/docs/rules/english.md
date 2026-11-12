# Writing English

An English page reads as if an engineer who writes English natively explained the system to a colleague: direct, concrete, and in the field's usual terms. These rules hold for docs pages and for all other project writing in English: PR descriptions, review comments, commit messages, and issues.

## Voice

- Address the reader as "you" and give instructions in the imperative: "Set `cluster.num_nodes`", "Run the script". Use "we" and "let's" only to walk through an example on a concept page, and sparingly there.
- Write in the present tense and the active voice: "The env worker sends the observations", not "The observations are sent".
- Lead with the outcome: say what the reader gets or does, then how.
- Write "Use X" instead of "Consider using X" or "You may want to use X"; keep "may" and "might" for real uncertainty.
- Use full sentences. On recipe and guide pages keep them short, one idea each; on concept pages let them finish their thought ([concept.md § Cadence](concept.md#cadence)).

## Words to cut

| Cut | Why | Write instead |
|---|---|---|
| simply, just, easily, obviously, basically, of course | the step may not be simple for the reader | nothing |
| seamlessly, powerful, robust, comprehensive, cutting-edge, state-of-the-art, leverage, crucial, pivotal, delve, showcase, unlock, empower | sales or machine register | the concrete fact, or nothing |
| it is worth noting that, note that, please note, importantly | filler | the fact itself, or a `note` |
| in order to, allows you to, enables you to, is able to | wordy | to, you can, can |
| currently, new, now, no longer, recently | goes stale | the version, or nothing |
| a wide range of, various, a variety of | vague | the actual items or number |

## Constructions to avoid

- "Not just X, but Y" and "X rather than Y" when nothing suggested X.
- A trailing "-ing" clause that claims a benefit: "…, ensuring reliable training". State the consequence, or cut it.
- Lists and adjective series padded to three items. List the real number.
- A closing sentence that praises the design or restates the paragraph, and a closing section that summarizes the page.
- "serves as", "acts as", "boasts", "features" where "is" or "has" is meant.
- Announcements instead of content: "In this section, we will…", "As mentioned above", "Notice that…", "Let's take a look at…".
- Rotating synonyms for one thing (env, environment, simulator, task); use the term in [glossary.md](glossary.md).
- Questions as transitions ("So how does this work?") on recipe and guide pages.

## Calques from Chinese

| Avoid | Write |
|---|---|
| video memory | GPU memory |
| realize the function, realize support | implement, support |
| carry out training, perform the training | train |
| open or close the switch | enable or disable |
| the relevant configurations | the config keys (name them) |
| in the case of | if, for |
| based on (for 基于, everywhere) | with, using, on, or restructure |
| it is necessary to | you must, or the imperative |

## Headings

- Title Case, as [base.md](base.md#structure) defines it.
- Task headings start with a verb (Install, Run It, Configure Placement); concept headings are noun phrases (Placement Strategies). A heading never needs the body to make sense.

## Other project writing

PR descriptions, review comments, commit messages, and issues follow the voice and the words above, and the code-name rule of [base.md](base.md#write-about-what-the-reader-does-not-about-the-code): say what changed in behavior first, in words the reader knows, and name a file, class, or function only where the reader needs it to find the change. Write each paragraph on one line, since GitHub shows every newline as a line break.
