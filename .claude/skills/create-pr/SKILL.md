---
name: create-pr
description: Open a GitHub pull request for RLinf, or fix an existing one — checks the PR title against Conventional Commits, writes a precise description that follows .github/PULL_REQUEST_TEMPLATE.md, and lints both before pushing. Use when asked to create/open/submit a PR, write or rewrite a PR description, or fix a PR title or body.
---

# Create a PR

Write the title and the description, then **run the linter before you post**:

```bash
python3 .claude/skills/create-pr/lint_pr.py lint \
    --title "fix(libero): reset action queues between episodes" \
    --body-file /tmp/pr-body.md
```

Exit code 1 means at least one ERROR — fix it and re-run. It is the same
checker you point at a PR that is already open:

```bash
python3 .claude/skills/create-pr/lint_pr.py lint --pr 1444
```

The rules come from [CONTRIBUTING.md](../../../CONTRIBUTING.md) ("PR Title and
Description"), [.github/PULL_REQUEST_TEMPLATE.md](../../../.github/PULL_REQUEST_TEMPLATE.md),
and — for the title — the required `pr-title-check` job, which runs
[`openGemini/pr-title-checker`](https://github.com/openGemini/pr-title-checker)
with `strict: true`. Every title rule that job enforces is an ERROR in the
linter, so a clean `lint` means the check passes. Paths below are relative to
the repo root.

## Why this exists

Over the 25 most recently merged PRs the linter reports errors on **11 of
them**. The dominant failures, in order: template sections dropped (14),
`Types of changes` / `Checklist` with nothing ticked (9), template
`<!--- ... -->` hints left in (9), prose above the first heading (5). PR
[#1421](https://github.com/RLinf/RLinf/pull/1421) is the worst case — the whole
body was pasted with a two-space indent, which nests every `###` heading inside
the Description and breaks the results table outright.

## 1. Gather the facts before writing a word

Never describe the PR from memory of what you intended. Read the diff:

```bash
git fetch origin main
git log --oneline origin/main..HEAD
git diff --stat origin/main...HEAD
```

You need, concretely:

- **Which component the diff touches** — that becomes the scope.
- **Every user-visible change** — a new config key, a renamed flag, a moved doc
  page, a changed default. Internal churn does not go in the description.
- **What you actually ran to verify it** — the e2e config name, the eval
  numbers, `pre-commit run --all-files`. "Tested locally" is not an answer.
- **The issue it closes**, if any.

## 2. Title

```
<type>(<scope>): <description>
```

- **type**: `feat` `fix` `docs` `style` `refactor` `test` `chore` `perf`
  `build` `ci` `revert`. Lowercase. On `main` the distribution is roughly
  `feat` 166 / `fix` 122 / `docs` 76 / `chore` 18 / `refactor` 14.
- **scope**: the component the diff touches — `embodiment`, `libero`,
  `realworld`, `readme`, `docker`, `collective`, `openpi_pytorch`. Optional;
  drop it rather than invent one. `[a-z0-9_-]` only — a space, slash, dot or
  capital fails CI. `feat(new)` (PR #1421) passes CI but is the anti-pattern
  the linter warns on: it names the PR, not the code.
- **description**: **≤ 50 characters**, and that is the part *after* `: `, not
  the whole title. This is the rule PRs fail most often. It must start with a
  lowercase `a-z` (so `support 5D parallelism`, never `5D parallelism
  support`), stay imperative, carry no trailing period, and use plain ASCII —
  `π₀` in a title fails the check.
  `add MolmoAct2 LIBERO evaluation support`, not `Added support for MolmoAct2`.
- Never append `(#123)` — GitHub adds it at squash-merge.

Fitting 50 characters is the whole discipline of the title: name the one change
a reviewer needs to see in the PR list, and let the body carry the rest. If the
PR does several things, title it after the largest and enumerate in
`Description`.

Check just the title, no body needed:

```bash
python3 .claude/skills/create-pr/lint_pr.py lint --title "feat(new): add stuff."
```

## 3. Body

Start from the skeleton so no section goes missing:

```bash
python3 .claude/skills/create-pr/lint_pr.py template > /tmp/pr-body.md
```

Then fill it in. **Keep all six headings**, even where a section is one line.
What each one is for:

| Section | What goes in it |
| --- | --- |
| `Description` | One sentence on what changed and where, then a bullet per user-visible change. ≤ 250 words, ≤ 12 bullets. |
| `Motivation and Context` | Why the change is needed. Link the issue (`Fixes #NNN`). |
| `How has this been tested?` | The command you ran and its outcome. Required by CONTRIBUTING.md when the PR can move the reward curve — give the numbers. |
| `Additional information` | Figures, logs, repro commands. May be empty. |
| `Types of changes` | Tick what applies. One or two boxes, not four. |
| `Checklist:` | Tick what applies. **CONTRIBUTING.md: a PR with an empty Description or Checklist is marked draft and not reviewed.** |

Style rules the linter enforces:

- **No leading indentation.** Not on headings, not on paragraphs, not on table
  rows. Four spaces turns a paragraph into a code block; a table whose header
  and rows disagree on indent does not render as a table at all.
- **Delete the `<!--- ... -->` hints** once a section is written.
- **Nothing above the first `###` heading** — put it in `Description`.
- **One line per paragraph.** GitHub reflows; hard wraps only survive until
  someone edits the body.

## 4. Lint, then create

```bash
python3 .claude/skills/create-pr/lint_pr.py lint \
    --title "$TITLE" --body-file /tmp/pr-body.md
```

Add `--docs-only` for a documentation-only PR to drop the "how was this tested"
warning. Add `--quiet` to omit the suggested fixes. Once it exits 0:

```bash
gh pr create --base main --title "$TITLE" --body-file /tmp/pr-body.md
```

Commits must be signed off (`git commit -s`) and follow the same
`<type>(<scope>): <description>` format — DCO checks the trailer, so make sure
your name is not wrapped in quotes.

## 5. Fixing a PR that is already open

```bash
python3 .claude/skills/create-pr/lint_pr.py lint --pr 1421
```

For the indent/wrap damage specifically, `fix` rewrites the body: it strips the
paste offset (preserving genuine list nesting), pulls headings and tables flush
left, and drops the template hints. `--unwrap` additionally joins hard-wrapped
prose into one line per paragraph.

```bash
python3 .claude/skills/create-pr/lint_pr.py fix --pr 1421 --unwrap > /tmp/fixed.md
python3 .claude/skills/create-pr/lint_pr.py lint --title "$TITLE" --body-file /tmp/fixed.md
gh pr edit 1421 --body-file /tmp/fixed.md
gh pr edit 1421 --title "feat(embodiment): add MolmoAct2 LIBERO evaluation support"
```

`fix` only de-indents a body the linter already flagged, so running it on a
well-formed PR is a no-op apart from comment removal. **Read the output before
posting it** — it is a mechanical rewrite, not an editor.

## Gotchas

- **`lint --pr N` shells out to `gh pr view`**, so it inherits your `gh` auth
  and the repo you are standing in. It is read-only; nothing is posted until
  you run `gh pr edit` / `gh pr create` yourself.
- **The skeleton from `template` fails its own lint** — deliberately. It ships
  with no boxes ticked, and `checkbox-unticked` is an ERROR. Tick them.
- **The linter accepts `##` as well as `###`** for section headings, because
  GitHub renders both. What it will not accept is a *renamed* section: a body
  using `## Testing` instead of `### How has this been tested?` reports
  `section-missing`.
- **Warnings do not fail the run** (exit 0). `body-hardwrap`, `body-comments`
  and `title-scope-vague` are judgement calls — but a reviewer reads them the
  same way the linter does.
- **`fix` cannot recover a broken code fence.** If `body-fence` fires (odd
  number of ```` ``` ````), everything after the stray fence has been swallowed
  by GitHub's renderer; find it and close it by hand.
- **Tables inside `<details>` still need flush-left rows.** The `<details>`
  wrapper does not exempt them.
- **A good commit subject is not always a legal PR title.** `commit-check` and
  `pr-title-check` are different jobs with different limits — commit subjects
  get ~72 characters, PR descriptions get 50. A one-commit PR that reuses its
  subject verbatim can fail.

## Troubleshooting

- `error: gh pr view N failed: ... could not resolve to a PullRequest` — wrong
  repo. `cd` to the RLinf checkout, or pass the number of a PR in this repo.
- `[E] body/body-indented` persists after `fix` — the body mixes two paste
  offsets (e.g. part typed, part pasted). Dedent by hand; `fix` only removes a
  single uniform offset.
- `[W] body/section-order` — you kept all the headings but reordered them.
  Harmless, but reviewers scan for the template order.
- `lint --title` alone reports `0 error(s)` on an empty body — that is correct,
  it only checks what you gave it. Pass `--body-file` too before you create.
