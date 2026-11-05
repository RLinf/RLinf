---
name: review-pr
description: Reviews an RLinf pull request for correctness, design, code reuse, tests, and docs against RLinf's code and design patterns, reports the findings in a fixed format, and, after asking, posts them as one GitHub review with a main comment that answers the eight review questions and inline comments with suggested fixes. Use when the user asks to review a PR, check a PR before merge, or post review comments.
---

# Review a PR

A review produces two things: a report in the terminal for the user, in the
fixed format of step 6, and, only if the user says yes, a GitHub review on the
PR made of one main comment that answers the eight review questions and one
inline comment per code finding, each with a suggested fix.

Most of the review is about correctness, clean design, and reuse of what RLinf
already has. [patterns.md](patterns.md) is the standard the PR is measured
against: RLinf's code, design, test, and docs patterns. [reference.md](reference.md)
is the checklist for each finding category. Read both before step 4.

## 1. Set up

Take a PR URL or number. All cross-references are against the `main` branch of
`github.com/RLinf/RLinf`, never the local working tree, which may be on another
branch or hold uncommitted work. The remote that points there may be `origin`
in a contributor clone or `upstream` in a maintainer clone:

```bash
N=<PR number>
REMOTE=$(git remote -v | awk '/github\.com[:\/]RLinf\/RLinf(\.git)? \(fetch\)/ {print $1; exit}')
git fetch "$REMOTE" main "pull/$N/head:pr-$N"
BASE="$REMOTE/main"; PR_HEAD="pr-$N"

gh pr view "$N" --repo RLinf/RLinf --json title,body,author,headRefOid,commits,files
git diff --stat "$BASE...$PR_HEAD"            # the same diff GitHub shows
git show "$BASE:<path>"                       # a file on main
gh api "repos/RLinf/RLinf/pulls/$N/comments" --paginate   # earlier review threads
```

When you need to run code, check out `$PR_HEAD` in a separate git worktree.
Never edit the PR branch or push to it. Without `gh`, read `<PR URL>.diff` and
the PR page instead; the review can still run, but posting needs `gh`.

On a second round of review, go through every earlier thread first. Note which
ones the new commits fix and which are still open, and carry the open ones
forward rather than writing them again.

## 2. Understand the PR

Read the description, the linked issue, the commits, and the whole diff. Then
write down, for step 6:

- the problem the PR solves and its core idea, in one or two sentences;
- the areas it touches and what could regress (backends, envs, models, tasks);
- the closest current sibling on `main` it should look like (`git log "$BASE"
  -- <dir>` shows which sibling is recent).

## 3. Plan the reproduction

A PR that adds or changes a user-facing workflow (a new model, env, algorithm,
or feature with docs) or that claims metrics needs a reproduction. The review
questions 1 to 3 report it. Follow this rule:

> Prefer a human-led run. A reproduction agent (if used) uses a fresh session
> and follows the user documentation without inspecting the implementation or
> fixing failures. Record the first failure and any interventions; a run that
> needed undocumented changes does not establish that the submitted workflow is
> reproducible. Functional success and metric success are separate conclusions.

Ask the user once, with `AskUserQuestion`, which of these to use:

- **A human-led run.** The user gives you the record (settings, logs, results,
  where the docs were unclear), now or before posting.
- **A reproduction agent.** Start a fresh `general-purpose` agent in the
  background (never a fork, which inherits what you read). Its brief holds only
  the PR URL, the head SHA, the doc pages to follow, the hardware it may use,
  and the host's install rules. It works in its own worktree at the PR head,
  reads only the user docs, does not open `rlinf/` or `tests/`, does not fix
  anything, and reports the settings of question 1, the first failure, every
  intervention, and the functional and metric results separately.
- **No reproduction.** Questions 1 and 2 say so and what a run would need.

You never reproduce the PR yourself: you have read the implementation. The unit
tests you run while verifying findings are not a reproduction; they belong in
question 7.

## 4. Review

Work through [reference.md](reference.md) in this order, and judge every design
choice against [patterns.md](patterns.md). Each finding gets one category,
which also decides the question of the main comment that summarizes it:

| Category | Covers | Question |
|---|---|---|
| `correctness` | bugs, edge cases, distributed and lifecycle errors, refactor regressions | 4 |
| `design` | extension points, containment, special cases, abstractions, protection code, config, unnecessary changes | 5 |
| `reuse` | duplicated logic and the existing helper to use | 6 |
| `tests` | test coverage and layout, mocks, install, Docker, CI | 7 |
| `docs` | docs against code, EN/ZH parity, doc style and usability | 7 (3 for usability) |
| `style` | code conventions, headers, commits, PR title and description | 8 |

For every changed function, read the `main` version and the PR version side by
side. Label a finding that came from reading `main` rather than the diff alone
as a main-branch cross-check.

## 5. Verify every finding

Before a finding goes in the report, settle it as one of:

- **confirmed**: there is evidence. Best is a run: a focused unit test, a
  scratch script in the PR worktree (CPU when it allows), or the input that
  fails. Otherwise a complete trace that cites every line on the PR head.
- **unresolved**: plausible, but it needs hardware, assets, or the author's
  intent to settle. Say what would settle it.
- **dismissed**: the trace shows it cannot happen, or the code matches the
  convention on `main`. Keep one line on why.

Then give each confirmed or unresolved finding a severity:

- **critical**: wrong results, a crash, data loss, a broken existing path or
  backend, or a documented workflow that cannot be completed.
- **major**: a break with [patterns.md](patterns.md), or missing tests, docs, or
  CI for a user-facing change. It must change before merge.
- **minor**: a local fix such as naming, a comment, or a small style issue. It
  should change but does not block.

When correctness, design, and reuse yield nothing, the report says so and lists
what was checked. Do not invent findings to fill a category.

## 6. Report

Print the review in exactly this shape. Sort findings by severity, then by the
category order of step 4. The comment text of each finding is written exactly
as it will be posted (step 8). The code walkthrough and the evidence stay in the
terminal, so the user can check each finding before anything is posted.

````markdown
# Review of #<N>: <PR title>

<PR URL> · head `<sha7>` · base `main@<sha7>` · <YYYY-MM-DD>

## What the PR does

- Goal: <the problem it solves>
- Idea: <the core idea in one or two sentences>
- Changes: <areas touched, with file counts>
- Risk: <paths, backends, or envs that could regress>
- Compared with: <the closest sibling on main>

## Findings

| ID | Severity | Category | Location | Summary | Status | Posted in |
|---|---|---|---|---|---|---|
| F1 | critical | correctness | `rlinf/x.py:42` | <one line> | confirmed | inline |
| F2 | major | docs | `docs/source-zh/...` | <one line> | confirmed | question 7 |

### F1 · critical · correctness: <plain title>

`rlinf/x.py:42`, compared with `main:rlinf/x.py:40` (main-branch cross-check)

<The comment text from step 8, exactly as it will be posted.>

Code walkthrough:

<The key snippets from the PR, and from main when compared, each under a
`path:line` header, then the steps that lead from the input to the bug.>

Evidence: <the command and its result, or the trace; for an unresolved finding,
what would settle it>

## Dismissed

- <candidate>: <why it does not apply>

## Main comment

<The eight sections from step 8, exactly as they will be posted.>
````

## 7. Ask before posting

Posting speaks for the user, so ask first with `AskUserQuestion`, every time:
"Post this review to PR #N?" with the options **Post as a comment**
(recommended), **Post as request changes**, and **Don't post**. The user can
also answer with edits; apply them, show the changed text, and ask again.
Approve only when the user asks for it.

## 8. Post

Write the review as the JSON payload of GitHub's create-review API in the
scratchpad, with `json.dump` so the escaping is right, then lint it and post it:

```bash
python3 .agents/skills/review-pr/lint_review.py template      # main-comment skeleton
python3 .agents/skills/review-pr/lint_review.py lint --pr "$N" review.json
python3 .agents/skills/review-pr/lint_review.py post --pr "$N" review.json
```

```json
{
  "commit_id": "<PR head SHA>",
  "event": "COMMENT",
  "body": "<the main comment>",
  "comments": [
    {"path": "rlinf/x.py", "line": 42, "body": "<inline comment>"},
    {"path": "rlinf/y.py", "start_line": 10, "line": 12, "body": "<inline comment>"}
  ]
}
```

`event` is `COMMENT` or `REQUEST_CHANGES`, as the user chose. `lint` checks the
eight headings, the recommendation, the `Suggested fix:` paragraph of every
inline comment, hard-wrapped lines, and, with `--pr`, that every comment sits on
a line of the PR diff and not on a line that already has a thread. GitHub
rejects the whole review when one comment is outside the diff. `post` runs the
same lint, posts everything as one review, and prints its URL. Report that URL
and the number of inline comments to the user.

### Where a finding goes

A finding goes inline when a line of the diff carries it: the line with the bug,
the line where a special case enters shared code, or, for a block, `start_line`
to `line`. Use `"side": "LEFT"` only for a deleted line that should stay. A
finding with no such line goes in the main comment under its question: a
missing doc page, a missing test, PR metadata. Reply in an existing thread
instead of opening a new one on the same line:
`gh api "repos/RLinf/RLinf/pulls/$N/comments/<id>/replies" -f body=...`.

### Inline comments

Write for the PR author, who knows their own code but may not know RLinf's
internals. Each comment covers one finding:

1. The first paragraph says what goes wrong and when, then why it matters.
   Start from the situation the author recognizes, then name the identifiers.
   One to three sentences.
2. When the cause or the pattern to follow lives elsewhere, point to it as
   `main:<path>:<line>` or quote the line.
3. The last paragraph starts with `Suggested fix:` and says in one sentence what
   to change. A `suggestion` block follows when the fix replaces exactly the
   commented lines, so the author can apply it with one click; otherwise a
   fenced code block with a language tag shows the change. The code is complete
   and correct; run it when you can.

Keep the language plain: short sentences, active voice, no hedging, no filler,
no praise, and no severity labels except `Minor:` at the start of a minor
finding. Keep each paragraph on one line, because GitHub shows every newline as
a line break. Write in English unless the user asks otherwise.

A correctness finding:

````markdown
`dones` is cast to `int` before the mask is built, so `~dones` gives `-1` and `-2` instead of `False` and `True`. The loss then weights every step with a negative factor and the policy is pushed the wrong way.

Suggested fix: keep `dones` boolean when you build the mask.

```suggestion
        mask = ~dones.bool()
```
````

A design finding:

````markdown
The shared rollout worker now checks `model_type == "foo"` to add the wrist image, so the worker has to know about one model, and the next model with its own inputs would add another branch here. Model-specific inputs belong to the policy: the worker already passes the raw observation to `predict_action_batch` for every model (`main:rlinf/workers/rollout/hf/huggingface_worker.py:565`).

Suggested fix: build the wrist image inside `FooPolicy.predict_action_batch` and drop the check from the worker.

```python
class FooPolicy(BasePolicy):
    def predict_action_batch(self, env_obs, **kwargs):
        env_obs = {**env_obs, "wrist_image": self._wrist_view(env_obs)}
        ...
```
````

### Main comment

The main comment holds only the eight sections, each under its question as a
`###` heading, word for word (`lint_review.py template` prints them). Answer in
short bullets, keep each one to one line, point to inline comments by
`path:line`, and put long logs in a `<details>` block.

1. **What settings did you use to reproduce this PR?** The PR revision (head
   SHA), hardware and nodes, software environment, model and dataset, config,
   and commands, and who ran it (a human or the reproduction agent). Link a
   saved config or environment summary rather than listing everything. Without
   a run, say so and what a run needs.
2. **What were the reproduction results?** The behavior or metrics the PR
   claims, next to what was observed, with links to logs and results. For a
   metric, give the workload, sample size or seeds, and the baseline. State the
   functional result and the metric result separately, the first failure and
   every intervention, and what was not reproduced.
3. **Is the documentation clear and usable?** Whether every step from setup to
   the expected result could be followed; unclear lines, missing prerequisites,
   failed commands, and any extra help or workaround needed. Without a run, say
   that the docs were read but not followed.
4. **Are there bugs or unresolved concerns?** The confirmed issues and the
   unresolved concerns, each with its trigger and evidence and a pointer to its
   inline comment, then the dismissed findings with one line each on why. "No
   issue found" says what was checked.
5. **Does the integration follow RLinf's design?** The existing implementation
   compared against; whether interfaces, registration, config, worker
   conventions, and containment match it; mismatches; and anything that needs a
   senior maintainer's judgment.
6. **Does it reuse existing modules/functions properly?** Avoidable
   duplication and the existing helper to reuse, or else the implementation
   compared against.
7. **Are tests and documentation adequate?** Whether the tests cover the changed
   behavior; whether docs, EN/ZH updates, and install and CI integration are
   there where needed; the tests that ran (command, hardware, result) and the
   ones skipped, with why.
8. **What is your recommendation?** Exactly one of `Ready for senior review`
   (no open critical or major finding, and the reproduction succeeded or was
   not needed), `Changes requested` (open critical or major findings the
   author can fix), or `Blocked` (progress needs more than the author's fixes:
   a maintainer decision, hardware or assets nobody has, another PR, or a
   reproduction failure the docs cannot explain). Then the remaining actions as
   a list, and the expert review needed.
