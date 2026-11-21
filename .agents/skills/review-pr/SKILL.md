---
name: review-pr
description: Reviews an RLinf pull request for correctness, design, and code reuse against RLinf's design patterns, runs the check of every area the PR touches (code, docs, install) through those skills and the reproduction when the host can, reports the findings in a fixed format, each with its code walkthrough and evidence, and, when the user asks, posts them as one GitHub review: a main comment that leads with the reproduction and the findings, and inline comments with suggested fixes. Use when the user asks to review a PR, check a PR before merge, or post review comments.
---

# Review a PR

A review is one concise document in the fixed format of step 8: a verdict, a two-bullet reproduction summary, the findings table, detail sections only for non-inline findings, and one inline comment per diff-line finding (never repeated in the main comment). The terminal shows the main comment and inline comments once, followed by a two-bullet check summary that is not posted. Only when the user asks for it does it become a GitHub review on the PR: the main comment plus the inline comments on their diff lines.

Most of the review is about correctness, clean design, reuse of what RLinf already has, and changes that deviate from the existing codebase without need. Everything else is checked by the skill that owns the area: code and tests by [code](../code/SKILL.md), docs by [docs](../docs/SKILL.md), and install, Docker, and e2e by [install](../install/SKILL.md). Step 3 finds every area the PR touches and step 6 runs each one's check, so no area goes unchecked. PR formatting and commit conventions are checked but should not dominate. [code/patterns.md](../code/patterns.md) is the design standard the PR is measured against, and [reference.md](reference.md) is the checklist for correctness, design, reuse, and style. Read both before step 5.

## 1. Set up

Take a PR URL or number. All cross-references are against the `main` branch of `github.com/RLinf/RLinf`, never the local working tree, which may be on another branch or hold uncommitted work, so a search over it does not show the state of the project. The remote that points to RLinf may be `origin` in a contributor clone or `upstream` in a maintainer clone:

```bash
N=<PR number>
REMOTE=$(git remote -v | awk '/github\.com[:\/]RLinf\/RLinf(\.git)? \(fetch\)/ {print $1; exit}')
git fetch "$REMOTE" main "pull/$N/head:pr-$N"
BASE="$REMOTE/main"; PR_HEAD="pr-$N"
SKILLS="$(git rev-parse --show-toplevel)/.agents/skills"   # the skills this review follows

gh pr view "$N" --repo RLinf/RLinf --json title,body,author,headRefOid,commits,files
gh pr checks "$N" --repo RLinf/RLinf                     # CI status
git log --oneline "$BASE..$PR_HEAD"                      # what the PR adds
git diff --stat "$BASE...$PR_HEAD"                       # the same diff GitHub shows
git show "$BASE:<path>"                                  # a file on main
git grep <pattern> "$BASE"                               # search main
gh api "repos/RLinf/RLinf/pulls/$N/comments" --paginate  # earlier review threads
```

Check out `$PR_HEAD` in a separate git worktree; steps 3 and 6 run there, and so does any code you run to verify a finding. Never edit the PR branch or push to it. Without `gh`, read the PR page, `<PR URL>.diff` (or `<PR URL>.patch`), and files on `https://github.com/RLinf/RLinf/blob/main/<path>`; the review can still run, but posting needs `gh`. If the PR is private and neither works, tell the user and ask for an exported diff and description.

On a second round of review, go through every earlier thread and every finding of the earlier main comment first. Verify each fix in the current code rather than trusting commit messages, and check whether the fix introduced a regression or shifted the bug elsewhere. Note which ones the new commits fix and which are still open, and carry the open ones forward rather than writing them again: an open inline thread gets a reply, and an open main-comment finding keeps its earlier ID in the new review. Do not reopen a minor preference the author already answered reasonably. The verdict sentence names the earlier findings the new commits fix.

## 2. Understand the PR

Read the description, the linked issue, the commits, and the whole diff. Then write down, for step 8:

- the problem the PR solves and its core idea, in one or two sentences;
- the areas it touches (for example packaging, docs, CI, refactor) and what could regress (backends, envs, models, tasks);
- the closest current siblings on `main` it should look like, for both code/config and docs (`git log "$BASE" -- <dir>` shows recent siblings; for docs, match the model or domain family and its shared setup pages, not only the directory or gallery the PR chose);
- every claim in the PR description (such as "behavior-preserving refactor", test coverage, hardware runs, or reported metrics) treated as a hypothesis to verify against the diff, tests, and baseline rather than a fact.

## 3. Route the change

In the PR worktree, list the areas the PR touches:

```bash
python3 "$SKILLS/route.py" list --base "$BASE"
```

It prints each touched area, the `SKILL.md` to read for it, and the area's files. Every listed area is checked in step 6, and any failed or skipped check is noted in the check summary. Files no area owns, such as agent instructions, are listed as unrouted; review them yourself in step 5. The script takes the skills from `$SKILLS`, so a PR that predates them is checked by the current ones.

## 4. Run the reproduction

A PR that adds or changes a user-facing workflow (a new model, env, algorithm, or feature with docs) or that claims metrics needs a reproduction, and the Reproduction section of the review reports it. Run it without asking whenever this host can:

1. **See what the workflow needs.** Read the PR's docs for the hardware (accelerator type and count, nodes), the installation, the model weights, the datasets, and other assets, and compare them with the host: `nvidia-smi -L` or the accelerator's own tool, the install environments, and whether the weights and datasets exist (for an embodied e2e config, `python3 "$SKILLS/install/driver.py" check-paths <config>`). Follow the host's install and download rules, such as mirrors and shared venv locations.
2. **Start the run when the host has what it needs**, in the background and right away, so it runs while you review. Check first that the accelerators are free (`nvidia-smi`, and `ray status` for other users' clusters), and never stop someone else's job. The run is a fresh agent, never a fork, which would inherit what you have read. Its brief holds only the PR URL, the head SHA, the doc pages to follow, the hardware it may use, and the host's install rules. It works in its own worktree at the PR head, follows only the user docs, does not open `rlinf/` or `tests/`, fixes nothing, and reports the settings it used, the first failure, every intervention, and the functional and metric results separately. A metric claim that needs a long run gets the shortest run the docs define, such as the e2e config, and the report says what that run can and cannot show.
3. **Otherwise say what is missing.** When the host lacks the hardware, weights, datasets, or accounts, the Reproduction section names what is missing and what a run would need. When the user hands you the record of their own run, report that instead.

Every reproduction follows the user documentation, records the first failure and every intervention, and keeps functional success and metric success as separate conclusions; a run that needed undocumented changes does not show that the workflow is reproducible. You never reproduce the PR yourself, because you have read the implementation. A step of the docs that the run could not follow is a `docs` finding. A passing reproduction only proves the commands execute, not that they are clean, standard, or easy for a human to follow; judge instruction simplicity, YAML-first configuration, and shared setup reuse in step 6. The unit tests and area checks you run are not a reproduction; they belong in the check summary.

## 5. Review the code

Work through [reference.md](reference.md) in this order, and judge every design choice against [code/patterns.md](../code/patterns.md): its patterns first, and its design philosophy when no pattern fits. Each finding gets one category, and the review lists findings in this order within one severity:

| Category | Covers |
|---|---|
| `correctness` | bugs, edge cases, distributed and lifecycle errors, refactor regressions |
| `design` | extension points, containment, special cases, abstractions, protection code, config, unnecessary changes |
| `reuse` | duplicated logic and the existing helper to use |
| `tests` | test coverage and layout, mocks, install, Docker, CI, from the code and install checks |
| `docs` | docs against code, EN/ZH parity, README highlights, asset references, usability, from the docs check and the reproduction |
| `docs-style` | changed pages and README text against the docs rules for their type and language, from the docs check |
| `style` | code conventions, headers, commits, PR title and description |

For a large PR (more than ~15 files or ~1,000 changed lines), partition the diff by concern (core logic, tests and CI, install and config, docs) and review each group explicitly—using parallel focused agents when helpful—so files late in the diff (especially tests, configs, and Chinese docs) are not skimmed.

For every changed function, read the `main` version and the PR version side by side. Whenever the PR changes a function signature, return shape, base-class contract, shared helper, or config key, grep all callers, subclasses, YAMLs, tests, and docs at `$PR_HEAD`—not only the diff—to catch broken untouched consumers. Label a finding that came from reading `main` rather than the diff alone as a main-branch cross-check.

## 6. Check each area

For each area step 3 listed, read `$SKILLS/<area>/SKILL.md`, section Check, and do all of it in the PR worktree: its script on the routed files, then its judgment list.

```bash
python3 "$SKILLS/route.py" check --base "$BASE"            # every touched area's script
python3 "$SKILLS/create-pr/lint_pr.py" lint --pr "$N"      # PR title and description
```

`route.py check` runs each area's `check.py` with that area's files; the judgment list is yours to do. For docs, `check.py` only inspects build, markup, and mechanical prose checks—not readability, page density, or command quality. Delegate the docs readability and conciseness review to a fresh agent (never a fork) that sees only the changed EN and ZH pages, their closest sibling pages on `main` (and any shared setup page they link to), and the docs rules—without reading `rlinf/` source code—so it reads like a newcomer instead of mentally filling in gaps from the implementation. Read every changed page in full in both languages with **readability and conciseness as primary review criteria**, and actively flag:

- **Unreadable or unnatural prose**: stiff jargon, abstract nominalizations, convoluted sentences that do not read like plain human technical writing, or Chinese pages directly machine-translated / literally translated sentence by sentence from English.
- **Long-winded explanations of details**: walls of prose, over-explained internals, exhaustive options, defaults, or defensive caveats that bury the main path.
- **Code citations in user docs**: internal class, function, module, or file names, source-tree tours, or code snippets where the reader only needs user-visible behavior or the config keys they edit.
- **Complex commands**: shell plumbing, long CLI override lists, inline prep scripts, or ad-hoc env-var workarounds instead of YAML defaults and standard RLinf launchers.
- **Repeated negative-and-positive restatement**: explaining the same point from both sides (for example "this is not A, not B, but C" / "这不是 A，不是 B，而是 C") or restating what something does by listing what it is not; state what it is directly, once.

Also compare recipes side by side against their closest model- or domain-family siblings on `main` (and any shared setup page they link to). When a missing detail would clutter a recipe or guide, ask the author to fix the YAML default, move multi-step logic into a script, or link to Reference rather than adding more prose or workarounds to the page. A script's finding becomes a review finding once step 7 settles it, filed under the category of its rule:

- `tests`: `code/ci-filter`, `code/test-layout`, and every `install/*` rule;
- `docs`: `docs/symbol`, `docs/rtd-link`, `docs/parity`, docs-to-code checks, sibling workflow alignment, setup reuse, and command simplicity/usability from the docs judgment list and reproduction;
- `docs-style`: `docs/build`, `docs/markup`, `docs/code-name`, `docs/math`, `docs/structure`, `docs/heading`, `docs/en-style`, `docs/zh-style`, `docs/zh-hard-wrap`, readability and conciseness (long-winded details, code citations, repeated negative-and-positive framing), and the prose/structure checks of page reading;
- `style`: `code/pre-commit`, `code/init`, `code/license`, and the PR lint.

When an area's script cannot run, for example without a docs virtualenv or without GPUs for the install Verify step, the report says it did not run and what it needs. Never leave a routed area out.

## 7. Verify every finding

Try to falsify every candidate before reporting it. When a claim can be checked in seconds on CPU in the PR worktree—config resolution (`hydra.compose`, `validate_cfg`), shape/dtype/mask logic, imports, regexes, or focused unit tests—run it instead of relying on a static trace. For third-party libraries (`torch`, `transformers`, `omegaconf`, `ray`, `sglang`, `vllm`), inspect the installed version or source rather than guessing from memory. Settle each candidate as one of:

- **confirmed**: there is evidence. Best is a run: a focused unit test, a scratch script in the PR worktree (CPU when it allows), or the input that fails. Otherwise a complete trace that cites every line on the PR head.
- **unresolved**: plausible, but it needs hardware, assets, or the author's intent to settle. Say what would settle it.
- **dismissed**: the trace or run shows it cannot happen, or the code matches the convention on `main`. Drop it silently—do not list dismissed candidates in the output.

Merge findings that share one root cause into a single finding that lists the affected locations instead of splitting one defect across code, YAML, test, and docs rows. Then give each confirmed or unresolved finding a severity on the scale in [README.md § Findings](../README.md#findings): critical, major, or minor. A script's severity is where you start; change it when the evidence says otherwise. Hold `critical` strictly to concrete failure paths (wrong results, crash, deadlock, data loss, broken existing path, failed docs build, or uncompletable workflow); downgrade or drop speculative edge cases with no realistic trigger. The docs skill's Check says when a docs-style finding is major.

Report only real problems. Collapse or drop low-value minor nitpicks so major issues stay in focus. When correctness, design, and reuse yield nothing, the report says so and lists what was checked. Do not invent findings to fill a category.

## 8. Report

Keep the output tight and free of repetition. Print the main comment as it will be posted, then each inline comment once, then a two-bullet check summary (not posted).

- **Reproduction**: lead with the bold outcome (`**Passed**`, `**Failed**`, `**Partial**`, or `**Not run**`) and the key functional/metric conclusion in one short sentence. Under `Setup:`, keep only essential hardware count/type and the config or command run—omit hostnames, OS, driver/CUDA/Python/venv paths, and the commit SHA already in the verdict line. Keep each bullet concise (under ~300 characters); use a small 2–3 row table only when comparing multiple metrics.
- **No duplicate findings**: the findings table indexes every finding (`F1`, `F2`, ...), sorted by severity then category. Below the table, write a `#### F<n>` section **only for `Posted = here` findings** (findings with no diff line). Do **not** write a `#### F<n>` section in the main comment for an `inline` finding—an inline finding lives once in its inline comment (carrying its problem, `Suggested fix:`, and one-line `Evidence:`).
- **No trailing noise**: omit `Dismissed:` lists, restated Goal/Idea/Changes/Risk bullets, and per-page tables of docs that passed.

````markdown
**<Recommendation>** · <n critical, n major, n minor, or no findings> · head `<sha7>` · base `main@<sha7>`

<One short sentence: what blocks merge, or why it is ready.>

### Reproduction

- Result: **<Passed | Failed | Partial | Not run>** — <one short sentence: functional status and key metric vs. baseline, or first blocker / what is missing>
- Setup: <hardware count/type and config or command run; or what a run needs if not run>

### Findings

| ID | Severity | Category | Location | Summary | Status | Posted |
|---|---|---|---|---|---|---|
| F1 | critical | correctness | `rlinf/x.py:42` | <one line> | confirmed | inline |
| F2 | major | docs | `docs/source-zh/...` | <one line> | confirmed | here |

#### F2 · major · docs: <plain title>

`docs/source-zh/...`, compared with `main:...` (main-branch cross-check)

<What the code or page tries to do, what happens instead, and why it matters, in 1–3 plain sentences.>

Suggested fix: <one sentence>, then a code block when code changes.

Code walkthrough:

<Quoted snippets from the PR and from main when compared, each under a `path:line` line, then the steps from input to failure. Omit when not about code.>

Evidence: <one line: command and result, or trace; for unresolved, what settles it>
````

Then list the inline comments once (posted as `comments[]` in step 10, never duplicated in the main comment):

````markdown
### Inline comments (posted on diff lines)

**F1** · `rlinf/x.py:42`

<inline comment body from step 10: problem, `Suggested fix:` + code/suggestion, and one-line `Evidence:`>
````

Then the check summary, which is never posted (keep to two compact bullets):

````markdown
## Check summary (not posted)

- Checks: `route.py check` (<pass or findings>) · `lint_pr` (<result>) · tests (<command, hardware, result; or skipped + why>)
- Context: compared with <closest sibling on main>; <any area check that could not run and why>
````

## 9. Post only when asked

The review ends with the report of step 8; do not post it by default. When the user asks you to post, follow step 10 with the event they name: `COMMENT` unless they ask for request changes. When they ask for edits first, apply them and show the changed text. Approve only when the user asks for it.

A posted review is changed in place rather than posted again; only a second round on a new head posts a new review (step 1). To change its main comment, run `lint_review.py update` (step 10); to change an inline comment, edit that comment or reply in its thread. When `post` fails without printing a URL, as on a dropped connection, list the PR's reviews (`gh api "repos/RLinf/RLinf/pulls/$N/reviews"`) before you try again, because GitHub may have created the review.

## 10. Post

Write the review as the JSON payload of GitHub's create-review API in the scratchpad, with `json.dump` so the escaping is right, then lint it and post it. The body is the main comment of step 8 (without `### Inline comments` or `## Check summary`):

```bash
python3 .agents/skills/review-pr/lint_review.py template      # main-comment skeleton
python3 .agents/skills/review-pr/lint_review.py lint --pr "$N" review.json
python3 .agents/skills/review-pr/lint_review.py post --pr "$N" review.json
python3 .agents/skills/review-pr/lint_review.py update --pr "$N" --review <id> review.json   # replace a posted body
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

`event` is `COMMENT`, or `REQUEST_CHANGES` when the user asks for it. `lint` checks the verdict line, the two sections in order, that every `here` row of the findings table has one section (and no `inline` row is duplicated as a section in the main comment), that every inline row has an inline comment at its location and every inline comment a row, that every finding carries `Suggested fix:` and `Evidence:` in one place, unclosed code fences, HTML and leftover placeholders, the concise `Result:` / `Setup:` reproduction bullets, hard-wrapped lines, the body length, warnings for overly long reproduction bullets, unquoted `Code walkthrough:` blocks, hedging phrases, or excessive minor findings, and, with `--pr`, that every comment sits on a line of the PR diff and not on a line that already has a thread. `update` runs the same checks against the inline comments the review already holds and replaces only its body. GitHub rejects the whole review when one comment is outside the diff. `post` runs the same lint, posts everything as one review, and prints its URL. Report that URL and the number of inline comments to the user.

### Where a finding goes

A finding goes inline when a line of the diff carries it: the line with the bug, the line where a special case enters shared code, or, for a block, `start_line` to `line`. Use `"side": "LEFT"` only for a deleted line that should stay. A docs-style finding goes on the line it rewrites; when one rule fails on many lines of a page, comment on the first and list the others in that finding's section rather than opening a thread per line. A finding with no such line is posted in the main comment, in its own section with the full comment text and its `Suggested fix:`: a missing doc page, a missing test, a file the PR should have changed, PR metadata. Reply in an existing thread instead of opening a new one on the same line: `gh api "repos/RLinf/RLinf/pulls/$N/comments/<id>/replies" -f body=...`.

### Inline comments

Write for a reader with little background in the module: the PR author knows their own code but may not know RLinf's internals, and the human reviewer may know RLinf but not this module. Each comment covers one finding:

1. The first paragraph says what the code is trying to do, what happens instead, and why it matters. Start from the situation the reader recognizes, then name the identifiers, so the idea is not buried in variable names. One to three sentences.
2. When the cause or the pattern to follow lives elsewhere, point to it as `main:<path>:<line>` or quote the line.
3. Next comes `Suggested fix:` saying in one sentence what to change, followed by a `suggestion` block when the fix replaces exactly the commented lines (so the author can apply it with one click), or a fenced code block with a language tag otherwise. Verify indentation, target lines, and syntax, and run the code when you can.
4. End with one short `Evidence:` line (the command and result, or the exact trace; for an unresolved finding, what settles it). Because inline findings are not repeated in the main comment, the inline comment is self-contained.

Keep the language plain: short sentences, active voice, no hedging, no filler, no praise, and no severity labels except `Minor:` at the start of a minor finding. Keep each paragraph on one line, because GitHub shows every newline as a line break. Write in English unless the user asks otherwise.

A correctness finding:

````markdown
`dones` is cast to `int` before the mask is built, so `~dones` gives `-1` and `-2` instead of `False` and `True`. The loss then weights every step with a negative factor and the policy is pushed the wrong way.

Suggested fix: keep `dones` boolean when you build the mask.

```suggestion
        mask = ~dones.bool()
```

Evidence: `~torch.tensor([0, 1], dtype=torch.int)` returns `tensor([-1, -2])`.
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

Evidence: `huggingface_worker.py:565` passes raw `env_obs` to every `BasePolicy.predict_action_batch` (patterns.md § Contain component code).
````

A docs-style finding:

````markdown
This paragraph explains the rollout loop by listing the worker, model, and channel files it passes through. A reader new to RLinf cannot connect those paths to anything they see when they run the example, and none of them is a file the reader opens or edits. The docs rules ask pages to explain a mechanism through what happens in the run (`.agents/skills/docs/rules/base.md`, "Write about what the reader does, not about the code").

Suggested fix: describe the loop as the reader sees it, and keep only the config key they set.

```suggestion
During rollout, the policy reads each observation from the simulator, predicts the next chunk of actions, and sends it back; the simulator steps every environment and returns the next observation. `actor.model.num_action_chunks` sets how many actions the policy predicts at a time.
```

Evidence: `docs/code-name` flagged 4 internal class/file names on this recipe page.
````

### Main comment

The main comment is the first block of step 8 (without `### Inline comments` or `## Check summary`); `lint_review.py template` prints its skeleton. It puts the problems first and holds nothing else:

- **Verdict.** The first line names exactly one recommendation in bold, then the counts per severity and the revisions reviewed. `Ready for senior review`: no open critical or major finding, and the reproduction succeeded or was not needed. `Changes requested`: open critical or major findings the author can fix. `Blocked`: progress needs more than the author's fixes, such as a maintainer decision, hardware or assets nobody has, another PR, or a reproduction failure the docs cannot explain. Follow with one short sentence stating what blocks merge or what decision a maintainer must make.
- **Reproduction.** Two short bullets, `Result:` (starting with `**Passed**`, `**Failed**`, `**Partial**`, or `**Not run**` and one short conclusion sentence) and `Setup:` (only essential hardware count/type and the config or command run, or what is missing if not run). Omit machine, OS, driver, venv, and SHA boilerplate.
- **Findings.** The table indexes every finding (`Posted` is `inline` or `here`). Only `here` findings get a `####` section below the table (headed `F<n> · <severity> · <category>: <title>`, with the problem, `Suggested fix:`, optional `Code walkthrough:` when about code, and `Evidence:`). `inline` findings live only in `comments[]` so nothing is repeated. When nothing survived verification, the section says `No findings.` and, in one line, what was checked.

Write the main comment and every inline comment in plain Markdown: no HTML tags such as `<details>` or `<br>`, and no HTML comments; `lint` rejects them, and any `<placeholder>` left from the template. Keep each paragraph on one line, and keep the body under GitHub's limit of 65,536 characters; `lint` reports a longer one.
