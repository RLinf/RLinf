---
name: review-pr
description: Reviews an RLinf pull request for correctness, design, and code reuse against RLinf's design patterns, runs the check of every area the PR touches (code, docs, install) through those skills and the reproduction when the host can, reports concise, deduplicated findings with evidence, and, when the user asks, posts them as one GitHub review. Use when the user asks to review a PR, check a PR before merge, or post review comments.
---

# Review a PR

Produce one concise, deduplicated review in the step 8 format: a verdict, a two-bullet reproduction summary, the findings table, detail sections only for non-inline (`here`) findings, inline comments printed once, and a two-bullet check summary (not posted). Post to GitHub only when the user asks ([posting.md](posting.md)).

Focus on correctness, design, reuse, and unnecessary divergence from `main`. Area checks live in [code](../code/SKILL.md), [docs](../docs/SKILL.md), and [install](../install/SKILL.md). Read [code/patterns.md](../code/patterns.md) and [reference.md](reference.md) before step 5.

## 1. Set up

Cross-reference against `main` of `github.com/RLinf/RLinf`, never the local working tree:

```bash
N=<PR number>
REMOTE=$(git remote -v | awk '/github\.com[:\/]RLinf\/RLinf(\.git)? \(fetch\)/ {print $1; exit}')
git fetch "$REMOTE" main "pull/$N/head:pr-$N"
BASE="$REMOTE/main"; PR_HEAD="pr-$N"
SKILLS="$(git rev-parse --show-toplevel)/.agents/skills"

gh pr view "$N" --repo RLinf/RLinf --json title,body,author,headRefOid,commits,files
gh pr checks "$N" --repo RLinf/RLinf
git log --oneline "$BASE..$PR_HEAD"
git diff --stat "$BASE...$PR_HEAD"
git show "$BASE:<path>"
git grep <pattern> "$BASE"
gh api "repos/RLinf/RLinf/pulls/$N/comments" --paginate
```

Check out `$PR_HEAD` in a separate git worktree for steps 3, 6, and verification; never edit or push the PR branch. Without `gh`, read `<PR URL>.diff` and files on `github.com/RLinf/RLinf/blob/main/<path>` (or ask the user if private).

On a second round, inspect every earlier thread and main-comment finding first. Verify each fix in the current code (never trust commit messages alone) and check for regressions or shifted bugs. Carry open findings forward (reply on open inline threads; keep earlier IDs on open main-comment findings), name fixed earlier findings in the verdict, and do not reopen minor preferences the author already answered reasonably.

## 2. Understand the PR

Read the PR description, linked issue, commits, and diff, and note:

- the problem and core idea;
- touched areas and what could regress (backends, envs, models, tasks);
- closest current siblings on `main` for code/config (`git log "$BASE" -- <dir>`) and docs (match model/domain family and shared setup pages, not just the gallery);
- every claim in the PR description ("behavior-preserving", test coverage, metrics, hardware runs) as a hypothesis to verify against the diff and tests.

## 3. Route the change

In the PR worktree:

```bash
python3 "$SKILLS/route.py" list --base "$BASE"
```

Check every listed area in step 6; review unrouted files yourself in step 5.

## 4. Run the reproduction

When a PR adds/changes a user-facing workflow or claims metrics, run a reproduction whenever this host has the hardware and assets:

1. **Check prerequisites** from the PR's docs against the host (`nvidia-smi -L` or accelerator tool, install envs, weights/datasets, or `python3 "$SKILLS/install/driver.py" check-paths <config>`). Follow host mirrors and shared-venv rules; verify accelerators are free (`nvidia-smi`, `ray status`) and never stop others' jobs.
2. **Launch a fresh agent (never a fork)** in the background while you review, working in its own worktree at `$PR_HEAD`. Give it only the PR URL, head SHA, doc pages to follow, allowed hardware, and host install rules. It must follow only user docs, never open `rlinf/` or `tests/`, fix nothing, and report the minimal setup, first failure, every intervention, and functional vs. metric results separately (for long metric claims, run the shortest documented config such as e2e and state what it does and does not prove).
3. **If blocked**, state what hardware, assets, or accounts are missing (or report the user's run if provided).

Never reproduce the PR yourself after reading the implementation. An undocumented step or failure is a `docs` finding; a passing run still requires step 6 checks on command simplicity and workflow standardness. Unit tests and area scripts are not reproductions—put them in Check summary.

## 5. Review the code

Work through [reference.md](reference.md) against [code/patterns.md](../code/patterns.md). File each finding under one category, ordered within each severity as:

| Category | Covers |
|---|---|
| `correctness` | bugs, edge cases, distributed/lifecycle errors, refactor regressions |
| `design` | extension points, containment, abstractions, test-driven guards, config, scope creep |
| `reuse` | duplicated logic and the existing helper to reuse |
| `tests` | test coverage/layout, mocks, install, Docker, CI |
| `docs` | docs-to-code accuracy, EN/ZH parity, README highlights, assets, workflow/command usability |
| `docs-style` | readability, conciseness, build/markup, prose/structure rules for page type and language |
| `style` | code conventions, headers, commits, PR title/description |

- **Large PRs** (> ~15 files or ~1,000 changed lines): partition by concern (core logic, tests/CI, install/config, docs) and review each group explicitly—using parallel focused agents when helpful—so late files (tests, YAMLs, Chinese docs) are not skimmed.
- **Side-by-side & caller sweep**: compare every changed function against `main` (label `main`-derived findings as main-branch cross-checks). Whenever a signature, return shape, base-class contract, shared helper, or config key changes, `git grep` all callers, subclasses, YAMLs, tests, and docs at `$PR_HEAD` to catch broken untouched consumers.

## 6. Check each area

In the PR worktree, run the automated checks and follow each routed area's `SKILL.md § Check` judgment list:

```bash
python3 "$SKILLS/route.py" check --base "$BASE"
python3 "$SKILLS/create-pr/lint_pr.py" lint --pr "$N"
```

- **Docs readability pass (fresh agent)**: `check.py` enforces mechanical structure, commands, paths, YAML keys, jargon, and markup, but not workflow ergonomics or prose clarity. Delegate the full EN/ZH reading to a **fresh agent (never a fork)** that sees only the changed doc pages, closest `main` sibling pages (and linked shared setup pages), and [docs/SKILL.md](../docs/SKILL.md) + `docs/rules/`—with no `rlinf/` source context. Prioritize user-friendliness and conciseness: flag (1) unreadable/stiff prose, testing jargon, or machine-translated/literal Chinese, (2) long-winded detail explanations or redundant sections, (3) internal code citations, (4) complex/non-standard commands, multi-step shell glue, or manual YAML edits where a 1-line CLI override works, (5) missing prerequisites (hardware/GPU count, expected output/metrics, next steps), and (6) repeated negative-and-positive framing ("not A, not B, but C" / "不是……而是……"). Prefer fixing YAML defaults, scripts, or Reference links over adding prose workarounds.
- **Script-to-category mapping**:
  - `tests`: `code/ci-filter`, `code/test-layout`, `install/*`
  - `docs`: `docs/symbol`, `docs/path-exists`, `docs/algo-keys`, `docs/eval-command`, `docs/multinode-trap`, `docs/rtd-link`, `docs/parity`, `docs/tab-parity`, `docs/no-todo`, `docs/local-asset`, docs-to-code, sibling alignment, setup reuse, command usability
  - `docs-style`: `docs/build`, `docs/markup`, `docs/code-name`, `docs/math`, `docs/latex-sanity`, `docs/structure`, `docs/heading`, `docs/en-style`, `docs/zh-style`, `docs/zh-hard-wrap`, `docs/jargon`, `docs/long-paragraph`, `docs/raw-html-grid`, `docs/index-style`, `docs/command-*`, `docs/nested-tabset`, `docs/tab-title`, `docs/mirror-comment`, `docs/dir-tree`, `docs/boilerplate-yaml`, `docs/long-yaml`, `docs/yaml-edit-friction`, `docs/extending-*`, `docs/guide-quality`, `docs/concept-purity`, readability/conciseness
  - `style`: `code/pre-commit`, `code/init`, `code/license`, PR lint
- Report any area script that could not run (and what it needs) in Check summary.

## 7. Verify every finding

Try to falsify each candidate before reporting it. Run cheap CPU checks in the PR worktree whenever possible (`hydra.compose`, `validate_cfg`, shape/dtype/mask checks, imports, focused tests) instead of relying on static traces. For third-party libraries (`torch`, `transformers`, `omegaconf`, `ray`, `sglang`, `vllm`), inspect the installed version or source instead of guessing from memory.

- **confirmed**: proven by a run or a complete line-cited trace on `$PR_HEAD`.
- **unresolved**: plausible, but needs hardware, assets, or author intent; state what settles it.
- **dismissed**: cannot happen or matches `main`; drop silently (never list in the output).

Merge findings with the same root cause into one finding listing affected locations. Assign severity strictly per [README.md § Findings](../README.md#findings): `critical` requires a concrete failure path (wrong result, crash, deadlock, data loss, broken backend, failed docs build, or uncompletable workflow); downgrade or drop speculative edge cases and low-value nits. If nothing survives, write `No findings.` and one line on what was checked.

## 8. Report

Keep the output tight and free of repetition: print the main comment, then each inline comment once, then a two-bullet check summary (not posted).

- **Verdict**: bold recommendation (`Ready for senior review`, `Changes requested`, or `Blocked`) + severity counts + head/base SHA, then one short sentence on what blocks merge or needs a maintainer decision.
- **Reproduction**: two short bullets (under ~300 chars each). `Result:` starts with `**Passed**`, `**Failed**`, `**Partial**`, or `**Not run**` + one-sentence functional/metric conclusion (use a tiny table only for multiple metrics). `Setup:` names only hardware count/type and config/command (omit host, OS, driver, venv/Python paths, and SHA).
- **No duplicate findings**: the table indexes every finding (`F1`, `F2`, ...), sorted by severity then category. Write a `#### F<n>` section in the main comment **only for `Posted = here` findings**. `inline` findings live once in `### Inline comments` (each carrying problem, `Suggested fix:`, and one-line `Evidence:`).
- **No trailing noise**: no `Dismissed:` list, no restated Goal/Idea/Changes/Risk bullets, and no tables of passing doc pages.

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

<Problem and impact in 1–3 plain sentences.>

Suggested fix: <one sentence, plus code block if code changes.>

Code walkthrough:

<Quoted snippets under `path:line` lines, then input-to-failure steps. Omit when not about code.>

Evidence: <one line: command and result, or trace; for unresolved, what settles it.>
````

````markdown
### Inline comments (posted on diff lines)

`rlinf/x.py:42`

**F1 · critical · correctness: <plain title>**

<Problem (1–3 sentences), `Suggested fix:` + verified `suggestion`/code block, and one-line `Evidence:`.>
````

````markdown
## Check summary (not posted)

- Checks: `route.py check` (<result>) · `lint_pr` (<result>) · tests (<command, hardware, result; or skipped + why>)
- Context: compared with <closest sibling on main>; <any area check that could not run and why>
````

## 9. Post only when asked

Stop after step 8; never post by default. When the user asks to post, update, or reply on GitHub, read [posting.md](posting.md) and run `lint_review.py lint` before posting. Approve only when explicitly asked.
