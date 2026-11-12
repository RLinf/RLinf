# Project skills

This directory is the single source for RLinf's project skills. Codex, Cursor, and Gemini CLI discover `.agents/skills/` directly, and Claude Code discovers the same directories through the symlinks in `.claude/skills/`. Edit skills only here, keep one symlink per skill in `.claude/skills/`, and give each skill a folder whose name matches the `name` in its `SKILL.md`. Repository-wide agent guidance lives in [AGENTS.md](../../AGENTS.md), which `CLAUDE.md` links to.

## Area skills and workflow skills

An area skill owns part of the repository. It tells an agent what to read before changing those files and checks the change afterwards, so the same rules hold while a change is written and when it is reviewed.

| Skill | Owns |
|---|---|
| [code](code/SKILL.md) | code, tests, configs, and CI: `rlinf/`, `examples/`, `tests/`, `toolkits/`, `evaluations/`, `ray_utils/`, `.github/`, any `.py` file |
| [docs](docs/SKILL.md) | `docs/`, `README.md`, `README.zh-CN.md` |
| [install](install/SKILL.md) | `requirements/`, `pyproject.toml`, `docker/`, `tests/e2e_tests/`, and the Docker, install, and e2e workflows |

A workflow skill acts on a pull request and uses the area skills.

| Skill | Does |
|---|---|
| [create-pr](create-pr/SKILL.md) | writes and lints a PR title and description, after the area checks pass |
| [review-pr](review-pr/SKILL.md) | reviews a PR, runs the check of every area it touches, and posts the review after asking |

## The area skill contract

Every area skill has the same shape, so that `route.py`, `create-pr`, and `review-pr` work with any of them without naming it.

- **`paths:`** in the frontmatter is a YAML block list of the globs the skill owns, relative to the repository root. `*` matches within one directory and `**` across directories; quote a pattern that starts with `*`. Claude Code and Cursor load the skill when an agent works on a matching file, and `route.py` reads the same list. A file may belong to more than one area. The field is not part of the Agent Skills specification; Codex and Gemini CLI ignore it.
- **`description`** opens with the files and the action, because clients shorten long descriptions from the end.
- **Write** says what to read before editing and points to the rules by section. Each rule lives in one place: `docs/rules/` for docs, `AGENTS.md` and `code/patterns.md` for code, and `install/SKILL.md` for install.
- **Check** is `check.py`, run as `python3 .agents/skills/<area>/check.py [--base REF] [--all-lines] [FILE ...]`. With no files it checks the area's files that differ from the merge base of `--base` (default: `main` of the RLinf remote), including uncommitted and untracked files. Line findings count only on lines the change adds, so a check reports what the change introduces; `--all-lines` reports every line. A list of judgment items follows in `SKILL.md`, for what a script cannot decide.
- **Verify** (optional) covers runs that need GPUs, datasets, or other assets.

## Findings

A check prints one finding per line, most severe first, and then a summary line:

```text
<severity> <path>[:<line>] <rule>: <message>
```

One severity scale holds for every check and for the review:

- **critical**: wrong results, a crash, data loss, a broken existing path or backend, a docs build that fails Read the Docs, or a documented workflow that cannot be completed.
- **major**: a break with a rule of the area, or a missing test, doc, or CI entry for a user-facing change. It must change before merge.
- **minor**: a local fix such as naming, a comment, or a line of wording. It should change but does not block.

A check exits 1 when it reports a critical or major finding. A check that cannot run a step, such as a missing tool, prints `skipped:` and the reason.

## Route a change

[route.py](route.py) maps the changed files to the areas that own them and runs their checks:

```bash
python3 .agents/skills/route.py list    # touched areas, the SKILL.md to read, and their files
python3 .agents/skills/route.py check   # run each touched area's check
```

Both take `--base REF`. Files no area owns, such as agent instructions, are listed as unrouted. The skills come from the checkout that holds `route.py`, and the repository under check is the one the current directory is in, so a review can run the current skills inside the worktree of an older PR.

## Add a skill

Create `.agents/skills/<name>/SKILL.md`, add the symlink `.claude/skills/<name>`, and add the skill to the tables here and in AGENTS.md. An area skill also declares `paths:` and ships `check.py` built on `route.check_parser`, `route.select`, and `route.report`; `route.py` refuses a skill that declares `paths:` without a check.

## If a skill is not recognized

Clients discover skills when a session starts, so restart the client after adding or renaming one. Any agent can also read a skill's `SKILL.md` by path and follow it.
