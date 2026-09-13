# Project skills

This directory is the single source for RLinf project skills. Codex and Cursor
discover `.agents/skills/` directly. Claude discovers the same skill directories
through the symlinks in `.claude/skills/`.

Edit skills only in this directory. When adding or removing a skill, update the
corresponding symlink in `.claude/skills/`. Repository-wide agent guidance lives
in [AGENTS.md](../../AGENTS.md); `CLAUDE.md` points to the same file.

**If a skill is not recognized:**

1. **Restart the client** – Skills are discovered when the session starts.
2. **Invoke manually** – Reference the skill by name (e.g. `add-example-doc-model-env`)
   and follow the steps in its `SKILL.md`.

Each skill is a folder whose name **must match** the `name` in its `SKILL.md` frontmatter.
