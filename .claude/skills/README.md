# Project skills

Skills in this folder live under `.claude/skills/` for Claude tooling.
If this project also uses Cursor, treat `.cursor/skills/` as a separate Cursor-specific location unless the two directories are intentionally kept mirrored.

**If a skill is not recognized:**

1. **Restart Cursor** – Skills are discovered when Cursor starts.
2. **Check discovery** – Open **Cursor Settings** (Ctrl+Shift+J / Cmd+Shift+J) → **Rules**. Skills appear under **Agent Decides**.
3. **Invoke manually** – In Agent chat, type `/` and search for the skill name (e.g. `add-example-doc-model-env`).

Each skill is a folder whose name **must match** the `name` in its `SKILL.md` frontmatter.
