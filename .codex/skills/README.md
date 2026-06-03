# Project skills

Skills in this folder live under `.codex/skills/` for use with OpenAI Codex.
They mirror the canonical Claude skills in `.claude/skills/`; keep the three
copies (`.claude/`, `.cursor/`, `.codex/`) in sync when a skill changes.
Codex reads its repository guidance from [AGENTS.md](../../AGENTS.md).

**If a skill is not recognized:**

1. **Restart Codex** – Skills are discovered when the session starts.
2. **Invoke manually** – Reference the skill by name (e.g. `add-example-doc-model-env`)
   and follow the steps in its `SKILL.md`.

Each skill is a folder whose name **must match** the `name` in its `SKILL.md` frontmatter.
