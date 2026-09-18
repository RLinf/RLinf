---
name: rlinf-conventions
description: RLinf codebase conventions and patterns to follow when maintaining code
metadata:
  type: project
---

# RLinf Code Conventions

This document records **specific patterns and anti-patterns** observed in the RLinf codebase that should be followed when adding new features or maintaining existing code.

Related: See [AGENTS.md](../AGENTS.md) for architecture overview and [CONTRIBUTING.md](../CONTRIBUTING.md) for code style.

---

## Documentation

### ❌ Anti-pattern: Scattered README.md files

**Do NOT create README.md files scattered across subdirectories.**

```
# Wrong
rlinf/workers/rollout/grpc/README.md
examples/embodiment/so101/README.md
```

**Why:** RLinf uses a centralized documentation system based on Sphinx RST files in `docs/`.

### ✅ Pattern: Use RST documentation

Documentation goes into `docs/source-en/rst_source/` and `docs/source-zh/rst_source/`:

```
docs/source-en/rst_source/
  examples/
    embodied/
      so101.rst          # SO-101 example
  extending/
    new_env.rst          # How to add new environments
  guides/
    grpc_backend.rst     # gRPC inference guide
```

**When to update docs:**
- New robot/environment: Add to `docs/source-{en,zh}/rst_source/examples/embodied/`
- New feature/guide: Add to `docs/source-{en,zh}/rst_source/guides/`
- New API: Update `docs/source-{en,zh}/rst_source/reference/api/`

**Exception:** Top-level `README.md` is allowed for the project root.

---

## Configuration

### ❌ Anti-pattern: .env files

**Do NOT use `.env` or `.env.example` files for configuration.**

```bash
# Wrong
examples/embodiment/so101/lerobot_grpc_policy.env.example
SO101_GRPC_HOST=0.0.0.0
SO101_GRPC_PORT=50051
```

**Why:** RLinf uses Hydra YAML configs consistently across the entire project.

### ✅ Pattern: Use YAML configs

All configuration goes through Hydra YAML files in `examples/`:

```yaml
# examples/embodiment/config/realworld_so101_dagger_openpi.yaml
defaults:
  - realworld_so101_collect_data

rollout:
  use_grpc_backend: true
  grpc:
    server_address: "localhost:50051"
    timeout: 30.0
```

**Pattern to follow:**
- Base configs with common settings (e.g., `realworld_so101_collect_data.yaml`)
- Task-specific configs that override via `defaults:` (e.g., `realworld_so101_dagger_openpi.yaml`)
- Pass runtime values via command-line: `python script.py rollout.grpc.server_address=192.168.1.100:50051`

---

## Scripts and Tools

### ❌ Anti-pattern: Mixed script locations

**Do NOT mix script hierarchies without clear purpose.**

```
# Inconsistent
examples/embodiment/run_dagger_sft.sh
examples/embodiment/run_sft.sh
examples/embodiment/so101/run_so101_inference.sh
```

### ✅ Pattern: Consistent script organization

Follow existing patterns in the codebase:

1. **Entry point scripts** → `examples/<domain>/run_<task>.sh`
   ```
   examples/embodiment/run_embodiment.sh
   examples/embodiment/run_so101_dagger.sh
   ```

2. **Development/check tools** → `toolkits/<purpose>/`
   ```
   toolkits/realworld_check/check_so101_devices.py
   toolkits/realworld_check/run_so101_policy.py
   ```

3. **Service wrappers** (if needed) → keep with entry scripts or `ray_utils/`

**Look at existing implementations:**
- Franka robot: How are scripts organized?
- Other environments: Where do their tools live?

---

## Reusing Existing Infrastructure

### ❌ Anti-pattern: Reimplementing existing features

**Do NOT reimplement features that already exist in RLinf.**

Example violation: Implementing a separate SO-101 DAgger pipeline when RLinf already has:
- `hg-dagger` for human-guided DAgger
- Unified data collection and training pipeline
- Standardized intervention mechanisms

### ✅ Pattern: Extend existing systems

**Before adding new code, check:**

1. **Does this feature already exist?**
   - DAgger → `rlinf/runners/embodied/` already has DAgger support
   - Data collection → `rlinf/envs/wrappers/collect_episode.py`
   - Teleoperation → `rlinf/envs/wrappers/teleop/`

2. **How do similar robots do it?**
   ```bash
   # Look at Franka, for example
   find . -path "*/franka/*" -name "*.py" | head -5
   grep -r "franka" examples/embodiment/config/
   ```

3. **Can I extend instead of replace?**
   - Add a new wrapper instead of a new pipeline
   - Add a config variant instead of a new script
   - Register in existing registries instead of building parallel ones

**Example: SO-101 should follow Franka's pattern:**
- Config: `examples/embodiment/config/realworld_so101_*.yaml`
- Environment: `rlinf/envs/real/so101/base.py` (already correct)
- Teleop: `rlinf/robotics/parts/teleop/so101_leader.py` (already correct)
- Data loaders: Extend existing patterns in `rlinf/data/`

---

## When Introducing New Patterns

Sometimes you DO need to introduce something new. When doing so:

1. **Check if it's truly needed**
   - Is there no existing pattern that covers this?
   - Have you looked at ALL similar features in the codebase?

2. **Discuss first**
   - Ask: "I noticed X doesn't have a pattern for Y. Should I follow Z's approach?"
   - Don't silently introduce external conventions (like .env files)

3. **Document the new pattern**
   - Update this file
   - Add to AGENTS.md if it's architectural
   - Add examples showing the pattern

4. **Be consistent**
   - If you add it for SO-101, it should work the same way for future robots
   - Think about the **second instance** of this pattern

---

## Macro Perspective (宏观视角)

> "要有宏观视角，要遵从惯例" - Maintain awareness of the whole codebase and follow established conventions.

**What this means in practice:**

1. **Before implementing:**
   - Search: `find . -name "*<similar_feature>*"`
   - Grep: `grep -r "<keyword>" examples/ rlinf/`
   - Read: Look at 2-3 similar existing implementations

2. **During implementation:**
   - File paths: Where do similar files live?
   - Config: How is this configured in existing code?
   - Scripts: What's the entry point pattern?
   - Documentation: Where would someone look for docs on this?

3. **After implementation:**
   - Does my code look foreign compared to the rest?
   - If I added a new pattern, did I document it?
   - Can the next person extend this without creating a third pattern?

---

## Memory Aid: Quick Checklist

Before committing a feature, check:

- [ ] No scattered README.md files (use RST in docs/)
- [ ] No .env files (use YAML configs)
- [ ] Scripts follow existing location patterns
- [ ] Looked at how similar features are implemented (e.g., Franka)
- [ ] Extended existing infrastructure rather than reimplemented
- [ ] File organization matches the codebase structure
- [ ] Configuration follows Hydra YAML patterns
- [ ] Documentation will be added to docs/ (not inline READMEs)

**When in doubt:** Find 2-3 examples of similar features and follow their pattern.
