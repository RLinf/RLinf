# Standalone Evaluation Scripts

Scripts in `toolkits/standalone_eval_scripts/` run **outside** the RLinf
embodied eval pipeline (`evaluations/eval_embodied_agent.py` + Hydra YAML).
They are aligned with upstream framework eval flows and provide finer-grained
per-task metrics, but are typically slower than distributed RLinf eval.

| Subdirectory | Framework | Benchmarks |
|--------------|-----------|------------|
| `openpi/` | OpenPI | LIBERO, MetaWorld, CALVIN |
| `dexbotic/` | Dexbotic | LIBERO |
