# MolmoAct2 LIBERO Eval Validation

This file records the small-scale validation results for the MolmoAct2-LIBERO evaluation support added in this PR.

## Setup

- Model: MolmoAct2-LIBERO
- Input observations: agentview image + wrist image + robot state + language instruction
- norm_tag: libero
- num_steps: 10
- Rendering backend: OSMesa
- Number of environments: 1
- Rollout trajectories per suite: 5
- Max episode steps: 500
- Seed: 0

## Small-scale validation results

| Suite | RLinf task suite name | Trajectories | Success |
|---|---|---:|---:|
| Spatial | libero_spatial | 5 | 5/5 |
| Object | libero_object | 5 | 5/5 |
| Goal | libero_goal | 5 | 5/5 |
| Long | libero_10 | 5 | 5/5 |

Average small-scale success rate: 100%.

## LIBERO-10 detailed result

Command:

```bash
python evaluations/eval_embodied_agent.py \
  --config-name libero_10_molmoact2_eval \
  rollout.model.num_steps=10 \
  env.eval.rollout_epoch=5 \
  env.eval.max_episode_steps=500 \
  env.eval.max_steps_per_rollout_epoch=500 \
  env.eval.total_num_envs=1 \
  env.eval.seed=0 \
  runner.logger.experiment_name=libero_10_molmoact2_eval_two_view_5traj_500steps
```

Observed result:

```text
task_id=0 success=True
task_id=1 success=True
task_id=2 success=True
task_id=3 success=True
task_id=4 success=True

eval/success_once: 1.0
eval/success_at_end: 1.0
eval/num_trajectories: 5
```

## Note

This is a small-scale validation with 5 trajectories per suite, not a full benchmark reproduction.
