# YAM PPO + TOPReward (`yam_ppo_openpi`)

This is the AI2-facing Markdown guide for the baseline YAM PPO config:
`examples/embodiment/config/yam_ppo_openpi.yaml`.

This config runs:

- PPO + GAE
- π₀.5 / OpenPI policy
- TOPReward dense reward
- no VLM subtask planning (`subtask_interval: 0`)

For the variant that also enables VLM subtask planning, see
[yam_ppo_openpi_topreward](yam_ppo_openpi_topreward.md).

## Topology

Canonical topology:

- Beaker runs all Ray workers
- the desktop runs only `RobotServer`
- `RemoteEnv` connects over gRPC through the reverse SSH tunnel

Main component placement:

- GPU 0: actor
- GPU 1: rollout
- GPU 2: VLM planner / TOPReward
- CPU: `RemoteEnv`

## VLM Planner Placement

`train_embodied_agent_staged.py` now launches the VLM planner through RLinf's
placement stack instead of as a standalone Ray `num_gpus=1` actor. This matters
because actor and rollout workers were already using RLinf-managed GPU
isolation, while the older VLM planner path only used a best-effort
`CUDA_VISIBLE_DEVICES` heuristic.

Current behavior:

- actor, rollout, and VLM planner all use RLinf placement-backed GPU isolation
- the default YAM single-node layout still resolves to GPU 0 for actor, GPU 1
  for rollout, and GPU 2 for the VLM planner
- the planner uses the `beaker_vlm` node group by default

Planner placement overrides:

```yaml
vlm_planner:
  # Optional explicit GPU index inside the planner node group.
  placement: 2
  # Optional node-group override. Defaults to "beaker_vlm".
  node_group: "beaker_vlm"
```

Use an explicit `vlm_planner.placement` override if you want to pin the VLM to
a different GPU than the default inferred one.

## Standard Workflow

Submit training from the repo root:

```bash
bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi \
    --model-path /path/to/RLinf-Pi05-SFT
```

Then start the desktop-side robot server:

```bash
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam.yaml \
    --remote-host <tailscale-ip>
```

For pipeline testing without hardware:

```bash
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam.yaml \
    --dummy
```

## Key Config Knobs

Model paths:

```yaml
rollout:
  model:
    model_path: "/path/to/RLinf-Pi05-SFT"

actor:
  model:
    model_path: "/path/to/RLinf-Pi05-SFT"

vlm_planner:
  model_path: "Qwen/Qwen3-VL-8B-Instruct"
```

Task description:

```yaml
env:
  train:
    task_description: "pick up the red block"
```

Reward / planner settings:

```yaml
env:
  train:
    top_reward_enabled: True
    subtask_interval: 0
```

## Local Simulated Desktop Mode

`train_embodied_agent_staged.py` now supports simulating the remote desktop
input path locally. This keeps the normal `RemoteEnv -> gRPC -> RobotServer`
flow, but the training process starts a local dummy `RobotServer`
automatically, so no separate desktop machine or reverse SSH tunnel is needed.

Enable it with:

```bash
python examples/embodiment/train_embodied_agent_staged.py \
    --config-path examples/embodiment/config \
    --config-name yam_ppo_openpi \
    env.remote_desktop_simulation.enabled=true
```

Optional overrides:

```bash
python examples/embodiment/train_embodied_agent_staged.py \
    --config-path examples/embodiment/config \
    --config-name yam_ppo_openpi \
    env.remote_desktop_simulation.enabled=true \
    env.remote_desktop_simulation.env_config_path=/path/to/yam.yaml \
    env.train.remote_server_url=localhost:50051 \
    env.eval.remote_server_url=localhost:50051
```

Shared config block:

```yaml
env:
  remote_desktop_simulation:
    enabled: true
    dummy: true
    env_config_path: null
    startup_timeout: 30.0
```

Notes:

- Only local RemoteEnv URLs are supported in this mode, such as
  `localhost:50051` or `127.0.0.1:50051`.
- This mode is for dummy input simulation only.
- For real hardware, keep using `start_robot_server.sh` on the desktop.

## Related Docs

- [quickstart](quickstart.md)
- [training_architecture](training_architecture.md)
- [network_infrastructure](network_infrastructure.md)
- [openpi_joint_logprob](openpi_joint_logprob.md)
- [openpi_nan_gradients](openpi_nan_gradients.md)
