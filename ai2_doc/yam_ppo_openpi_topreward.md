# YAM PPO + TOPReward + Subtask Planning (`yam_ppo_openpi_topreward`)

This is the AI2-facing Markdown guide for the staged YAM PPO config:
`examples/embodiment/config/yam_ppo_openpi_topreward.yaml`.

This config runs:

- PPO + GAE
- π₀.5 / OpenPI policy
- TOPReward dense reward
- VLM subtask planning enabled (`subtask_interval > 0`)

For the simpler TOPReward-only variant, see
[yam_ppo_openpi](yam_ppo_openpi.md).

## Topology

Canonical topology:

- Beaker runs all Ray workers
- the desktop runs only `RobotServer`
- `RemoteEnv` connects over gRPC through the reverse SSH tunnel

Main component placement:

- GPU 0: actor
- GPU 1: rollout
- GPU 2: VLM planner / TOPReward / subtask planning
- CPU: `RemoteEnv`

## Standard Workflow

Submit training from the repo root:

```bash
bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi_topreward \
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
    task_description: "pick up the red block and place it in the bowl"
```

Reward / planner settings:

```yaml
env:
  train:
    top_reward_enabled: True
    top_reward_max_frames: 16
    subtask_interval: 3

vlm_planner:
  max_new_tokens_subtask: 64
  max_new_tokens_reward: 16
  success_threshold: 0.5
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
    --config-name yam_ppo_openpi_topreward \
    env.remote_desktop_simulation.enabled=true
```

Optional overrides:

```bash
python examples/embodiment/train_embodied_agent_staged.py \
    --config-path examples/embodiment/config \
    --config-name yam_ppo_openpi_topreward \
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
