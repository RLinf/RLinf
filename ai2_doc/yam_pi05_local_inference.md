# YAM Local PI0.5 Inference (`infer_embodied_agent.py`)

This is the AI2-facing Markdown guide for the local YAM inference entrypoint:
`examples/embodiment/infer_embodied_agent.py`.

This workflow runs:

- desktop-local inference only
- π₀.5 / OpenPI policy inference
- YAM follower servers on localhost (`1234` / `1235`)
- desktop-local `RobotServer` over gRPC (`localhost:50051`)
- `640x360@30` RealSense inputs with raw gRPC image transport
- no Ray, no Beaker, no reverse SSH tunnel in the default path

The current recommended env config for this workflow is:
`examples/embodiment/config/env/yam_pi05_follower.yaml`.

## Topology

Canonical topology:

- one desktop runs the follower arm servers
- the same desktop runs `RobotServer`
- the same desktop runs `infer_embodied_agent.py`

Main local process layout:

- follower server for right arm: `localhost:1234`
- follower server for left arm: `localhost:1235`
- `RobotServer`: `localhost:50051`
- inference client: local Python process calling gRPC

Unlike the Beaker training flow, this path does not require:

- Ray
- `RemoteEnv`
- reverse SSH tunneling

## Standard Workflow

### Step 1: Start the desktop-side robot server

Recommended follower-based setup:

```bash
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam_pi05_follower.yaml \
    --use-follower-servers \
    --no-tunnel
```

This does three things:

- launches YAM follower servers
- waits for ports `1234` and `1235`
- starts `RobotServer` on `localhost:50051`
- serves uncompressed `640x360` images to match the original follower setup more closely
- keeps the downstream PI0.5 image path closer to the original runtime: raw `640x360` uint8 frames are resized directly to `224x224` without letterbox padding

### Step 2: Run local inference

Example:

```bash
python examples/embodiment/infer_embodied_agent.py \
    --model-path sengi/pi05_put_dolls_cloth_lerobot \
    --task-description "put the dolls on the cloth"
```

The default `--grpc-timeout` is `120` seconds, so you only need to pass it
when overriding that default.

Behavior of the current inference client:

- stays quiet by default and auto-runs immediately
- optionally shows the current state / predicted chunk preview with `--show-state-chunk`
- when `--show-state-chunk` is enabled, waits for `Enter` before auto-run starts
- during auto-run, pressing `s` stops the current episode, returns home, and then waits for `Enter` before the next episode starts
- pressing `Ctrl+C` resets to home and then switches the arms into zero-torque / zero-gravity mode before the inference process exits
- runs chunked async inference with `weighted_average` aggregation
- follows the original async client more closely by using `must_go` scheduling and
  filtering out repeated / near-identical observations before re-predicting
- returns home after each episode
- sends `Reset` on clean shutdown

## Key Runtime Knobs

Model and task:

```bash
python examples/embodiment/infer_embodied_agent.py \
    --model-path <hf-model-or-local-path> \
    --task-description "<natural-language-task>"
```

Only add `--grpc-timeout <seconds>` if you need a timeout other than the
default `120`.

Inference-side knobs:

```bash
python examples/embodiment/infer_embodied_agent.py \
    --config-name pi05_yam_follower \
    --server-url localhost:50051 \
    --action-chunk 30 \
    --num-steps 10 \
    --show-state-chunk \
    --max-episode-steps 10000 \
    --chunk-size-threshold 0.0 \
    --aggregate-fn-name weighted_average \
    --return-home \
    --return-home-steps 50
```

Robot-server-side knobs:

```bash
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam_pi05_follower.yaml \
    --use-follower-servers \
    --no-tunnel \
    --port 50051
```

Follower-server-side knobs:

```bash
python scripts/start_yam_follower_servers.py \
    --right-can can_right \
    --left-can can_left \
    --right-port 1234 \
    --left-port 1235
```

Optional gripper limit overrides:

```bash
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam_pi05_follower.yaml \
    --use-follower-servers \
    --gripper-open <value> \
    --gripper-close <value> \
    --no-tunnel
```

## Where To Set Common Parameters

### Model path

Best place:

- pass `--model-path` at runtime

Default is defined in:

- `examples/embodiment/infer_embodied_agent.py`

### Task description

Best place:

- pass `--task-description` at runtime

Important note:

- `infer_embodied_agent.py` sends the task to `RobotServer` with `SetTaskDescription`
- this overrides the placeholder `task_description` field in the env YAML during inference

### Camera serial numbers and camera roles

Set them in:

- `examples/embodiment/config/env/yam_pi05_follower.yaml`

Relevant fields:

```yaml
img_height: 360
img_width: 640
main_camera: "cam_top"
wrist_cameras: ["cam_left", "cam_right"]
compress_images: False

camera_cfgs:
  cam_right:
    camera:
      serial_number: "128422272697"
      resolution: [640, 360]
  cam_top:
    camera:
      serial_number: "215222073684"
      resolution: [640, 360]
  cam_left:
    camera:
      serial_number: "218622275075"
      resolution: [640, 360]
```

Important note:

- the env camera roles stay semantic on the RLinf side: `top`, `left`, `right`
- `pi05_yam_follower` now reads the checkpoint's `config.json` and infers the
  expected visual feature order automatically
- this matters because different PI0.5 checkpoints can use different camera
  orders, for example `left, top, right` or `left, right, top`

### Episode length and control rate

Set them in:

- `examples/embodiment/config/env/yam_pi05_follower.yaml`

Relevant fields:

```yaml
max_episode_steps: 10000
control_rate_hz: 30.0
```

### Follower server ports

Set them in both places if you change them:

- `scripts/start_yam_follower_servers.py`
- `examples/embodiment/config/env/robot_clients/yam_follower_left.yaml`
- `examples/embodiment/config/env/robot_clients/yam_follower_right.yaml`

The default mapping is:

- right arm: `1234`
- left arm: `1235`

## Important Files

Inference entrypoint:

- `examples/embodiment/infer_embodied_agent.py`

Recommended env config:

- `examples/embodiment/config/env/yam_pi05_follower.yaml`

Follower robot client configs:

- `examples/embodiment/config/env/robot_clients/yam_follower_left.yaml`
- `examples/embodiment/config/env/robot_clients/yam_follower_right.yaml`

Follower launcher:

- `scripts/start_yam_follower_servers.py`

Robot server launcher:

- `scripts/start_robot_server.sh`

Follower client compatibility layer:

- `rlinf/envs/yam/yam_follower_client.py`

YAM env wrapper:

- `rlinf/envs/yam/yam_env.py`

OpenPI inference path:

- `rlinf/models/embodiment/openpi/openpi_action_model.py`
- `rlinf/models/embodiment/openpi/dataconfig/__init__.py`
- `rlinf/models/embodiment/openpi/dataconfig/libero_dataconfig.py`
- `rlinf/models/embodiment/openpi/policies/libero_policy.py`

## Notes and Troubleshooting

Follower mode vs direct CAN mode:

- the recommended local inference path is the follower-based config
- this avoids depending on the older direct-CAN robot config path during inference

If the process still truncates at `100` steps:

- restart `RobotServer`
- the YAML episode limits are read at server startup, not live-reloaded

If follower servers fail to connect:

- make sure no stale processes are already bound to `1234`, `1235`, or `50051`
- restart both the follower launcher and `RobotServer`

If shutdown/reset returns home but still reports a reset failure:

- restart `RobotServer` so it loads the latest `YAMEnv` reset logic
- the current reset path now waits briefly for follower state feedback to settle before declaring failure
- if needed, tune `reset_motion_settle_timeout_s`, `reset_motion_poll_interval_s`, and `reset_motion_tolerance` in `examples/embodiment/config/env/yam_pi05_follower.yaml`

If behavior is numerically stable but task performance is poor:

- verify the camera serial numbers and camera role mapping
- verify `--model-path` and `--task-description`
- keep using the current uint8 image path in `proto_to_obs`; this matches the native PI0.5/OpenPI inference expectation
- keep using the direct `640x360 -> 224x224` resize path for `pi05_yam_follower`; the original `lerobot_pi05` helper does a plain bilinear resize rather than `resize_with_pad`

## Related Docs

- [quickstart](quickstart.md)
- [network_infrastructure](network_infrastructure.md)
- [training_architecture](training_architecture.md)
- [yam_ppo_openpi](yam_ppo_openpi.md)
- [yam_ppo_openpi_topreward](yam_ppo_openpi_topreward.md)
