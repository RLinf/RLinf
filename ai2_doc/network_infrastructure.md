# RLinf Network Infrastructure for Beaker Training

This document describes the network architecture for running embodied RL training
on AI2 Beaker GPU clusters while the physical robot remains on a local desktop.

## Overview

```
                         Tailscale VPN (mesh)
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │  Robot Desktop                    Beaker Container       │
    │  (Tailscale IP: 100.x.y.z)       (Tailscale IP: 100.a.b.c)
    │                                                          │
    │  ┌──────────────┐     reverse     ┌──────────────────┐   │
    │  │ RobotServer  │◄── SSH tunnel ──│ RemoteEnv        │   │
    │  │ (gRPC :50051)│   (desktop →    │ (gRPC client)    │   │
    │  │              │    container)    │                  │   │
    │  │ YAMEnv       │                 │ Ray Cluster      │   │
    │  │  └ Robot HW  │                 │  ├ Actor  (GPU0) │   │
    │  │  └ Cameras   │                 │  ├ Rollout(GPU1) │   │
    │  └──────────────┘                 │  └ VLM    (GPU2) │   │
    │                                   └──────────────────┘   │
    └──────────────────────────────────────────────────────────┘
```

**Key constraint:** Beaker containers can accept inbound connections but cannot
initiate outbound connections to arbitrary hosts. The desktop *can* reach the
container via Tailscale. The solution is a **reverse SSH tunnel** initiated by the
desktop, which exposes the local gRPC port inside the container.

## Components

### 1. Robot Desktop (local machine)

Runs two processes:

| Process | Purpose |
|---|---|
| **RobotServer** | gRPC server wrapping `YAMEnv` — drives the physical robot, streams observations |
| **Reverse SSH tunnel** | `ssh -R 50051:localhost:50051 shiruic@<container-tailscale-ip>` — makes the gRPC server reachable at `localhost:50051` inside the container |

Both are managed by `scripts/start_robot_server.sh`.

### 2. Beaker Container (GPU cluster)

Runs the Ray-based training pipeline. Components are placed on GPUs by the
scheduler:

| Component | GPU | Role |
|---|---|---|
| **Actor** | GPU 0 | Policy training (FSDP) |
| **Rollout** | GPU 1 | Action inference |
| **VLM Planner** | GPU 2 | Dense reward via Qwen3-VL-8B (TOPReward configs only) |
| **RemoteEnv** | CPU | gRPC client connecting to `localhost:50051` (via the SSH tunnel) |

The container runs Tailscale in **userspace networking** mode (no TUN device
needed in unprivileged containers).

## Network Stack

### Tailscale

Every Beaker replica installs and starts Tailscale on boot:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
tailscaled --tun=userspace-networking --state=mem: &
tailscale up --authkey=${TAILSCALE_AUTHKEY} --hostname=beaker-${BEAKER_REPLICA_RANK}
```

- `--tun=userspace-networking` — required for unprivileged containers (no
  `/dev/net/tun`)
- `--state=mem:` — ephemeral state, no persistent disk needed
- `--authkey` — pulled from the Beaker secret `tailscale_authkey_shirui`
- `--hostname=beaker-<rank>` — makes replicas distinguishable in the Tailscale
  admin console

After startup, the container's Tailscale IP is printed to logs:

```
=== Tailscale IP ===
100.a.b.c
==================
```

### Reverse SSH Tunnel

The desktop initiates an SSH connection *to* the container (which it can reach
via Tailscale) and maps a remote port back to itself:

```
Desktop (initiator)                    Container (listener)
ssh -R 50051:localhost:50051  ──────►  sshd
                                        │
                                        └─► localhost:50051 now routes
                                            back to Desktop:50051
```

The `-R` flag means: "anyone connecting to port 50051 on the container gets
forwarded through this SSH connection to port 50051 on my local machine."

This is the critical trick — the container can't reach the desktop directly,
but the desktop can reach the container. By initiating the connection from the
desktop and requesting a reverse port forward, traffic flows bidirectionally
through the single SSH connection.

### gRPC Protocol

Communication between `RemoteEnv` (client) and `RobotServer` (server) uses
gRPC with Protocol Buffers. The proto definition lives at
`rlinf/envs/remote/proto/robot_env.proto`.

**RPCs:**

| RPC | Direction | Purpose |
|---|---|---|
| `GetSpaces` | client → server | Fetch observation/action space metadata |
| `Reset` | client → server | Reset environment, return initial observation |
| `ChunkStep` | client → server | Send action chunk, receive step results |
| `SetTaskDescription` | client → server | Update task string (used by VLM planner) |
| `Close` | client → server | Graceful shutdown |

**Observation encoding:**

- **Joint states**: raw `float32` bytes with shape metadata
- **Camera images**: JPEG-compressed (quality 90) to reduce bandwidth
  (~10x smaller than raw); decoded client-side via OpenCV
- **Max message size**: 16 MB (configurable via `grpc_max_message_size`)
- **Timeout**: 30s per RPC (configurable via `grpc_timeout`)

## Ray Cluster on Beaker

### Single-Replica (default)

All components run in one Beaker replica. Ray head starts on the same node.

```
Replica 0 (head)
  ├── Ray head (:6379)
  ├── Actor    (GPU 0)
  ├── Rollout  (GPU 1)
  ├── VLM      (GPU 2, TOPReward only)
  └── Env      (CPU, gRPC to localhost:50051)
```

### Multi-Replica

Replica 0 is the Ray head; replicas 1..N join as workers. Worker replicas
discover the head via `BEAKER_LEADER_REPLICA_HOSTNAME` (set automatically by
Beaker for multi-replica experiments).

```
Replica 0 (head)                    Replica 1..N (workers)
  ├── Ray head (:6379)  ◄──────────  Ray worker
  ├── Actor (GPU 0)                   ├── Actor (GPU 0)
  ├── Rollout (GPU 1)                 └── Rollout (GPU 1)
  ├── VLM (GPU 2)
  └── Env (CPU)
```

Worker discovery flow in `ray_utils/start_ray_beaker.sh`:

1. Resolve `BEAKER_LEADER_REPLICA_HOSTNAME` via DNS (retries up to 5 min)
2. `ray start --address=<head-ip>:6379` (retries up to 5 min)
3. Block and monitor Ray connection; exit when head disconnects

## Configuration

### Hydra Config: `remote_yam`

File: `examples/embodiment/config/env/remote_yam.yaml`

```yaml
env_type: remote
remote_server_url: "${oc.env:ROBOT_SERVER_URL,localhost:50051}"
compress_images: true
jpeg_quality: 90
grpc_max_message_size: 16777216  # 16 MB
grpc_timeout: 30.0
auto_reset: false
ignore_terminations: true
```

The training script overrides the env config to `remote_yam` for both train and
eval environments:

```
'env/remote_yam@env.train' 'env/remote_yam@env.eval'
```

### Supported Training Configs

| Config | Entry Point | GPUs | Components |
|---|---|---|---|
| `yam_ppo_openpi` | `train_embodied_agent.py` | 2 | actor, rollout, env |
| `yam_ppo_openpi_topreward` | `train_embodied_agent_staged.py` | 3 | actor, rollout, env, VLM planner |

## Scripts Reference

### `scripts/submit_yam_training.sh`

Submits a Beaker training job via gantry.

```bash
# Basic PPO (2 GPUs)
bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi \
    --model-path thomas0829/folding_towel_pi05 \
    --dry-run

# TOPReward (3 GPUs)
bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi_topreward \
    --model-path thomas0829/folding_towel_pi05 \
    --dry-run

# With Hydra overrides
bash scripts/submit_yam_training.sh \
    --model-path /weka/.../checkpoint \
    -- algorithm.update_epoch=2 runner.save_freq=50
```

**What the script does:**

1. Auto-detects config type (basic vs TOPReward) and sets GPU count
2. Builds Hydra training command with cluster placement overrides
3. Base64-encodes the training command (avoids nested shell quoting issues)
4. Builds entrypoint that installs Tailscale, starts Ray, runs training
5. Submits via `gantry run` with the correct Beaker image, secrets, and mounts

**Key options:**

| Option | Default | Description |
|---|---|---|
| `--config` | `yam_ppo_openpi` | Hydra config name |
| `--model-path` | (none) | Model checkpoint or HuggingFace ID |
| `--task` | `"pick and place"` | Task description |
| `--replicas` | 1 | Beaker replicas (Ray nodes) |
| `--gpus` | auto | GPUs per replica (2 for basic, 3 for TOPReward) |
| `--cluster` | `ai2/ceres-cirrascale` | Beaker cluster |
| `--budget` | (none) | Beaker budget account |
| `--priority` | `normal` | Job priority |
| `--dry-run` | off | Print command without executing |

### `scripts/start_robot_server.sh`

Launches the gRPC robot server and optional reverse SSH tunnel.

```bash
# Local testing (no tunnel, dummy robot)
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam.yaml \
    --dummy

# With reverse SSH tunnel to Beaker container
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam.yaml \
    --remote-host 100.87.5.72 \
    --dummy
```

**What the script does:**

1. Starts `python -m rlinf.envs.remote.robot_server` in background
2. If `--remote-host` is given, opens a reverse SSH tunnel:
   `ssh -N -R 50051:localhost:50051 shiruic@<host>`
3. Waits for both processes; cleans up on SIGINT/SIGTERM

| Option | Default | Description |
|---|---|---|
| `--config` | (required) | Path to YAM env YAML config |
| `--port` | `50051` | gRPC server port |
| `--remote-host` | (none) | Beaker container Tailscale IP |
| `--remote-user` | `shiruic` | SSH user on the container |
| `--dummy` | off | Zero observations, no real hardware |

### `ray_utils/start_ray_beaker.sh`

Beaker replica entrypoint for Ray cluster setup. Each replica runs this script;
it detects its role from `BEAKER_REPLICA_RANK`:

- **Rank 0 (head):** `ray start --head`, then runs the training command
- **Rank 1+ (workers):** Resolves head hostname, `ray start --address=<head>:6379`,
  then blocks until the cluster shuts down

Also usable as a standalone submission script (without `submit_yam_training.sh`).

## Beaker Secrets

The following secrets must exist in the Beaker workspace:

| Secret Name | Purpose |
|---|---|
| `hf_token_shirui` | HuggingFace token for model downloads |
| `tailscale_authkey_shirui` | Tailscale auth key for container VPN setup |

Create them with:

```bash
beaker secret write hf_token_shirui "hf_..."
beaker secret write tailscale_authkey_shirui "tskey-auth-..."
```

Generate a Tailscale auth key at: Tailscale admin console > Settings > Keys >
Generate auth key. Use a **reusable** key if running multiple jobs.

## End-to-End Workflow

### Step 1: Submit the Beaker job

```bash
bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi \
    --model-path openvla/openvla-7b \
    --allow-dirty
```

### Step 2: Get the container's Tailscale IP

Watch the Beaker logs for:

```
=== Tailscale IP ===
100.a.b.c
==================
```

### Step 3: Start the robot server with reverse SSH tunnel

```bash
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam.yaml \
    --remote-host 100.a.b.c \
    --dummy   # remove for real hardware
```

### Step 4: Training runs

The `RemoteEnv` inside the container connects to `localhost:50051` (routed
through the SSH tunnel to the desktop's `RobotServer`). The training loop
proceeds:

```
Actor (GPU 0) ─── generates actions ───► RemoteEnv ─── gRPC ───► RobotServer
     ▲                                       │                        │
     │                                       │                    YAMEnv
     │                                       │                    (robot HW)
     └──── policy update ◄── rewards ◄───────┘                        │
                                             ◄─── observations ◄──────┘
```

## Troubleshooting

### Container can't install Tailscale

The install script requires `curl` and root access. Most Beaker images include
these. If not, bake Tailscale into a custom Beaker image.

### SSH tunnel won't connect

- Verify the container's Tailscale IP is reachable: `ping 100.a.b.c`
- Ensure `sshd` is running in the container (most Beaker images include it)
- Check that the SSH user (`shiruic`) exists in the container
- The container must have `--host-networking` enabled (set in the submit script)

### gRPC timeout errors

- Increase `grpc_timeout` in `remote_yam.yaml` (default: 30s)
- For long action chunks, timeout scales as `grpc_timeout * chunk_size`
- Check that the SSH tunnel is still alive (it uses keepalive pings every 30s)

### Ray workers can't find head

- Workers resolve `BEAKER_LEADER_REPLICA_HOSTNAME` via DNS — this can take up
  to 5 minutes for the first replica
- If DNS resolution fails after 5 minutes, check that `--host-networking` is
  enabled and replicas are in the same experiment

### Tailscale shows "connected" but SSH fails

- The container may need a few seconds after `tailscale up` before accepting
  connections
- Verify with `tailscale status` on both ends
- Userspace networking can be slower to establish routes — wait 5-10 seconds
  after `tailscale up`
