# RLinf Network Infrastructure for Beaker Training

This document covers the network stack, SSH tunnel mechanics, CAN bus setup,
and scripts reference for YAM training on Beaker. For the end-to-end workflow
and prerequisites, see [quickstart](quickstart.md). For algorithm and config
details, see [training_architecture](training_architecture.md).

## Topology 1: Beaker-Driven (RemoteEnv)

```
    Robot Desktop                          Beaker Container
    (Tailscale IP: 100.x.y.z)             (Tailscale IP: 100.a.b.c)

    ┌──────────────┐     reverse           ┌──────────────────┐
    │ RobotServer  │◄── SSH tunnel ────────│ RemoteEnv        │
    │ (gRPC :50051)│   desktop initiates   │ (gRPC client)    │
    │              │   ssh -R to container │                  │
    │ YAMEnv       │                       │ Ray Cluster      │
    │  └ Robot HW  │                       │  ├ Actor  (GPU0) │
    │  └ Cameras   │                       │  ├ Rollout(GPU1) │
    └──────────────┘                       │  └ VLM    (GPU2) │
                                           └──────────────────┘
```

The desktop initiates an SSH connection to the container (desktop → container via
Tailscale), creating a reverse tunnel that exposes the desktop's gRPC port inside
the container at `localhost:50051`. All Ray workers run entirely on Beaker — the
desktop is never a Ray node.

> **Why the desktop cannot be a Ray node (confirmed):** Beaker containers run
> Tailscale in `--tun=userspace-networking` mode. In this mode `tailscaled`
> handles WireGuard in userspace — there is **no kernel TUN interface** and
> therefore **no kernel route** to the desktop's Tailscale IP (`100.x.x.x`).
> Interactive testing from a running Beaker container confirmed:
>
> ```
> socket.create_connection('100.117.148.69', 9999, timeout=5)  → FAILED (timed out)
> tailscale ping 100.117.148.69                                 → relay via DERP only
> ```
>
> Ray's GCS active health checks (Beaker → desktop) use plain TCP and fail
> immediately. After 5 consecutive failures (~30 s) the GCS marks the desktop
> node dead; the local raylet receives `RAYLET_MARKED_DEAD` and self-terminates.
> No amount of tuning `RAY_health_check_failure_threshold` or
> `RAY_health_check_timeout_ms` on the desktop side helps — those variables
> are read by the GCS (on Beaker), not the worker.
>
> The fix is `RAY_health_check_period_ms=3600000` set in the **Beaker
> entrypoint** (before `ray start`), which effectively disables active health
> checks. Ray then relies solely on heartbeats (desktop → Beaker), which do
> work. This is baked into `submit_yam_beaker_cluster.sh`.
>
> Even with health checks disabled, the Desktop-Driven topology is unreliable
> in practice because other Ray internal TCP calls (object store, GCS
> subscriptions) also originate from Beaker and reach the same routing
> dead-end. Use Topology 1 for all production experiments.

## Topology 2: Desktop-Driven (Direct YAMEnv)

> **Status: Non-functional.** Beaker's userspace Tailscale cannot make TCP
> connections to desktop Tailscale IPs. Ray GCS health checks from Beaker to
> the desktop always fail, causing the desktop raylet to be marked dead within
> ~30 s of joining. See the note in [Topology 1](#topology-1-beaker-driven-remoteenv)
> for the root-cause analysis. **Use Topology 1 for all YAM experiments.**

```
    Robot Desktop                          Beaker Container
    (Tailscale IP: 100.x.y.z)             (Tailscale IP: 100.a.b.c)

    ┌────────────────────────┐            ┌──────────────────┐
    │ join_beaker_cluster.sh │            │ Ray head (:6379) │
    │                        │◄── TCP ───►│                  │
    │ Ray worker (macOS)     │  desktop   │ Actor  (GPU 0)   │
    │  └─ Training script    │  initiates │ Rollout(GPU 1)   │
    │  └─ EnvWorker          │            │ VLM    (GPU 2)   │
    │       └─ YAMEnv        │            └──────────────────┘
    │           └─ Robot HW  │
    └────────────────────────┘
    ▲ Desktop → Beaker TCP: OK (desktop initiates, Tailscale proxies inbound SSH)
    ✗ Beaker → Desktop TCP: FAILS (no kernel route in userspace Tailscale)
```

The desktop → Beaker direction works but the reverse fails — Beaker's userspace
Tailscale has no kernel TUN interface and cannot route TCP to `100.x.x.x`. See
[Troubleshooting: Desktop node marked dead](#desktop-node-marked-dead-immediately-after-joining-ray-cluster-topology-2).

**Config:** `examples/embodiment/config/yam_ppo_openpi_desktop.yaml`
(`num_nodes: 2`, `env/yam` direct, env on desktop `node_rank: 1`). Created
as a reference but not validated end-to-end on the current AI2 Beaker setup.

> **Incompatible with current canonical configs.** `yam_ppo_openpi` and
> `yam_ppo_openpi_topreward` both use `env/remote_yam` (RemoteEnv) and
> `cluster.num_nodes: 1` — they are designed for Topology 1.
> For standard YAM experiments, use Topology 1 (`submit_yam_training.sh`).

## Components *(Topology 1: Beaker-Driven)*

**Robot Desktop** runs two processes (managed by `scripts/start_robot_server.sh`):

| Process | Purpose |
|---|---|
| **RobotServer** | gRPC server wrapping `YAMEnv` — drives the physical robot, streams observations |
| **autossh tunnel** | `autossh -M 0 -N -R 50051:localhost:50051 shiruic@beaker-0` — persistent reverse tunnel; auto-reconnects when a new Beaker job starts |

Before starting RobotServer, `start_robot_server.sh` resets all CAN interfaces
(`reset_all_can.sh`) to clear any state from a previous run. This is a no-op
when no CAN interfaces are present (dummy mode or non-robot machine).

**Beaker Container** runs the Ray-based training pipeline:

| Component | GPU | Role |
|---|---|---|
| **Actor** | GPU 0 | Policy training (FSDP) |
| **Rollout** | GPU 1 | Action inference (π₀.5 requires a dedicated GPU) |
| **VLM Planner** | GPU 2 | TOPReward scoring (both configs); subtask planning only when `subtask_interval > 0` |
| **RemoteEnv** | CPU | gRPC client connecting to `localhost:50051` (via the SSH tunnel) |

## Network Stack

### Tailscale

Every Beaker replica installs and starts Tailscale on boot:

```bash
# Add Tailscale APT repository and install
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.noarmor.gpg \
  -o /usr/share/keyrings/tailscale-archive-keyring.gpg
echo 'deb [signed-by=/usr/share/keyrings/tailscale-archive-keyring.gpg] \
  https://pkgs.tailscale.com/stable/ubuntu jammy main' \
  > /etc/apt/sources.list.d/tailscale.list
(apt-get update || true) && apt-get install -y tailscale

# Start daemon (background, output suppressed)
nohup tailscaled --tun=userspace-networking --state=mem: > /dev/null 2>&1 &

# Join Tailscale network
tailscale up --authkey=${TAILSCALE_AUTHKEY} \
  --hostname=beaker-${BEAKER_REPLICA_RANK:-0} \
  --accept-routes

# Print IP to logs
echo '=== Tailscale IP ===' && tailscale ip -4 && echo '=================='
TAILSCALE_NODE_IP=$(tailscale ip -4)
```

- `--tun=userspace-networking` — required for unprivileged containers (no
  `/dev/net/tun`). In this mode, `tailscaled` handles all WireGuard traffic
  in userspace. The Tailscale IP is not assigned to any kernel network interface,
  but inbound SSH connections to the Tailscale IP are proxied by `tailscaled` to
  the local `sshd`.
- `--accept-routes` — **required** for the desktop to reach the container.
  The AI2 Tailscale network has subnet routes advertised by a subnet router
  covering the Beaker cluster. Without this flag the container does not install
  those routes and the desktop cannot reach it.
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

The container can't reach the desktop directly, but the desktop can reach the
container. By initiating the connection from the desktop and requesting a reverse
port forward, traffic flows bidirectionally through the single SSH connection.

### gRPC Protocol

Communication between `RemoteEnv` (client) and `RobotServer` (server) uses
gRPC with Protocol Buffers. The proto definition lives at
`rlinf/envs/remote/proto/robot_env.proto`.

For `remote_yam.yaml` Hydra config details (`grpc_timeout`, `auto_reset`,
`task_description`, etc.), see [training_architecture: Hydra Config](training_architecture.md#hydra-config-remote_yam).

**RPCs:**

| RPC | Direction | Purpose |
|---|---|---|
| `GetSpaces` | client → server | Fetch observation/action space metadata |
| `Reset` | client → server | Reset environment, return initial observation |
| `ChunkStep` | client → server | Send action chunk, receive step results |
| `SetTaskDescription` | client → server | Sync task string to server. Called once at `RemoteEnv.__init__` with the training-config task description, and again by the VLM subtask planner each time it generates a new subtask. Client always uses its locally-tracked `self._task_description` for obs regardless of what the server proto returns. |
| `Close` | client → server | Graceful shutdown |

**Observation encoding:**

- **Joint states**: raw `float32` bytes with shape metadata
- **Camera images**: JPEG-compressed (quality 90) to reduce bandwidth
  (~10x smaller than raw); decoded client-side via OpenCV
- **Max message size**: 16 MB (configurable via `grpc_max_message_size`)
- **Timeout**: 30s per RPC (configurable via `grpc_timeout`)

## CAN Bus (Robot Hardware Interface)

`YAMEnv` communicates with the robot arms over [SocketCAN](https://www.kernel.org/doc/html/latest/networking/can.html). CAN interfaces must be configured before `RobotServer` starts.

### Hardware

- YAM bimanual robot: two arms, each with 7 DOF (DM4340 / DM4310 motors)
- Two CAN channels: `can_left` and `can_right` (USB-to-CAN adapters)
- Control rate: 250 Hz (hardware); training control rate: 10 Hz (configurable)

### Interface Setup

`start_robot_server.sh` calls `reset_all_can.sh` automatically before launching
the server. To reset manually:

```bash
bash YAM/yam_realtime/yam_realtime/scripts/reset_all_can.sh
```

The script auto-detects all `can*` interfaces and brings each one down then up
at 1 Mbps:

```bash
ip link set canX down
ip link set canX up type can bitrate 1000000
```

> **When to reset:** Reset before every `RobotServer` start to clear stale motor
> state. If a previous run crashed without cleanly releasing the CAN bus, the
> motors may ignore commands until the interface is reset.

### Dummy Mode

Pass `--dummy` to `start_robot_server.sh` to run without robot hardware.
`YAMEnv` skips CAN initialization and returns zeroed observations. Useful for
debugging the full training pipeline (Beaker submission, SSH tunnel, gRPC,
training loop) without a physical robot.

```bash
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam.yaml \
    --no-tunnel --dummy
```

The training config does not need any changes for dummy mode — `is_dummy` is a
server-side setting that `RemoteEnv` does not read.

## Ray Cluster on Beaker

### Single-Replica (default)

All components run in one Beaker replica. Ray head starts on the same node.

`yam_ppo_openpi` (3 GPUs — TOPReward scoring, no subtask planning):
```
Replica 0 (head)
  ├── Ray head  (:6379)
  ├── Actor     (GPU 0 — FSDP training)
  ├── Rollout   (GPU 1 — inference)
  ├── VLM       (GPU 2 — Qwen3-VL-8B, TOPReward only)
  └── Env       (CPU, gRPC → localhost:50051)
```

`yam_ppo_openpi_topreward` (3 GPUs — TOPReward + subtask planning):
```
Replica 0 (head)
  ├── Ray head  (:6379)
  ├── Actor     (GPU 0 — FSDP training)
  ├── Rollout   (GPU 1 — inference)
  ├── VLM       (GPU 2 — Qwen3-VL-8B, TOPReward + subtask planning)
  └── Env       (CPU, gRPC → localhost:50051)
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

## Scripts Reference

### `scripts/submit_yam_training.sh`

Submits a Beaker training job via gantry, or creates an interactive Beaker
session for manual debugging (with `--interactive`).

Use `--workspace <beaker-workspace>` to override the default `ai2/molmo-act`
workspace.

```bash
# TOPReward only, no subtask planning (3 GPUs: actor GPU 0, rollout GPU 1, VLM GPU 2)
bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi \
    --model-path /path/to/pi05_checkpoint \
    --dry-run

# TOPReward + subtask planning (3 GPUs: actor GPU 0, rollout GPU 1, VLM GPU 2)
bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi_topreward \
    --model-path /path/to/pi05_checkpoint \
    --dry-run

# With Hydra overrides
bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi \
    --model-path /weka/.../checkpoint \
    -- algorithm.update_epoch=2 runner.save_interval=50

# Interactive session: full setup, then drop into a shell (no training)
bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi \
    --interactive --allow-dirty
# Beaker prints a session ID; then from the cluster:
# beaker session attach <session-id>
```

**What the script does (training mode):**

1. Auto-detects config type (`yam_ppo_openpi` exact / `*topreward*` / `*staged*` → 3 GPUs + `train_embodied_agent_staged.py`)
2. Builds Hydra training command (placement is baked into config defaults for single replica)
3. Base64-encodes the training command (avoids nested shell quoting issues)
4. Builds entrypoint that installs Tailscale, starts Ray, runs training
5. Submits via `gantry run` with the correct Beaker image, secrets, and mounts

**What the script does (interactive mode):**

1. Auto-detects GPU count from config (same as training mode)
2. Builds the same Beaker-side startup flow as `submit_yam_beaker_cluster.sh`: Tailscale → Tailscale-IP alias on `lo` when available → install → Ray head startup
3. Uses the shared `ray_utils/start_ray_beaker.sh --entrypoint` path, but starts an interactive shell on the head node instead of idling forever
4. Submits via `beaker session create --remote --bare`
5. User attaches with `beaker session attach <session-id>` and drives training manually
6. The local machine running `beaker session create` must have a default SSH key (for example `~/.ssh/id_ed25519`) so Beaker can attach

**Key options:**

| Option | Default | Description |
|---|---|---|
| `--config` | `yam_ppo_openpi` | Hydra config name |
| `--model-path` | (none) | Model checkpoint / HF ID (training mode only) |
| `--task` | `"pick and place"` | Task description (training mode only) |
| `--name` | `rlinf-<config>` | Beaker experiment/session name (appends `-interactive` in interactive mode) |
| `--replicas` | 1 | Beaker replicas / Ray nodes (training mode only) |
| `--gpus` | auto | GPUs per replica (3 for all YAM configs) |
| `--cluster` | `ai2/ceres-cirrascale` | Beaker cluster |
| `--budget` | (none) | Beaker budget account |
| `--priority` | `urgent` | Job priority |
| `--interactive` | off | Create an interactive Beaker session instead of submitting a training job |
| `--show-logs` | off | Stream Beaker logs after submission (training mode only) |
| `--allow-dirty` | off | Allow dirty git working directory (training mode only) |
| `--dry-run` | off | Print command without executing |

### `scripts/start_robot_server.sh`

Launches the gRPC robot server and a **persistent** reverse SSH tunnel on the
desktop. The robot server can run indefinitely — `autossh` automatically
reconnects the tunnel whenever a new Beaker job starts.

```bash
# Local testing only (no tunnel, dummy robot — no hardware needed)
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam.yaml \
    --no-tunnel --dummy

# Persistent server + auto-reconnecting tunnel (default beaker-0 hostname)
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam.yaml

# With explicit Tailscale IP instead of hostname (e.g. one-off debugging)
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam.yaml \
    --remote-host 100.a.b.c
```

**What the script does:**

1. Resets all CAN interfaces (`YAM/yam_realtime/.../reset_all_can.sh`) — safe
   no-op when no CAN interfaces are present (dummy mode or non-robot machine)
2. Starts `python -m rlinf.envs.remote.robot_server --config-path <config> --port <port>` in background
3. Unless `--no-tunnel` is set, starts an `autossh` reverse tunnel:
   ```bash
   autossh -M 0 -N \
       -R <port>:localhost:<port> \
       -o ServerAliveInterval=10 \
       -o ServerAliveCountMax=3 \
       -o ExitOnForwardFailure=yes \
       -o StrictHostKeyChecking=no \
       -o ConnectTimeout=10 \
       <user>@<remote-host>
   ```
   `autossh -M 0` relies on SSH keepalives (not its own monitoring socket).
   When a Beaker job dies and a new one starts, `autossh` reconnects to
   `beaker-0` (the fixed hostname all Beaker jobs register) automatically.
4. Waits for all processes; cleans up on SIGINT/SIGTERM (`kill 0`)

**Persistent operation across Beaker jobs:**

Because all Beaker Tailscale nodes register the hostname `beaker-<replica-rank>`,
a new Beaker job always comes up as `beaker-0`. `autossh` will reconnect the
tunnel to the new job's container without any manual intervention. The robot
server never needs to restart. `RemoteEnv`'s `grpc_connect_timeout: 300 s`
(5-minute retry window) provides the window for the tunnel to reconnect after
job submission.

| Option | Default | Description |
|---|---|---|
| `--config` | (required) | Path to YAM env YAML config |
| `--port` | `50051` | gRPC server port |
| `--remote-host` | `beaker-0` | Beaker Tailscale hostname or IP |
| `--remote-user` | `shiruic` | SSH user on the container |
| `--no-tunnel` | off | Start RobotServer only, no SSH tunnel |
| `--dummy` | off | Zero observations, no real hardware |

> **`autossh` prerequisite.** Must be installed on the desktop:
> `sudo apt-get install autossh` (Ubuntu/Debian) or `brew install autossh` (macOS).
> The script checks for `autossh` and prints install instructions if it is missing.

### `scripts/submit_yam_beaker_cluster.sh`

*(Desktop-Driven topology — Topology 2; see [status note](#topology-2-desktop-driven-direct-yamenv))*

Submits a Beaker job that starts Ray head with GPUs and **idles** — no training
command. The desktop then joins and drives training via `join_beaker_cluster.sh`.

```bash
# TOPReward only, no subtask planning (3 GPUs, idle, waiting for desktop)
bash scripts/submit_yam_beaker_cluster.sh \
    --config yam_ppo_openpi \
    --dry-run
```

**What the script does:**

1. Auto-detects GPU count from config (`yam_ppo_openpi` exact / `*topreward*` / `*staged*` → 3, else 2)
2. Installs Tailscale in the container (same as `submit_yam_training.sh`)
3. Attempts `ip addr add <tailscale-ip>/32 dev lo` to make Ray advertise the
   Tailscale IP; falls back gracefully if `CAP_NET_ADMIN` is absent
4. **Exports `RAY_health_check_period_ms=3600000`** before `ray start` —
   disables GCS active health checks (see Topology 1 networking note). Without
   this, the desktop raylet is marked dead within 30 s of joining
5. Calls `start_ray_beaker.sh --entrypoint` with no `--train-cmd` → Ray head
   starts and blocks indefinitely (no training loop)

| Option | Default | Description |
|---|---|---|
| `--config` | `yam_ppo_openpi` | Config for GPU auto-detection |
| `--gpus` | auto | GPUs (3 for all canonical YAM configs) |
| `--name` | `rlinf-cluster-<config>` | Beaker experiment name |
| `--cluster` | `ai2/ceres-cirrascale` | Beaker cluster |
| `--budget` | (none) | Beaker budget account |
| `--priority` | `urgent` | Job priority |
| `--show-logs` | off | Stream Beaker logs after submission |
| `--allow-dirty` | off | Allow dirty git working directory |
| `--dry-run` | off | Print command without executing |

### `scripts/join_beaker_cluster.sh`

*(Desktop-Driven topology — Topology 2)*

Joins the idle Beaker Ray cluster from the local desktop and runs training.
The env worker runs with direct `YAMEnv` — no gRPC.

```bash
# TOPReward, no subtask planning (desktop at node rank 1 in custom multi-node config)
bash scripts/join_beaker_cluster.sh \
    --head-ip 100.64.1.2 \
    --config my_custom_yam_config \
    --model-path /path/to/pi05_checkpoint \
    --task "pick and place"

# TOPReward + subtask planning (custom multi-node config)
bash scripts/join_beaker_cluster.sh \
    --head-ip 100.64.1.2 \
    --config my_custom_yam_topreward_config \
    --node-rank 0 \
    --model-path /path/to/pi05_checkpoint \
    --task "bimanual manipulation"
```

**What the script does:**

1. TCP-checks `<head-ip>:<ray-port>` (fails fast if container unreachable)
2. Detects desktop Tailscale IP (`tailscale ip -4`) and passes it to
   `ray start --node-ip-address=<tailscale-ip>` so the raylet binds to the
   correct interface; falls back to auto-detect if Tailscale is unavailable
3. `ray start --address=<head-ip>:6379` with retries (up to 30 × 10 s)
4. Waits 15 s and checks that the local raylet socket (`/tmp/ray/session_latest/sockets/raylet`)
   still exists — if it is gone, the raylet self-terminated (likely due to GCS
   health check failure; see Topology 2 status note) and the script exits with an error
5. Activates `.venv` if present; installs deps if needed
6. Runs the Hydra training command on the desktop
7. Cleans up (`ray stop --force`) on exit

| Option | Default | Description |
|---|---|---|
| `--head-ip` | (required) | Beaker container Tailscale IP |
| `--config` | `yam_ppo_openpi` | Hydra config name |
| `--model-path` | (none) | Model checkpoint path |
| `--task` | `"pick and place"` | Task description |
| `--node-rank` | `1` | This desktop's `RLINF_NODE_RANK` |
| `--ray-port` | `6379` | Ray head port |

> **macOS desktops:** `RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1` is set
> automatically. Ray workers on macOS cannot use `fork` for actor processes;
> this env var opts in to a spawn-based workaround.

### `ray_utils/start_ray_beaker.sh`

Beaker replica entrypoint for Ray cluster setup. Each replica runs this script;
it detects its role from `BEAKER_REPLICA_RANK`:

- **Rank 0 (head):** `ray start --head`, then runs the training command
- **Rank 1+ (workers):** Resolves head hostname, `ray start --address=<head>:6379`,
  then blocks until the cluster shuts down

Also usable as a standalone submission script (without `submit_yam_training.sh`).

## Beaker Secrets

> Beaker secrets (`hf_token_shirui`, `tailscale_authkey_shirui`) must be written
> to the workspace before submitting any job. See
> [quickstart: Beaker Secrets](quickstart.md#beaker-secrets) for setup instructions.

## Troubleshooting

### Container can't install Tailscale

The install script requires `curl`, `apt-get`, and root access. Most Beaker
images (Ubuntu-based) include these. If not, bake Tailscale into a custom
Beaker image.

### Desktop can't reach the Beaker container (ping/SSH fails)

The most common cause: `--accept-routes` was not passed to `tailscale up` in the
container. The AI2 Tailscale network advertises subnet routes, and without
`--accept-routes` the container doesn't install them, making it unreachable even
when both sides are on the same Tailscale account.

Check the Beaker logs for the `tailscale up` line and confirm `--accept-routes`
is present. `submit_yam_training.sh` includes it by default.

### SSH tunnel won't connect

- Verify the container's Tailscale IP is reachable: `ping 100.a.b.c`
- Ensure `sshd` is running in the container (most Beaker images include it)
- Check that the SSH user (`shiruic`) exists in the container
- The container must have `--host-networking` enabled (set in the submit script)
- Beaker containers have a fresh SSH host key each run. `start_robot_server.sh`
  passes `-o StrictHostKeyChecking=no` to handle this automatically. If you are
  running SSH manually, add that flag or the connection will hang waiting for
  keyboard input.

### gRPC timeout errors

- Increase `grpc_timeout` in `remote_yam.yaml` (default: 30s)
- For long action chunks, timeout scales as `grpc_timeout * chunk_size`
- Check that the SSH tunnel is still alive (it uses keepalive pings every 30s)

### Ray workers can't find head

- Workers resolve `BEAKER_LEADER_REPLICA_HOSTNAME` via DNS — this can take up
  to 5 minutes for the first replica
- If DNS resolution fails after 5 minutes, check that `--host-networking` is
  enabled and replicas are in the same experiment

### Desktop node marked dead immediately after joining Ray cluster (Topology 2)

Symptom in raylet logs (`/tmp/ray/session_latest/logs/raylet.out`):

```
GCS failed to check the health of this node for 5 times.
RAYLET_MARKED_DEAD
```

**Root cause:** Beaker's userspace Tailscale cannot route TCP connections to
desktop Tailscale IPs (`100.x.x.x`). Ray GCS active health checks (Beaker →
desktop) all fail; after 5 failures the node is marked dead.

**Fixes applied:**

1. `submit_yam_beaker_cluster.sh` now exports `RAY_health_check_period_ms=3600000`
   before `ray start` — this disables active health checks (period of 1 hour
   means they never fire in practice). Must be set on the Beaker side (GCS
   reads it); setting it only on the desktop has no effect.
2. `join_beaker_cluster.sh` now passes `--node-ip-address=<tailscale-ip>` to
   `ray start` so the raylet binds to the correct interface.
3. `join_beaker_cluster.sh` waits 15 s after joining and checks the raylet
   socket — exits with a clear error message if the raylet is already dead.

Even with these fixes, other Ray internal TCP calls originating from Beaker may
still fail. **Use Topology 1 (`submit_yam_training.sh`) for all production runs.**

### Stale node registrations in GCS (Topology 2)

Symptom: `ray status` shows more alive nodes than expected after multiple failed
`join_beaker_cluster.sh` runs.

**Cause:** Each failed run leaves a stale node registration in the Beaker-side
GCS. The GCS does not garbage-collect dead nodes immediately.

**Fix:** Re-submit the Beaker job (new job = fresh Ray cluster state). Do not
reuse the same `--head-ip` after the cluster has accumulated stale registrations.

### Tailscale shows "connected" but SSH fails

- The container may need a few seconds after `tailscale up` before accepting
  connections
- Verify with `tailscale status` on both ends
- Userspace networking can be slower to establish routes — wait 5-10 seconds
  after `tailscale up`
