# RLinf Network Infrastructure for Beaker Training

This document describes the network architecture for running embodied RL training
on AI2 Beaker GPU clusters while the physical robot remains on a local desktop.

Two topologies are supported:

- **Beaker-Driven (RemoteEnv)** — all Ray workers run on Beaker; the desktop
  runs only a gRPC `RobotServer` exposed via reverse SSH tunnel. Most reliable:
  no special container capabilities required. Use `submit_yam_training.sh`.

- **Desktop-Driven (Direct YAMEnv)** — Beaker provides GPUs for actor/rollout
  workers; the desktop joins the Ray cluster via Tailscale and runs the env
  worker (direct `YAMEnv`) and the training script. Simpler robot access but
  requires the Beaker container to advertise its Tailscale IP (needs
  `CAP_NET_ADMIN` or equivalent). Use `submit_yam_beaker_cluster.sh` +
  `join_beaker_cluster.sh`.

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

> **Why the desktop cannot be a Ray node in this topology:** Beaker containers
> run Tailscale in `--tun=userspace-networking` mode (no `CAP_NET_ADMIN` to
> create a TUN device). In this mode the container has no kernel route to the
> desktop's Tailscale IP, so Ray's bidirectional TCP (head ↔ worker node
> manager, object store, and worker process ports) cannot be established. Only
> a single outbound SSH connection — which the desktop initiates and which
> Tailscale's internal stack can proxy — is reliable.
>
> The [Desktop-Driven topology](#topology-2-desktop-driven-direct-yamenv)
> avoids this constraint by running the Ray head on Beaker and having the
> *desktop* initiate the connection outbound. That approach works when the
> Beaker container can advertise its Tailscale IP to Ray (requires
> `CAP_NET_ADMIN`); if the container lacks that capability, fall back to this
> topology.

## Topology 2: Desktop-Driven (Direct YAMEnv)

In this topology the desktop joins the Beaker Ray cluster as a worker node,
runs the training script locally, and drives `EnvWorker` directly via
`YAMEnv` — no gRPC, no SSH tunnel. Beaker provides only GPU capacity for
actor and rollout workers.

```
    Robot Desktop                          Beaker Container
    (Tailscale IP: 100.x.y.z)             (Tailscale IP: 100.a.b.c)

    ┌────────────────────────┐            ┌──────────────────┐
    │ join_beaker_cluster.sh │            │ Ray head (:6379) │
    │                        │◄── TCP ───►│                  │
    │ Ray worker (macOS)     │  Tailscale │ Actor  (GPU 0)   │
    │  └─ Training script    │            │ Rollout(GPU 1)   │
    │  └─ EnvWorker          │            │ VLM    (GPU 2)   │
    │       └─ YAMEnv        │            └──────────────────┘
    │           └─ Robot HW  │
    └────────────────────────┘
```

**Workflow:**

1. `submit_yam_beaker_cluster.sh` submits a Beaker job that installs Tailscale,
   starts Ray head, and idles — no training command is sent.
2. Watch the Beaker logs for the Tailscale IP banner (`=== Tailscale IP ===`).
3. `join_beaker_cluster.sh --head-ip <ip>` on the desktop: TCP-checks the head,
   joins as a Ray worker (`ray start --address=...`), then runs the training
   script on the desktop.

**Constraint — Tailscale IP advertisement:**

`submit_yam_beaker_cluster.sh` attempts `ip addr add <tailscale-ip>/32 dev lo`
so that Ray advertises the Tailscale IP for its backplane. This requires
`CAP_NET_ADMIN` in the Beaker container. Without it the container advertises
only its internal IP, which the desktop cannot reach, and `ray start` on the
desktop will time out. If this happens, use Topology 1 instead.

> **Incompatible with current canonical configs.** `yam_ppo_openpi` and
> `yam_ppo_openpi_topreward` both use `env/remote_yam` (RemoteEnv) and
> `cluster.num_nodes: 1` — they are designed for Topology 1.
> Topology 2 requires a custom config with `env/yam` (direct `YAMEnv`) and a
> multi-node cluster layout. For standard YAM experiments, use Topology 1
> (`submit_yam_training.sh`).

## Components *(Topology 1: Beaker-Driven)*

**Robot Desktop** runs two processes (managed by `scripts/start_robot_server.sh`):

| Process | Purpose |
|---|---|
| **RobotServer** | gRPC server wrapping `YAMEnv` — drives the physical robot, streams observations |
| **Reverse SSH tunnel** | `ssh -R 50051:localhost:50051 shiruic@<container-tailscale-ip>` — makes the gRPC server reachable at `localhost:50051` inside the container |

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

## Training Architecture

### Data Flow

The `EmbodiedRunner` spawns `EnvWorker.interact()` and `RolloutWorker.generate()` as **concurrent Ray tasks**. They synchronise through channels — neither calls the other directly.

```
Desktop                              Beaker Container
──────                               ─────────────────────────────────────────────────
                                     Runner spawns concurrently ↓
RobotServer                         ┌───────────────────┐     ┌─────────────────────┐
(wraps YAMEnv)                      │    EnvWorker      │     │   RolloutWorker     │
    │                               │  (RemoteEnv)      │◄────┤  ← actions          │
    │◄── gRPC (via SSH tunnel) ────►│                   │     │  (π₀.5 OpenPI)      │
    │                               │  calls VLMPlanner:│────►│  obs + reward →     │
    │                               │  · get_next_subtask     └──────────┬──────────┘
    │                               │  · compute_top_reward              │ trajectories
    │                               └───────────┬───────┘                ▼
    │                                           │              ┌─────────────────────┐
    │                                           │              │    ActorWorker      │
    │                               ┌───────────▼───────┐      │  (FSDP training)    │
    │                               │  VLMPlannerWorker │      │  PPO loss           │
    │                               │  (Qwen3-VL-8B)    │      └──────────┬──────────┘
    │                               └───────────────────┘                 │ sync weights
    │                                                                      └──► RolloutWorker
```

**Per-epoch loop (concurrent handoff via channels):**

```
For each epoch (rollout_epoch total):
EnvWorker                                   RolloutWorker
─────────                                   ─────────────
bootstrap_step() → send_env_batch() ──────►                           ─╮
                                            recv_env_output()            │
                                            predict() → send_chunk_actions()  │ ×n_train
recv_chunk_actions() ◄──────────────────────                            │  _chunk
env_interact_step()                                                      │  _steps
  └─ RemoteEnv.chunk_step() → gRPC → YAMEnv                            │
  └─ _compute_top_reward() [VLMPlanner, ~200-400 ms]                   │
  └─ if done: _reset_top_reward_state() [_prev_top_score=0.0]          │
_maybe_update_subtask()  [VLMPlanner if subtask_interval > 0]           │
  └─ if new_subtask non-empty: _reset_top_reward_state() [_prev_top_score=0.0] │
send_env_batch() ─────────────────────────►                           ─╯
                                            [post-loop, once per epoch]
                                            recv_env_output() [final obs+reward+done]
                                            predict() [GAE value bootstrap + last reward;
                                                       no actions sent]
After rollout_epoch epochs:               send_rollout_trajectories() ──► ActorWorker
```

`n_train_chunk_steps = max_steps_per_rollout_epoch // num_action_chunks`
(e.g. `100 // 10 = 10` for YAM with `max_steps_per_rollout_epoch: 100`, `num_action_chunks: 10`).

The post-loop `recv_env_output` (in `generate_one_epoch`) receives the last env step result
(observation + reward + done from the 10th `env_interact_step`). It serves two purposes:
(1) collects `prev_values = V(obs₁₀)` for GAE bootstrapping (the `n_steps+1` value estimate),
and (2) captures the last step's reward and done signal to complete the 10-entry reward window.
No action is sent back to the EnvWorker for this step (`prev_logprobs=None`, `actions=None`,
`forward_inputs=None`). The `prev_values` entry is collected because
`rollout.collect_prev_infos: true` (required for `adv_type: gae`).

Resulting trajectory shapes after `to_trajectory()` (for `n_train_chunk_steps=10`, `bsz=B`, `num_action_chunks=C`):

| Field | Shape | Contents |
|---|---|---|
| `rewards` | `[10, B, C]` | Loop iter 0 (bootstrap) has `rewards=None` so is skipped; iters 1–9 + post-loop provide 10 entries. Each entry is `[B, C]` (one reward scalar per sub-step). |
| `prev_values` | `[11, B, 1]` | V(obs₀)…V(obs₁₀): all 10 loop iters + post-loop. Value head output is per-observation, shape `[B, 1]`. |
| `dones` | `[11, B, C]` | dones₀ (bootstrap initial) + dones₁…dones₁₀. Shape `[B, C]` per entry (one done flag per sub-step). |
| `forward_inputs` | `dict[str, Tensor]` each `[10, B, ...]` | `stack_list_of_dict_tensor` converts the `EmbodiedRolloutResult.forward_inputs` (a list of 10 per-step dicts) into a stacked dict. Keys include `chains`, `denoise_inds`, `tokenized_prompt`, `tokenized_prompt_mask` + cloned observation tensors — the diffusion rollout state needed to recompute logprobs during the PPO update. `action` (raw action values) are **not** stored for standard PPO (`forward_action=None` in OpenPI non-DSRL mode; the key `"action"` is absent from each step's dict). |

For YAM (`C=10`, `B=1`): `rewards=[10,1,10]`, `prev_values=[11,1,1]`, `dones=[11,1,10]`.
`preprocess_embodied_advantages_inputs` with `reward_type="chunk_level"` then reduces the C dimension: `rewards.sum(-1,keepdim=True)→[10,1,1]`, `dones.max(-1,keepdim=True)→[11,1,1]`.

> **`bootstrap_type: always` note:** `bootstrap_type` controls whether
> `get_dones_and_rewards()` adds a discounted bootstrap value to the reward on truncated episodes
> (i.e. treats all dones as truncations rather than only true truncations). For YAM this branch
> is never entered because `env.train.auto_reset: false` — the condition in
> `get_dones_and_rewards` is `if last_step_truncations.any() and auto_reset`. The per-step value
> estimate for GAE is always provided by the post-loop `predict()` call regardless of
> `bootstrap_type`.

After `rollout_epoch` epochs, ActorWorker computes advantages (GAE), runs policy update epochs, and syncs updated weights to RolloutWorker at the start of the next training step.

### Supported Configs

| Config | Algorithm | Policy | Reward | Subtask Planning | Beaker Script |
|---|---|---|---|---|---|
| `yam_ppo_openpi` | PPO + GAE | π₀.5 (OpenPI, diffusion) | TOPReward (dense, VLM-based) | no (`subtask_interval: 0`) | `submit_yam_training.sh` |
| `yam_ppo_openpi_topreward` | PPO + GAE | π₀.5 (OpenPI, diffusion) | TOPReward (dense, VLM-based) | yes (`subtask_interval: 3`) | `submit_yam_training.sh` |

Both configs use TOPReward (Qwen3-VL-8B on GPU 2) and `group_size: 1`.
The only difference is `subtask_interval`: `yam_ppo_openpi` uses the VLM for reward
scoring only; `yam_ppo_openpi_topreward` also generates language subtask descriptions
that are injected into the policy's language conditioning.

> **`collect_prev_infos: true` required for GAE.** Both configs use `adv_type: gae`,
> which requires the value estimates (`prev_values`) collected by the rollout worker
> to be present in the trajectory batch for `preprocess_embodied_advantages_inputs`.
> Both configs explicitly set `rollout.collect_prev_infos: true`. Setting it to `false`
> with `adv_type: gae` does NOT crash — `EmbodiedRolloutResult.append_step_result` guards
> with `if result.prev_values is not None:`, so the list stays empty and
> `Trajectory.prev_values` remains `None`. In `compute_gae_advantages_and_returns`,
> `values=None` triggers the `critic_free` fallback (`gae_lambda=1`, `gamma=1`),
> silently degrading to plain REINFORCE without a value baseline. Training continues
> but the GAE advantage signal is lost.

> **`bootstrap_step()` resets the robot every training step.** For `auto_reset: false`
> (the YAM configs), `bootstrap_step()` calls `env.reset()` on every rollout epoch.
> Since `rollout_epoch: 1`, this means the robot is reset (gRPC `Reset` → `YAMEnv.reset()`)
> at the start of every training step. Each episode is a fresh rollout. The
> `store_last_obs_and_intervened_info()` call (after the chunk loop) stores the final
> observation for use only in `auto_reset: true` configs — it is a no-op for YAM.

> **`subtask_interval` unit:** chunk steps (not env steps). The `EnvWorker` resets
> the subtask counter at `bootstrap_step()` (once per rollout epoch / episode
> reset). With `max_steps_per_rollout_epoch: 100` and `num_action_chunks: 10`,
> `n_train_chunk_steps = 10`. Therefore `subtask_interval` must be ≤ 10 to fire
> within an episode. `subtask_interval: 3` → VLM updates the subtask at chunk
> steps 3, 6, 9 (env steps ~30, ~60, ~90 — roughly 30% / 60% / 90% of the episode).
> Setting `subtask_interval > n_train_chunk_steps` disables subtask planning
> (the config validator warns at startup; the counter resets before reaching the
> threshold so the planner never fires).

### Code Component Reference

Quick mapping from architecture terms to code locations:

| Term | Code component | Location |
|---|---|---|
| **Diffusion NFT** | RL algorithm for flow-matching / diffusion policies (π₀.5). The current YAM configs run standard PPO on the OpenPI model — Diffusion NFT is a planned upgrade. | `yam_ppo_openpi*.yaml`, `rlinf/models/embodiment/openpi/` — TODO(agent): not yet implemented |
| **VLM planner** | `VLMPlannerWorker` (Qwen3-VL-8B) | `rlinf/workers/vlm_planner/vlm_planner_worker.py` |
| **TOPReward** | `compute_top_reward()` — log P("True" \| frames, instruction) | Same file, called from `rlinf/workers/env/env_worker.py` |
| **Frame buffer** | Episode frame buffer `_episode_frames` in `EnvWorker` | `rlinf/workers/env/env_worker.py` — NOT a standalone Ray actor; frames are buffered in-process before each TOPReward call |
| **Rollout worker** | `MultiStepRolloutWorker` | `rlinf/workers/rollout/hf/huggingface_worker.py` |
| **Actor / Train** | `EmbodiedFSDPActor` | `rlinf/workers/actor/fsdp_actor_worker.py` |
| **YAMEnv / Robot server** | `YAMEnv` wrapped by `RobotServer` | `rlinf/envs/yam/yam_env.py`, `rlinf/envs/remote/robot_server.py` |

### Implementation Notes

#### VLMPlannerWorker GPU placement

Actor and rollout workers bypass Ray's GPU resource pool (they set `CUDA_VISIBLE_DEVICES` manually). Ray therefore sees all node GPUs as unclaimed. `_launch_vlm_planner` in `train_embodied_agent_staged.py` uses `_compute_vlm_gpu_index(cfg)` to determine the correct GPU:

1. If `vlm_planner.placement` is set explicitly in the config, use that.
2. Otherwise, collect distinct placement indices used by actor/rollout/env on the same physical node as `beaker_vlm`. If two or more distinct indices exist, return `max(indices) + 1`. Both YAM configs have actor=0, rollout=1 (two distinct indices), so the heuristic gives VLM GPU = 2 for both. ✓

#### Action dimension for YAM bimanual

YAM is a 14-DOF bimanual robot (2 × 7 joints). Both configs set `actor.model.action_dim: 14`,
which propagates to `openpi.action_env_dim` via Hydra interpolation. The OpenPI model generates
actions up to its internal `action_dim` and then slices to `action_env_dim` — without this
override the template default of 7 would silently truncate actions to single-arm size.

#### TOPReward reward baseline and episode resets

`_prev_top_score` (the running log-probability baseline for delta computation) is reset to `0.0` in three places:

1. **Epoch boundary** — `bootstrap_step()` calls `_reset_top_reward_state()` at the start of every rollout epoch (`env_worker.py:628–629`).
2. **Episode done** — `env_interact_step()` calls `_reset_top_reward_state()` whenever `chunk_dones[:, -1].any()` (`env_worker.py:434–435`).
3. **Subtask change** — `_maybe_update_subtask()` calls `_reset_top_reward_state()` whenever the VLM generates a new subtask and `top_reward_enabled` is True (`env_worker.py:219–220`). Without this reset, the first delta after a subtask change would mix log-probs from different instructions (`score_new_subtask(t+1) − score_old_subtask(t)`), which are not comparable.

The episode-done and subtask-change resets both clear `_episode_frames` as well, giving the VLM a clean context window for each new episode / subtask phase.

#### Subtask planner image context

`_maybe_update_subtask()` reads `env.last_obs` to supply the VLM subtask planner with the most recent camera frame. `RemoteEnv` maintains `self.last_obs` and updates it on every `reset()` and `chunk_step()` call. If `last_obs` is `None` (before the first step) or the env wrapper doesn't expose the attribute, `_maybe_update_subtask()` falls back to text-only subtask generation using only the memory buffer — the planner still produces a subtask but without visual context.

`gym.Wrapper.__getattr__` delegates non-private attribute reads to the inner env, so `getattr(env, "last_obs", None)` propagates transparently through `RecordVideo` and `CollectEpisode` wrappers.

For attribute **writes**, `gym.Wrapper` does NOT delegate — `wrapper.attr = value` creates an instance attribute on the wrapper and bypasses the inner env's property setter. `_maybe_update_subtask()` therefore uses `env.unwrapped` to reach `RemoteEnv` directly when calling `inner_env.task_description = new_subtask` (which triggers the `SetTaskDescription` gRPC call). `_compute_top_reward()` likewise reads instruction from `env.unwrapped`.

Note: `last_obs` (single latest frame for subtask planning) is distinct from `_episode_frames` (accumulated frame buffer for TOPReward scoring).

#### Subtask interval sizing

`_steps_since_subtask_update` is an instance variable reset to `0` in `bootstrap_step()` (once per rollout epoch). With `rollout_epoch: 1`, this reset happens once per training step. The effective maximum subtask interval within a single episode is therefore `n_train_chunk_steps = max_steps_per_rollout_epoch // num_action_chunks`.

For the YAM configs (`max_steps_per_rollout_epoch: 100`, `num_action_chunks: 10`): `n_train_chunk_steps = 10`. If `subtask_interval > 10` the subtask planner never fires because the counter is reset before it reaches the threshold. The correct value for 3 subtask updates per episode is `subtask_interval: 3` (chunk steps 3, 6, 9 = env steps 30, 60, 90).

#### TOPReward VLM latency

`compute_top_reward()` is called **synchronously** in the rollout loop — each chunk step blocks on Qwen3-VL-8B inference (~200–400 ms). The `_episode_frames` buffer in `EnvWorker` is an in-process list, not a standalone Ray actor. This is a known limitation; decoupling it for async reward scoring is a future improvement.

#### TOPReward requires the `transformers` backend

`VLMPlannerWorker.compute_top_reward()` requires `vlm_planner.backend: "transformers"` — it performs a **forward pass** to extract log-probabilities, not a generation call. When `backend: "sglang"`, `compute_top_reward()` logs a warning and returns `0.0`. Both YAM configs set `backend: "transformers"`. If you switch to `sglang` for faster subtask generation, TOPReward will yield zero rewards every step (warning logged, but training continues without crashing).

Similarly, `compute_top_reward()` returns `0.0` on any exception (network error, OOM, etc.) with only a warning log — training continues but reward signal is lost for that step.

#### `reward_scale` configuration path

`TOPReward` reads `reward_scale` from the **`vlm_planner`** config section (since `VLMPlannerWorker` passes `planner_cfg` to `TOPReward.__init__`), **not** from the `reward` section. The `reward` section with `use_reward_model: False` is metadata only — no separate reward worker is instantiated for TOPReward. To change the scale, set `vlm_planner.reward_scale` in the YAML. Both YAM configs now include `vlm_planner.reward_scale: 1.0` explicitly.

#### `global_batch_size` / `micro_batch_size` constraint

`EmbodiedFSDPActor.run_training` asserts:

```
rollout_size % (actor.global_batch_size // world_size) == 0
```

where `rollout_size = n_train_chunk_steps × total_num_envs × rollout_epoch`.

For YAM with `max_steps_per_rollout_epoch=100`, `num_action_chunks=10`,
`total_num_envs=1`, `rollout_epoch=1`:

```
n_train_chunk_steps = 100 // 10 = 10
rollout_size = 10 × 1 × 1 = 10
```

So `global_batch_size` must be a divisor of 10 (e.g. 1, 2, 5, 10) and
`micro_batch_size` must divide `global_batch_size`. Both YAM configs now use
`global_batch_size: 10` and `micro_batch_size: 10`.

If you scale up (e.g. 4 envs, 2 rollout epochs → `rollout_size = 80`), update
`global_batch_size` accordingly. The config validator in `rlinf/config.py` will
warn at startup if `global_batch_size` does not divide `rollout_size`.

#### Entropy loss mask alignment

`EmbodiedFSDPActor.run_training` applies an entropy bonus: `loss -= entropy_bonus * entropy_loss`. The entropy for OpenPI with `entropy_type: chunk_level` is collapsed to shape `[bsz]` (one scalar per chunk step) by `reshape_entropy`.

For the YAM configs (`ignore_terminations: True`), `loss_mask` is `None` — the loss-mask block in `_process_received_rollout_batch` is gated by `not auto_reset AND not ignore_terminations`, which is `False` for YAM. `masked_mean(entropy, mask=None)` correctly falls back to `.mean()`.

For configs where `ignore_terminations=False` and `auto_reset=False`, `loss_mask` is computed with `reward_type: chunk_level` any-reduction and ends up with shape `[bsz, 1]`. In that case, `masked_mean(entropy=[bsz], mask=[bsz, 1])` broadcasts incorrectly — PyTorch aligns `[bsz]` as `[1, bsz]` against `[bsz, 1]`, producing an outer product `[bsz, bsz]` and computing the **sum** instead of the **mean**.

The fix reshapes `loss_mask` to `entropy.shape` before calling `masked_mean`, which handles both cases correctly (no-op when `mask=None`, safe reshape when `mask=[bsz, 1]` and `entropy=[bsz]`).

#### `kl_beta` / `kl_penalty` are ignored for embodied tasks

`EmbodiedFSDPActor.run_training` does **not** compute a KL penalty term. The `kl_beta: 0.0` and `kl_penalty: kl` keys in the YAM configs are present for configuration consistency (they are unused fields; the config validator does not require them) but have no effect during training. KL penalty is only applied in `FSDPActor.run_training` (the reasoning-task actor).

#### YAMEnv base reward is always zero

`YAMEnv.step()` always returns `reward = np.zeros(num_envs)` and `terminated = np.zeros(num_envs, bool)`. There is no task-success signal wired from the robot hardware — success detection is not implemented at the environment level. The training reward comes **entirely from TOPReward** (delta log-prob injected by `_compute_top_reward`). Episodes end only via time-limit truncation (`_elapsed_steps >= max_episode_steps`).

As a result, the `success_once` field in `episode_info` will always be `False` for YAM training — this is expected behavior, not a bug. The policy's only learning signal is the TOPReward progress score.

The base rewards transmitted over gRPC (from `RobotEnvServicer.ChunkStep`) are also zero; TOPReward is computed and injected on the client (`EnvWorker._compute_top_reward`), **after** the gRPC call returns.

#### Multi-replica (REPLICAS > 1)

Single-replica (`--replicas 1`, the default) is fully tested. For `REPLICAS > 1`, `submit_yam_training.sh` adds placement range overrides, but multi-replica has not been validated end-to-end. Use `--replicas 1` for real-hardware experiments.

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

## Configuration

### Hydra Config: `remote_yam`

File: `examples/embodiment/config/env/remote_yam.yaml`

```yaml
env_type: remote
remote_server_url: "${oc.env:ROBOT_SERVER_URL,localhost:50051}"
grpc_max_message_size: 16777216  # 16 MB
grpc_timeout: 30.0               # seconds per RPC; scaled by chunk_size for ChunkStep

# Base task description — always overridden by the training config.
# e.g. yam_ppo_openpi sets this to "bimanual pick and place".
# RemoteEnv.__init__ calls SetTaskDescription gRPC with this value at startup
# so the robot server's YAMEnv starts with the correct instruction.
task_description: ""

# These config values take precedence over what the server returns via GetSpaces.
# RemoteEnv.__init__ overrides the server-reported auto_reset/ignore_terminations
# with the values set here (cfg.get("auto_reset", spaces.auto_reset)).
auto_reset: false
ignore_terminations: true

# compress_images / jpeg_quality are server-side settings — put them in the
# yam.yaml passed to start_robot_server.sh, not here. RemoteEnv handles both
# compressed and uncompressed images transparently.
# max_episode_steps / control_rate_hz are fetched from the server at init via
# GetSpaces() and are not read from this file.

video_cfg:
  save_video: false
  info_on_video: true
  video_base_dir: ${runner.logger.log_path}/video/train
```

> **`update_reset_state_ids()` interface.** `EnvWorker.finish_rollout()` calls
> `env.update_reset_state_ids()` after each rollout epoch to let vectorised envs
> (e.g. Libero, ManiSkill) rotate task indices. `RemoteEnv` and `YAMEnv` implement
> this as a no-op since single-instance real-robot envs have no state IDs to cycle.
> `finish_rollout` also guards with `hasattr` to prevent crashes for any env that
> doesn't implement the method.

> **`is_dummy` is a server-side setting.** `RemoteEnv` does not read `is_dummy`
> from the training config — it proxies all calls over gRPC. To test without
> real hardware, start the robot server with `--dummy`:
> ```bash
> bash scripts/start_robot_server.sh --config .../yam.yaml --dummy
> ```
> The training config requires no change for dummy mode.

Both YAM configs declare `remote_yam` as the env type for train and eval
via Hydra `defaults` (baked into the YAML, not passed as CLI overrides):

```yaml
defaults:
  - env/remote_yam@env.train
  - env/remote_yam@env.eval
```

### Supported Training Configs (Beaker-submittable via `submit_yam_training.sh`)

| Config | Entry Point | GPUs | Components |
|---|---|---|---|
| `yam_ppo_openpi` | `train_embodied_agent_staged.py` | 3 | actor (GPU 0), rollout (GPU 1), VLM planner (GPU 2, TOPReward only), env (RemoteEnv, CPU) |
| `yam_ppo_openpi_topreward` | `train_embodied_agent_staged.py` | 3 | actor (GPU 0), rollout (GPU 1), VLM planner (GPU 2, TOPReward + subtask), env (RemoteEnv, CPU) |

## Scripts Reference

### `scripts/submit_yam_training.sh`

Submits a Beaker training job via gantry.

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
```

**What the script does:**

1. Auto-detects config type (`yam_ppo_openpi` exact / `*topreward*` / `*staged*` → 3 GPUs + `train_embodied_agent_staged.py`)
2. Builds Hydra training command (placement is baked into config defaults for single replica)
3. Base64-encodes the training command (avoids nested shell quoting issues)
4. Builds entrypoint that installs Tailscale, starts Ray, runs training
5. Submits via `gantry run` with the correct Beaker image, secrets, and mounts

**Key options:**

| Option | Default | Description |
|---|---|---|
| `--config` | `yam_ppo_openpi` | Hydra config name |
| `--model-path` | (none) | Model checkpoint path |
| `--task` | `"pick and place"` | Task description |
| `--name` | `rlinf-<config>` | Beaker experiment name |
| `--replicas` | 1 | Beaker replicas (Ray nodes) |
| `--gpus` | auto | GPUs per replica (3 for all YAM configs) |
| `--cluster` | `ai2/ceres-cirrascale` | Beaker cluster |
| `--budget` | (none) | Beaker budget account |
| `--priority` | `urgent` | Job priority |
| `--show-logs` | off | Stream Beaker logs after submission |
| `--allow-dirty` | off | Allow dirty git working directory |
| `--dry-run` | off | Print command without executing |

### `scripts/start_robot_server.sh`

Launches the gRPC robot server and reverse SSH tunnel on the desktop.

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

1. Starts `python -m rlinf.envs.remote.robot_server --config-path <config> --port <port>` in background
2. If `--remote-host` is given, opens a reverse SSH tunnel with keepalive and `ExitOnForwardFailure`:
   `ssh -N -R <port>:localhost:<port> -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=no <user>@<host>`
3. Prints: `>>> On the Beaker container, set: export ROBOT_SERVER_URL=localhost:<port> <<<`
   (only needed when using a non-default port; the default in `remote_yam.yaml` is already `localhost:50051`)
4. Waits for both processes; cleans up on SIGINT/SIGTERM (`kill 0`)

| Option | Default | Description |
|---|---|---|
| `--config` | (required) | Path to YAM env YAML config |
| `--port` | `50051` | gRPC server port |
| `--remote-host` | (none) | Beaker container Tailscale IP |
| `--remote-user` | `shiruic` | SSH user on the container |
| `--dummy` | off | Zero observations, no real hardware |

### `scripts/submit_yam_beaker_cluster.sh`

*(Desktop-Driven topology — Topology 2)*

Submits a Beaker job that starts Ray head with GPUs and **idles** — no training
command. The desktop then joins and drives training via `join_beaker_cluster.sh`.

```bash
# TOPReward only, no subtask planning (3 GPUs, idle, waiting for desktop)
bash scripts/submit_yam_beaker_cluster.sh \
    --config yam_ppo_openpi \
    --dry-run

# TOPReward + subtask planning (3 GPUs, idle, waiting for desktop)
bash scripts/submit_yam_beaker_cluster.sh \
    --config yam_ppo_openpi_topreward \
    --dry-run
```

**What the script does:**

1. Auto-detects GPU count from config (`yam_ppo_openpi` exact / `*topreward*` / `*staged*` → 3, else 2)
2. Installs Tailscale in the container (same as `submit_yam_training.sh`)
3. Attempts `ip addr add <tailscale-ip>/32 dev lo` to make Ray advertise the
   Tailscale IP; falls back gracefully if `CAP_NET_ADMIN` is absent
4. Calls `start_ray_beaker.sh --entrypoint` with no `--train-cmd` → Ray head
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
2. `ray start --address=<head-ip>:6379` with retries (up to 30 × 10 s)
3. Activates `.venv` if present; installs deps if needed
4. Runs the Hydra training command on the desktop
5. Cleans up (`ray stop --force`) on exit

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

## End-to-End Workflow *(Topology 1: Beaker-Driven)*

#### Step 1: Submit the Beaker job

```bash
bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi \
    --model-path /path/to/pi05_checkpoint \
    --allow-dirty
```

#### Step 2: Get the container's Tailscale IP

Watch the Beaker logs for:

```
=== Tailscale IP ===
100.a.b.c
==================
```

#### Step 3: Start the robot server with reverse SSH tunnel

```bash
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam.yaml \
    --remote-host 100.a.b.c \
    --dummy   # remove for real hardware
```

#### Step 4: Training runs

The `RemoteEnv` inside the container connects to `localhost:50051` (routed
through the SSH tunnel to the desktop's `RobotServer`). Actor runs on GPU 0,
Rollout on GPU 1, VLMPlannerWorker on GPU 2. The training loop proceeds:

```
Rollout (GPU 1) ─── generates actions ──────► RemoteEnv ─── gRPC ───► RobotServer
     ▲                                             │                        │
     │ updated weights                             │                    YAMEnv
     │                                             │                    (robot HW)
Actor (GPU 0) ◄──── trajectories + rewards ◄──────┘                        │
     └──── updates weights ─────────────────────► Rollout ◄─ observations ─┘

VLMPlanner (GPU 2) ◄── frames + instruction ── EnvWorker ──────────────────┘
     │   (TOPReward delta injected into rewards; subtasks injected if interval > 0)
     └──────────────────────────────────────────────────────────────────────────►
```

> **Reward note:** Both YAM configs use TOPReward (Qwen3-VL-8B on GPU 2) —
> no custom reward code required. The only difference is `subtask_interval`:
> `yam_ppo_openpi` scores reward only; `yam_ppo_openpi_topreward` also
> generates VLM subtask descriptions injected into the policy's language conditioning.

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

### Tailscale shows "connected" but SSH fails

- The container may need a few seconds after `tailscale up` before accepting
  connections
- Verify with `tailscale status` on both ends
- Userspace networking can be slower to establish routes — wait 5-10 seconds
  after `tailscale up`
