# YAM Training Architecture

This document covers the data flow, tensor shapes, configuration reference, and
implementation notes for the YAM PPO training pipeline. For setup and running a
first training job, see [quickstart](quickstart.md). For network and
infrastructure details, see [network_infrastructure](network_infrastructure.md).

## Supported Configs

| Config | Algorithm | Policy | Reward | Subtask Planning | Entry Point | GPUs | Beaker Script |
|---|---|---|---|---|---|---|---|
| `yam_ppo_openpi` | PPO + GAE | π₀.5 (OpenPI, diffusion) | TOPReward (dense, VLM-based) | no (`subtask_interval: 0`) | `train_embodied_agent_staged.py` | 3 | `submit_yam_training.sh` |
| `yam_ppo_openpi_topreward` | PPO + GAE | π₀.5 (OpenPI, diffusion) | TOPReward (dense, VLM-based) | yes (`subtask_interval: 3`) | `train_embodied_agent_staged.py` | 3 | `submit_yam_training.sh` |

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
> but the GAE advantage signal is lost. See [Troubleshooting: Training signal silently degraded](#training-signal-silently-degraded-gae--reinforce-fallback).

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

## Data Flow

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

## Per-Epoch Loop

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

## Code Component Reference

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

## Implementation Notes

### VLMPlannerWorker GPU placement

Actor and rollout workers bypass Ray's GPU resource pool (they set `CUDA_VISIBLE_DEVICES` manually). Ray therefore sees all node GPUs as unclaimed. `_launch_vlm_planner` in `train_embodied_agent_staged.py` uses `_compute_vlm_gpu_index(cfg)` to determine the correct GPU:

1. If `vlm_planner.placement` is set explicitly in the config, use that.
2. Otherwise, collect distinct placement indices used by actor/rollout/env on the same physical node as `beaker_vlm`. If two or more distinct indices exist, return `max(indices) + 1`. Both YAM configs have actor=0, rollout=1 (two distinct indices), so the heuristic gives VLM GPU = 2 for both. ✓

### Action dimension for YAM bimanual

YAM is a 14-DOF bimanual robot (2 × 7 joints). Both configs set `actor.model.action_dim: 14`,
which propagates to `openpi.action_env_dim` via Hydra interpolation. The OpenPI model generates
actions up to its internal `action_dim` and then slices to `action_env_dim` — without this
override the template default of 7 would silently truncate actions to single-arm size.

### TOPReward reward baseline and episode resets

`_prev_top_score` (the running log-probability baseline for delta computation) is reset to `0.0` in three places:

1. **Epoch boundary** — `bootstrap_step()` calls `_reset_top_reward_state()` at the start of every rollout epoch (`env_worker.py:628–629`).
2. **Episode done** — `env_interact_step()` calls `_reset_top_reward_state()` whenever `chunk_dones[:, -1].any()` (`env_worker.py:434–435`).
3. **Subtask change** — `_maybe_update_subtask()` calls `_reset_top_reward_state()` whenever the VLM generates a new subtask and `top_reward_enabled` is True (`env_worker.py:219–220`). Without this reset, the first delta after a subtask change would mix log-probs from different instructions (`score_new_subtask(t+1) − score_old_subtask(t)`), which are not comparable.

The episode-done and subtask-change resets both clear `_episode_frames` as well, giving the VLM a clean context window for each new episode / subtask phase.

### Subtask planner image context

`_maybe_update_subtask()` reads `env.last_obs` to supply the VLM subtask planner with the most recent camera frame. `RemoteEnv` maintains `self.last_obs` and updates it on every `reset()` and `chunk_step()` call. If `last_obs` is `None` (before the first step) or the env wrapper doesn't expose the attribute, `_maybe_update_subtask()` falls back to text-only subtask generation using only the memory buffer — the planner still produces a subtask but without visual context.

`gym.Wrapper.__getattr__` delegates non-private attribute reads to the inner env, so `getattr(env, "last_obs", None)` propagates transparently through `RecordVideo` and `CollectEpisode` wrappers.

For attribute **writes**, `gym.Wrapper` does NOT delegate — `wrapper.attr = value` creates an instance attribute on the wrapper and bypasses the inner env's property setter. `_maybe_update_subtask()` therefore uses `env.unwrapped` to reach `RemoteEnv` directly when calling `inner_env.task_description = new_subtask` (which triggers the `SetTaskDescription` gRPC call). `_compute_top_reward()` likewise reads instruction from `env.unwrapped`.

Note: `last_obs` (single latest frame for subtask planning) is distinct from `_episode_frames` (accumulated frame buffer for TOPReward scoring).

### Subtask interval sizing

`_steps_since_subtask_update` is an instance variable reset to `0` in `bootstrap_step()` (once per rollout epoch). With `rollout_epoch: 1`, this reset happens once per training step. The effective maximum subtask interval within a single episode is therefore `n_train_chunk_steps = max_steps_per_rollout_epoch // num_action_chunks`.

For the YAM configs (`max_steps_per_rollout_epoch: 100`, `num_action_chunks: 10`): `n_train_chunk_steps = 10`. If `subtask_interval > 10` the subtask planner never fires because the counter is reset before it reaches the threshold. The correct value for 3 subtask updates per episode is `subtask_interval: 3` (chunk steps 3, 6, 9 = env steps 30, 60, 90).

### TOPReward VLM latency

`compute_top_reward()` is called **synchronously** in the rollout loop — each chunk step blocks on Qwen3-VL-8B inference (~200–400 ms). The `_episode_frames` buffer in `EnvWorker` is an in-process list, not a standalone Ray actor. This is a known limitation; decoupling it for async reward scoring is a future improvement.

### TOPReward requires the `transformers` backend

`VLMPlannerWorker.compute_top_reward()` requires `vlm_planner.backend: "transformers"` — it performs a **forward pass** to extract log-probabilities, not a generation call. When `backend: "sglang"`, `compute_top_reward()` logs a warning and returns `0.0`. Both YAM configs set `backend: "transformers"`. If you switch to `sglang` for faster subtask generation, TOPReward will yield zero rewards every step (warning logged, but training continues without crashing).

Similarly, `compute_top_reward()` returns `0.0` on any exception (network error, OOM, etc.) with only a warning log — training continues but reward signal is lost for that step.

### `reward_scale` configuration path

`TOPReward` reads `reward_scale` from the **`vlm_planner`** config section (since `VLMPlannerWorker` passes `planner_cfg` to `TOPReward.__init__`), **not** from the `reward` section. The `reward` section with `use_reward_model: False` is metadata only — no separate reward worker is instantiated for TOPReward. To change the scale, set `vlm_planner.reward_scale` in the YAML. Both YAM configs now include `vlm_planner.reward_scale: 1.0` explicitly.

### `global_batch_size` / `micro_batch_size` constraint

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

### Entropy loss mask alignment

`EmbodiedFSDPActor.run_training` applies an entropy bonus: `loss -= entropy_bonus * entropy_loss`. The entropy for OpenPI with `entropy_type: chunk_level` is collapsed to shape `[bsz]` (one scalar per chunk step) by `reshape_entropy`.

For the YAM configs (`ignore_terminations: True`), `loss_mask` is `None` — the loss-mask block in `_process_received_rollout_batch` is gated by `not auto_reset AND not ignore_terminations`, which is `False` for YAM. `masked_mean(entropy, mask=None)` correctly falls back to `.mean()`.

For configs where `ignore_terminations=False` and `auto_reset=False`, `loss_mask` is computed with `reward_type: chunk_level` any-reduction and ends up with shape `[bsz, 1]`. In that case, `masked_mean(entropy=[bsz], mask=[bsz, 1])` broadcasts incorrectly — PyTorch aligns `[bsz]` as `[1, bsz]` against `[bsz, 1]`, producing an outer product `[bsz, bsz]` and computing the **sum** instead of the **mean**.

The fix reshapes `loss_mask` to `entropy.shape` before calling `masked_mean`, which handles both cases correctly (no-op when `mask=None`, safe reshape when `mask=[bsz, 1]` and `entropy=[bsz]`).

### `kl_beta` / `kl_penalty` are ignored for embodied tasks

`EmbodiedFSDPActor.run_training` does **not** compute a KL penalty term. The `kl_beta: 0.0` and `kl_penalty: kl` keys in the YAM configs are present for configuration consistency (they are unused fields; the config validator does not require them) but have no effect during training. KL penalty is only applied in `FSDPActor.run_training` (the reasoning-task actor).

### YAMEnv base reward is always zero

`YAMEnv.step()` always returns `reward = np.zeros(num_envs)` and `terminated = np.zeros(num_envs, bool)`. There is no task-success signal wired from the robot hardware — success detection is not implemented at the environment level. The training reward comes **entirely from TOPReward** (delta log-prob injected by `_compute_top_reward`). Episodes end only via time-limit truncation (`_elapsed_steps >= max_episode_steps`).

As a result, the `success_once` field in `episode_info` will always be `False` for YAM training — this is expected behavior, not a bug. The policy's only learning signal is the TOPReward progress score.

The base rewards transmitted over gRPC (from `RobotEnvServicer.ChunkStep`) are also zero; TOPReward is computed and injected on the client (`EnvWorker._compute_top_reward`), **after** the gRPC call returns.

### Multi-replica (REPLICAS > 1)

Single-replica (`--replicas 1`, the default) is fully tested. For `REPLICAS > 1`, `submit_yam_training.sh` adds placement range overrides, but multi-replica has not been validated end-to-end. Use `--replicas 1` for real-hardware experiments.

## Hydra Config: `remote_yam`

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

## Troubleshooting

### Training signal silently degraded (GAE → REINFORCE fallback)

**Symptom:** Training runs without errors but the policy does not improve, or
improves much more slowly than expected.

**Root cause:** `rollout.collect_prev_infos` is `false` while `adv_type: gae` is
set. `EmbodiedRolloutResult.append_step_result` guards with
`if result.prev_values is not None:`, so the `prev_values` list stays empty and
`Trajectory.prev_values` remains `None`. In `compute_gae_advantages_and_returns`,
`values=None` triggers the `critic_free` fallback (`gae_lambda=1`, `gamma=1`),
silently degrading to plain REINFORCE without a value baseline.

**Fix:** Ensure `rollout.collect_prev_infos: true` in your YAML config whenever
`adv_type: gae` is set. Both canonical YAM configs already set this correctly.
