# YAM Training Quickstart

All Ray workers run on Beaker; the desktop runs only the gRPC `RobotServer`
exposed via a reverse SSH tunnel. This is the only working topology for standard
YAM experiments.

For network and infrastructure details, see [network_infrastructure](network_infrastructure.md).
For algorithm and implementation details, see [training_architecture](training_architecture.md).
For config-specific guides, see [yam_ppo_openpi](yam_ppo_openpi.md) and
[yam_ppo_openpi_topreward](yam_ppo_openpi_topreward.md).

## Prerequisites

- [ ] `autossh` installed on the desktop (`brew install autossh` on macOS,
  `sudo apt-get install autossh` on Ubuntu/Debian)
- [ ] Desktop has a Tailscale client connected to the AI2 network
- [ ] Beaker secrets written (see [Beaker Secrets](#beaker-secrets) below)
- [ ] Model checkpoint available (HuggingFace ID or local path; default: `thomas0829/folding_towel_pi05`)

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
    --model-path thomas0829/folding_towel_pi05 \
    --allow-dirty
```

Pass `--workspace <beaker-workspace>` if you want to submit outside the default
`ai2/molmo-act` workspace.

> **Interactive mode (optional):** To get a shell inside the container and drive
> training manually, pass `--interactive`. This creates a `beaker session` instead
> of a gantry job, runs the same Tailscale/install/Ray-head startup path as the
> idle-cluster workflow, then leaves you in an interactive shell you can attach to:
>
> ```bash
> bash scripts/submit_yam_training.sh \
>     --config yam_ppo_openpi \
>     --interactive --allow-dirty
> # Beaker prints a session ID; then from the cluster:
> beaker session attach <session-id>
> ```
>
> This path requires a default local SSH key such as `~/.ssh/id_ed25519`,
> because `beaker session create --remote` attaches via SSH.

> **Idle cluster mode (recommended for debugging):** Instead of submitting a job
> that runs training immediately, you can submit a job that starts Ray and idles.
> You then SSH into the container at any time and run (or re-run) the training
> script manually. This is better for iterative debugging because the cluster
> stays up between training attempts — no need to re-submit when a run fails or
> you want to tweak arguments.
>
> ```bash
> # 1. Submit the idle cluster job (starts Ray head, installs deps, then waits)
> bash scripts/submit_yam_beaker_cluster.sh \
>     --config yam_ppo_openpi \
>     --priority high \
>     --cluster ai2/jupiter \
>     --workspace ai2/molmoact-ablations \
>     --allow-dirty
>
> # 2. Watch Beaker logs for the Tailscale IP (same as Step 2 below)
>
> # 3. SSH into the container and run training manually
> ssh shiruic@beaker-0  # or ssh shiruic@<tailscale-ip>
> cd /weka/oe-training-default/shiruic/RLinf
> source .venv/bin/activate
> python examples/embodiment/train_embodied_agent_staged.py \
>     --config-name yam_ppo_openpi \
>     actor.model.model_path=thomas0829/folding_towel_pi05 \
>     rollout.model.model_path=thomas0829/folding_towel_pi05 \
>     'env.train.task_description=pick and place' \
>     'env.eval.task_description=pick and place'
> ```
>
> The robot server + SSH tunnel workflow (Steps 2–3 below) is the same — the
> container still uses `RemoteEnv` over `localhost:50051`. The only difference
> is that you drive the training command yourself instead of the job running it
> automatically.

### Step 2: Get the container's Tailscale IP

Watch the Beaker logs for:

```
=== Tailscale IP ===
100.a.b.c
==================
```

### Step 3: Start the robot server with persistent reverse SSH tunnel

```bash
# Real hardware — tunnel reconnects automatically when new Beaker jobs start
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam.yaml

# Dummy mode (no CAN bus / robot hardware needed — for pipeline testing)
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam.yaml \
    --dummy
```

The server stays running indefinitely. `autossh` reconnects the reverse tunnel
to each new Beaker job automatically (all jobs register `beaker-0`). You do not
need to restart the robot server between Beaker job submissions.

> **Note:** `autossh` must be installed on the desktop. The script prints
> install instructions if it is missing.

### Step 4: Training runs

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

## Supported Configs

| Config | Reward | Subtask Planning | Beaker Script |
|---|---|---|---|
| `yam_ppo_openpi` | TOPReward (dense, VLM-based) | no (`subtask_interval: 0`) | `submit_yam_training.sh` |
| `yam_ppo_openpi_topreward` | TOPReward (dense, VLM-based) | yes (`subtask_interval: 3`) | `submit_yam_training.sh` |

Both configs also work with `submit_yam_beaker_cluster.sh` (idle cluster mode) — the
script auto-detects the GPU count from the config name.

## Next Steps

- [Network & infrastructure details](network_infrastructure.md) — Tailscale
  setup, SSH tunnel mechanics, CAN bus, scripts reference, and troubleshooting
- [Training architecture](training_architecture.md) — data flow, tensor shapes,
  PPO/GAE internals, Hydra config reference, and implementation notes
- [YAM PPO + TOPReward config guide](yam_ppo_openpi.md)
- [YAM PPO + TOPReward + subtask planning guide](yam_ppo_openpi_topreward.md)
