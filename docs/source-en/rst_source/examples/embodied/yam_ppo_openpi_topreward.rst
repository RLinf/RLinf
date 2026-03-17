Real-World PPO on YAM with π₀.5 + TOPReward
============================================

This guide walks through running **PPO** with the **π₀.5 (OpenPI)** policy,
**TOPReward** dense reward, and **VLM subtask planning** on the **YAM bimanual
robot**, using a single Beaker GPU node and a desktop robot controller.

Both YAM configs use TOPReward.  This config additionally enables
**subtask planning** (``subtask_interval: 30``): the VLM periodically generates
language subtask descriptions that are injected into the policy's language
conditioning, replacing the fixed ``task_description`` for the remainder of
that interval.  Use :doc:`yam_ppo_openpi` if you want TOPReward without
subtask planning (``subtask_interval: 0``).

Overview
--------

**TOPReward + Subtask Planning (Qwen3-VL-8B)**

The ``VLMPlannerWorker`` serves two roles in this config:

1. **TOPReward scoring** — dense reward at every chunk step.
   ``r_t = log P("True" | frames_{1:t}, instruction) − log P("True" | frames_{1:t-1}, instruction)``
2. **Subtask planning** (``subtask_interval: 30``) — every 30 chunk steps the
   VLM generates a new language subtask description that replaces the
   policy's language conditioning for the next interval.  Without this
   (``subtask_interval: 0``) the policy sees only the user-given
   ``task_description`` for the full episode.

**Algorithm: PPO with GAE**

Same as :doc:`yam_ppo_openpi` — PPO with GAE advantage estimation and a
value head on π₀.5 VLM features.

Hardware Topology
-----------------

Single Beaker node (3 GPUs) + desktop robot controller:

.. code-block:: text

    Robot Desktop                        Beaker Container
    (Tailscale: 100.x.y.z)             (Tailscale: 100.a.b.c)

    ┌──────────────┐  reverse SSH       ┌────────────────────────────┐
    │ RobotServer  │◄─ tunnel ─────────│ RemoteEnv (CPU)            │
    │ (gRPC :50051)│                   │                            │
    │ YAMEnv       │                   │ Actor      (GPU 0, FSDP)   │
    │  └ Robot HW  │                   │ Rollout    (GPU 1, infer)  │
    └──────────────┘                   │ VLMPlanner (GPU 2, reward) │
                                       └────────────────────────────┘

The VLMPlannerWorker GPU is assigned automatically: since actor uses GPU 0 and
rollout uses GPU 1 (two distinct placements), ``_compute_vlm_gpu_index`` returns
``max(0, 1) + 1 = 2``.

Environment
-----------

- **Robot**: YAM bimanual arm (2 × 7 DOF)
- **Observation**: joint states (14-dim) + RGB image (224 × 224)
- **Action space**: 14-dim joint-position targets
- **Control rate**: 10 Hz
- **Reward**: TOPReward — log-prob progress signal from Qwen3-VL-8B.
  No custom reward code required.

Dependency Installation
-----------------------

**Desktop (robot controller)**

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   bash requirements/install.sh embodied --env yam
   source .venv/bin/activate

**Additional packages for VLM** (on Beaker, handled automatically by
``submit_yam_training.sh``):

.. code-block:: bash

   pip install "transformers @ git+https://github.com/huggingface/transformers.git@main"
   pip install qwen-vl-utils accelerate torchvision pillow

Model Download
--------------

Download π₀.5 SFT checkpoint and Qwen3-VL-8B:

.. code-block:: bash

   pip install huggingface-hub
   hf download RLinf/RLinf-Pi05-SFT --local-dir /path/to/RLinf-Pi05-SFT
   hf download Qwen/Qwen3-VL-8B-Instruct --local-dir /path/to/Qwen3-VL-8B-Instruct

Configuration
-------------

The ready-to-use config is at
``examples/embodiment/config/yam_ppo_openpi_topreward.yaml``.

**1. Model paths**

.. code-block:: yaml

   rollout:
     model:
       model_path: "/path/to/RLinf-Pi05-SFT"

   actor:
     model:
       model_path: "/path/to/RLinf-Pi05-SFT"

   vlm_planner:
     model_path: "Qwen/Qwen3-VL-8B-Instruct"  # or local path

**2. Task description**

.. code-block:: yaml

   env:
     train:
       task_description: "pick up the red block and place it in the bowl"

**3. TOPReward tuning (optional)**

.. code-block:: yaml

   env:
     train:
       top_reward_enabled: True
       top_reward_max_frames: 16    # frames per VLM call
       subtask_interval: 30         # steps between subtask replanning (0 to disable)

   vlm_planner:
     max_new_tokens_subtask: 64
     max_new_tokens_reward: 16
     success_threshold: 0.5

For initial testing without real hardware, start the robot server with ``--dummy``
(see :ref:`Dry-Run in Dummy Mode <dry-run-in-dummy-mode>` below).

Submitting to Beaker
--------------------

.. code-block:: bash

   bash scripts/submit_yam_training.sh \
       --config yam_ppo_openpi_topreward \
       --model-path /path/to/RLinf-Pi05-SFT

After the job starts:

1. Watch the Beaker logs for ``=== Tailscale IP ===``.
2. Start the robot server on the desktop:

   .. code-block:: bash

      bash scripts/start_robot_server.sh \
          --config examples/embodiment/config/env/yam.yaml \
          --remote-host <tailscale-ip> \
          --dummy   # remove for real hardware

Key Algorithm Parameters
------------------------

.. code-block:: yaml

   algorithm:
     adv_type: gae
     loss_type: actor_critic
     group_size: 1
     update_epoch: 5
     bootstrap_type: always
     entropy_bonus: 0.005
     clip_ratio_high: 0.2
     gamma: 0.99
     gae_lambda: 0.95

.. code-block:: yaml

   actor:
     model:
       action_dim: 14           # YAM bimanual: 2 × 7 DOF (sets openpi.action_env_dim)

.. code-block:: yaml

   reward:
     use_reward_model: False
     reward_type: top_reward   # TOPReward via VLMPlannerWorker
     reward_scale: 1.0

Dry-Run in Dummy Mode
---------------------

Dummy mode is controlled at the **robot server**, not in the training config.
``RemoteEnv`` proxies all calls over gRPC and does not read ``is_dummy``.

Start the robot server with ``--dummy`` to return zero observations:

.. code-block:: bash

   bash scripts/start_robot_server.sh \
       --config examples/embodiment/config/env/yam.yaml \
       --dummy

Then submit the training job (no config change needed):

.. code-block:: bash

   bash scripts/submit_yam_training.sh \
       --config yam_ppo_openpi_topreward \
       --model-path /path/to/RLinf-Pi05-SFT \
       --dry-run

Interactive Session
-------------------

Pass ``--interactive`` to create a `beaker session`_ instead of a gantry
training job.  The container runs full setup automatically (Tailscale, deps,
Ray head, model download), then drops into an interactive shell you can attach
to from the cluster.

.. code-block:: bash

   # Submit the interactive session
   bash scripts/submit_yam_training.sh \
       --config yam_ppo_openpi_topreward \
       --interactive --allow-dirty

   # Beaker prints a session ID.  Attach from the cluster:
   beaker session attach <session-id>

Inside the shell, the venv is activated and Ray is already running.  Start
training manually:

.. code-block:: bash

   python examples/embodiment/train_embodied_agent_staged.py \
       --config-name yam_ppo_openpi_topreward \
       actor.model.model_path=thomas0829/folding_towel_pi05 \
       rollout.model.model_path=thomas0829/folding_towel_pi05

Pass ``--model-path`` to pre-download a different checkpoint (default:
``thomas0829/folding_towel_pi05``).

.. _beaker session: https://beaker-py.readthedocs.io/

Visualisation and Results
--------------------------

**TensorBoard**

.. code-block:: bash

   tensorboard --logdir ../results --port 6006

**Key metrics**

- ``env/success_once``: Fraction of episodes with at least one success.
- ``env/return``: Cumulative episode reward (TOPReward deltas).
- ``train/actor/grad_norm``: Policy gradient norm.
- ``vlm/top_reward``: Mean TOPReward score per step.
- ``vlm/subtask``: Current VLM-generated subtask description (if planning
  enabled).

Troubleshooting
---------------

**VLM OOM on GPU 2**

Reduce ``vlm_planner.dtype`` from ``bfloat16`` to ``float16``, or use a smaller
model (``Qwen/Qwen3-VL-3B``).

**Slow training (VLM blocks rollout)**

``compute_top_reward()`` is synchronous — each chunk step waits for Qwen3-VL-8B
inference (~200–400 ms).  Reduce ``top_reward_max_frames`` to speed up scoring.

**gRPC timeout during VLM scoring**

Increase ``grpc_timeout`` in ``remote_yam.yaml`` if the VLM call exceeds the
per-RPC timeout.

**VLM planner not launching**

Verify ``env.train.top_reward_enabled: True`` is set and the config is using
``train_embodied_agent_staged.py`` as the entry point (auto-selected by
``run_realworld.sh`` for ``*topreward*`` configs).

See Also
--------

- :doc:`yam_ppo_openpi` — Same pipeline with TOPReward but without subtask
  planning (``subtask_interval: 0``)
- :doc:`franka` — Real-world RL with a Franka arm
