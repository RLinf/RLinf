Real-World PPO on YAM with π₀.5 (OpenPI)
=========================================

This guide walks through running **PPO** with the **π₀.5 (OpenPI)** policy on
the **YAM bimanual robot**, using a single Beaker GPU node and a desktop
robot controller.

Overview
--------

**Algorithm: PPO with GAE**

The training loop uses Proximal Policy Optimization with Generalized Advantage
Estimation (``adv_type: gae``).  A value head is added to the π₀.5 model
(``add_value_head: True``) and trained jointly with the policy.

**π₀.5 (OpenPI)**

π₀.5 is a diffusion-based robot manipulation policy.  It requires a dedicated
GPU for inference (rollout) and a separate GPU for training (actor) — they
cannot share the same device.

**YAM robot**

The YAM is a 7-DOF-per-arm bimanual platform.  Arms connect via **CAN bus**
(``DMChainCanInterface``).  Cameras attach as ``yam_realtime`` ``CameraNode``
objects.

**RemoteEnv (gRPC)**

The environment runs on the local desktop inside a ``RobotServer`` gRPC
service.  A reverse SSH tunnel exposes the service to the Beaker container
at ``localhost:50051``.  The ``RemoteEnv`` client inside the container connects
to that address and proxies observations and actions.

**TOPReward**

TOPReward provides a dense reward signal from Qwen3-VL-8B without requiring
a hand-crafted reward function.  At each chunk step the VLM scores task
progress as ``r_t = log P("True" | frames, instruction) − log P("True" | frames_{prev}, instruction)``.

Subtask planning is disabled here (``subtask_interval: 0``).  Without it the
policy's language conditioning is fixed to the user-given ``task_description``
for the entire episode (e.g. ``"pick up the red block"``).  To enable the VLM
to generate dynamic subtask instructions during rollout, use
:doc:`yam_ppo_openpi_topreward`.

Hardware Topology
-----------------

Single Beaker node (3 GPUs) + desktop robot controller:

.. code-block:: text

    Robot Desktop                          Beaker Container
    (Tailscale: 100.x.y.z)               (Tailscale: 100.a.b.c)

    ┌──────────────┐  reverse SSH tunnel  ┌──────────────────────────┐
    │ RobotServer  │◄─────────────────────│ RemoteEnv (CPU)          │
    │ (gRPC :50051)│                      │                          │
    │ YAMEnv       │                      │ Actor      (GPU 0, FSDP) │
    │  └ Robot HW  │                      │ Rollout    (GPU 1, infer) │
    └──────────────┘                      │ VLMPlanner (GPU 2, reward)│
                                          └──────────────────────────┘

Environment
-----------

- **Robot**: YAM bimanual arm (2 × 7 DOF)
- **Observation**: joint states (14-dim) + RGB image (224 × 224)
- **Action space**: 14-dim joint-position targets
- **Control rate**: 10 Hz (configurable via ``control_rate_hz``)
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

Install ``yam_realtime`` by following the instructions in your YAM hardware
package.

**Additional packages for VLM** (on Beaker, handled automatically by
``submit_yam_training.sh``):

.. code-block:: bash

   pip install "transformers @ git+https://github.com/huggingface/transformers.git@main"
   pip install qwen-vl-utils accelerate torchvision pillow

Model Download
--------------

Download the π₀.5 SFT checkpoint and Qwen3-VL-8B before submitting:

.. code-block:: bash

   pip install huggingface-hub
   hf download RLinf/RLinf-Pi05-SFT --local-dir /path/to/RLinf-Pi05-SFT
   hf download Qwen/Qwen3-VL-8B-Instruct --local-dir /path/to/Qwen3-VL-8B-Instruct

Configuration
-------------

The ready-to-use config is at
``examples/embodiment/config/yam_ppo_openpi.yaml``.

Before launching, set the model paths:

.. code-block:: yaml

   rollout:
     model:
       model_path: "/path/to/RLinf-Pi05-SFT"

   actor:
     model:
       model_path: "/path/to/RLinf-Pi05-SFT"

   vlm_planner:
     model_path: "Qwen/Qwen3-VL-8B-Instruct"  # or local path

And the task description:

.. code-block:: yaml

   env:
     train:
       task_description: "pick up the red block"

For initial testing without real hardware, start the robot server with ``--dummy``
(see :ref:`Dry-Run in Dummy Mode <dry-run-in-dummy-mode>` below).

Submitting to Beaker
--------------------

.. code-block:: bash

   bash scripts/submit_yam_training.sh \
       --config yam_ppo_openpi \
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
     adv_type: gae          # GAE advantage estimation
     loss_type: actor_critic # PPO actor-critic loss
     group_size: 1          # One episode per update step — correct for single robot
     update_epoch: 5        # Gradient passes per rollout
     bootstrap_type: always # Always bootstrap value at end of episode
     entropy_bonus: 0.005   # Entropy regularisation
     clip_ratio_high: 0.2
     gamma: 0.99
     gae_lambda: 0.95

.. code-block:: yaml

   actor:
     model:
       add_value_head: True           # PPO critic head on π₀.5 VLM features
       action_dim: 14                 # YAM bimanual: 2 × 7 DOF (sets action_env_dim)
       num_steps: 4                   # Flow matching denoising steps
       openpi:
         noise_method: "flow_noise"
         action_horizon: 10
         noise_params: [0.16, 0.12, 200]
         joint_logprob: True

.. code-block:: yaml

   reward:
     reward_type: top_reward   # TOPReward via VLMPlannerWorker
     reward_scale: 1.0

   env:
     train:
       top_reward_enabled: True
       subtask_interval: 0     # TOPReward only; set > 0 to enable subtask planning

.. _dry-run-in-dummy-mode:

Dry-Run in Dummy Mode
---------------------

Dummy mode is controlled at the **robot server**, not in the training config.
``RemoteEnv`` proxies all calls over gRPC; it does not read an ``is_dummy``
flag.

To test without real hardware, pass ``--dummy`` when starting the robot server:

.. code-block:: bash

   bash scripts/start_robot_server.sh \
       --config examples/embodiment/config/env/yam.yaml \
       --dummy   # returns zero observations, no robot movement

The training job itself requires no config change.  Use ``--dry-run`` to
print the Beaker submission command without executing:

.. code-block:: bash

   bash scripts/submit_yam_training.sh \
       --config yam_ppo_openpi \
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
       --config yam_ppo_openpi \
       --interactive --allow-dirty

   # Beaker prints a session ID.  Attach from the cluster:
   beaker session attach <session-id>

Inside the shell, the venv is activated and Ray is already running.  Start
training manually:

.. code-block:: bash

   python examples/embodiment/train_embodied_agent_staged.py \
       --config-name yam_ppo_openpi \
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
- ``train/actor/grad_norm``: Policy gradient norm; should stay below 10.
- ``train/actor/entropy``: Policy entropy; should not collapse to zero.
- ``train/actor/value_loss``: Critic (value head) loss.
- ``vlm/top_reward``: Mean TOPReward score per step.

Troubleshooting
---------------

**VLM OOM on GPU 2**

Reduce ``vlm_planner.dtype`` from ``bfloat16`` to ``float16``, or use a smaller
model (``Qwen/Qwen3-VL-3B``).

**OOM on GPU 0 or GPU 1**

Reduce ``actor.micro_batch_size`` and ``actor.global_batch_size``.  The actor
(FSDP training) and rollout (inference) use separate GPUs.

**gRPC timeout / connection refused**

Verify the reverse SSH tunnel is active:
``ssh -N -R 50051:localhost:50051 shiruic@<tailscale-ip>``

Check that ``start_robot_server.sh`` is running on the desktop and the gRPC
server bound to port 50051.

See Also
--------

- :doc:`yam_ppo_openpi_topreward` — Same pipeline with subtask planning enabled
  (``subtask_interval: 30``)
- :doc:`franka` — Real-world RL with a Franka arm
