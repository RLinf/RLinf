DSRL
====

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dsrl.png
   :align: center
   :width: 70%

   DSRL steers a frozen diffusion policy in noise space.

Train a pre-trained **Pi0 diffusion policy** with **DSRL (Diffusion Steering via Reinforcement Learning)** in RLinf.
DSRL steers a frozen Pi0 policy by training a lightweight SAC agent in the latent noise space, achieving RL fine-tuning with minimal trainable parameters (~500K).

Paper: `Steering Your Diffusion Policy with Latent Space Reinforcement Learning <https://arxiv.org/abs/2506.15799>`_ (CoRL 2025, Wagenmaker et al.)

Reference implementation: `dsrl_pi0 <https://github.com/nakamotoo/dsrl_pi0>`_

The key idea is:

1. **Lightweight SAC Agent**: A small SAC agent (~500K params) with compact CNN/MLP encoders processes observations and generates noise in the latent space.
2. **Noise Injection**: The generated noise is fed into Pi0's diffusion denoiser as the initial noise, replacing random sampling.
3. **Frozen VLM Backbone**: The pre-trained Pi0 VLM and diffusion expert remain frozen, preserving generalization capabilities.
4. **SAC Training in Noise Space**: The SAC agent is trained on the noise space using environment rewards, with a 10-Q-head ensemble critic for stable value estimation.

Overview
--------

Steer a frozen π₀ diffusion policy with a lightweight SAC agent (~500K params) in latent noise space.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      DSRL (SAC)

   .. grid-item-card:: Models
      :text-align: center

      π₀ (frozen)

   .. grid-item-card:: Environments / Data
      :text-align: center

      LIBERO-Spatial

   .. grid-item-card:: Training
      :text-align: center

      ~500K trainable params

| **You'll do:** install (same as π₀) → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · a pretrained π₀ checkpoint (see :doc:`pi0`).

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Environment
     - LIBERO-Spatial — tabletop manipulation with spatial reasoning.
   * - Observation
     - 8-dim proprioception + RGB images.
   * - Action
     - Continuous actions from π₀’s frozen diffusion denoiser, steered by the SAC noise.

How DSRL Works
--------------

**DSRL Pipeline**

1. **Observation Encoding**: A lightweight CNN (64×64 → 64-dim) and state encoder (8-dim → 64-dim) process the observation.

2. **Noise Generation**: A squashed Gaussian noise policy generates 32-dim latent noise actions for each action horizon step.

3. **Diffusion Denoising**: The generated noise replaces random initial noise inside Pi0's diffusion action sampler, and the frozen denoiser converts it into executable robot actions.

4. **SAC Training**: Standard SAC with automatic entropy tuning trains the noise generator:

   - **Actor**: 3-layer MLP squashed Gaussian policy (128-dim hidden)
   - **Critic**: 10-Q-network ensemble (~500K total params)
   - **Target Network**: Float32 EMA shadow buffer for bfloat16 precision

Setup and Run
-------------

.. include:: _setup_common.rst

DSRL shares the OpenPI environment and model dependencies with :doc:`pi0`.

.. tab-set::

   .. tab-item:: LIBERO (OpenPI π₀)
      :sync: dsrl-pi0

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate

      **2. Download**

      Download the pretrained OpenPI π₀ SFT checkpoint for LIBERO-Spatial:

      .. code-block:: bash

         hf download RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT --local-dir ./RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

      **3. Launch**

      .. list-table::
         :header-rows: 1
         :widths: 26 48 26

         * - Recipe
           - Config
           - Command suffix
         * - OpenPI π₀ + DSRL
           - ``examples/embodiment/config/libero_spatial_dsrl_openpi.yaml``
           - ``libero_spatial_dsrl_openpi``

      .. include:: _model_path.rst

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_spatial_dsrl_openpi

.. dropdown:: Advanced configuration (optional)

   Key default DSRL, SAC, and environment hyperparameters in ``libero_spatial_dsrl_openpi.yaml``:

   .. code-block:: yaml

      actor:
        model:
          openpi:
            use_dsrl: True              # Enable DSRL mode
            dsrl_state_dim: 8           # Robot proprioception dimension
            dsrl_action_noise_dim: 32   # Noise action dimension per step
            dsrl_num_q_heads: 10        # Number of Q-heads in ensemble critic
            dsrl_image_latent_dim: 64   # Image encoder output dimension
            dsrl_state_latent_dim: 64   # State encoder output dimension
            dsrl_hidden_dims: [128, 128, 128]  # MLP hidden layer dimensions

      algorithm:
        adv_type: embodied_sac
        loss_type: embodied_sac
        gamma: 0.999             # Discount factor
        tau: 0.005               # Target network soft update coefficient
        update_epoch: 200        # Training steps per interaction
        train_actor_steps: 10    # Delay actor training for this many critic updates
        entropy_tuning:
          alpha_type: softplus
          initial_alpha: 1.0
          target_entropy: -16
          optim:
            lr: 3.0e-4

      env:
        train:
          total_num_envs: 16
          use_step_penalty: True  # Use -1/0 reward style (step penalty + termination bonus)
          max_episode_steps: 240
        eval:
          total_num_envs: 500
          use_step_penalty: True

Visualization and Results
-------------------------

Monitor ``env/success_once`` with ``tensorboard --logdir ./logs --port 6006``, enable rollout video recording with ``env.eval.video_cfg.save_video=True``, and see :doc:`Training metrics <../../reference/metrics>` for all logged metrics and logger backends.

DSRL-relevant metrics:

- **Environment Metrics**:

  - ``env/episode_len``: The actual number of environment steps in the episode
  - ``env/return``: Total return of the episode
  - ``env/reward``: Step-level reward from the environment
  - ``env/success_once``: Flag indicating at least one success in the episode (0 or 1)

- **Training Metrics**:

  - ``train/sac/critic_loss``: Loss of the Q-function ensemble
  - ``train/critic/grad_norm``: Gradient norm of the Q-function

  - ``train/sac/actor_loss``: Policy loss (GaussianPolicy in noise space)
  - ``train/actor/entropy``: Policy entropy
  - ``train/actor/grad_norm``: Gradient norm of the policy

  - ``train/sac/alpha_loss``: Loss of the temperature parameter
  - ``train/sac/alpha``: Value of the temperature parameter

  - ``train/replay_buffer/size``: Current size of the replay buffer
  - ``train/replay_buffer/utilization``: Utilization of the replay buffer
