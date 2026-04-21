NFT on π\ :sub:`0`\ : Negative-aware FineTuning
=====================================================

This document provides a guide for fine-tuning a pre-trained **π₀ flow-matching policy** using **NFT (Negative-aware FineTuning)** in the RLinf framework.
NFT is a critic-and-likelihood-free online RL framework for flow-based vision-language-action (VLA) policies. It applies step-level DPO-style preference optimization directly on the flow-matching denoising trajectory, requiring only one forward pass per optimization step without auxiliary value networks.

Paper: `π-StepNFT: Wider Space Needs Finer Steps in Online RL for Flow-based VLAs <https://arxiv.org/abs/2603.02083>`_ (Wang et al., 2026)

Reference implementation: `pi-StepNFT <https://github.com/wangst0181/pi-StepNFT>`_

The key idea is:

1. **SDE-based Exploration**: A reverse-time SDE formulation injects stochasticity during flow-matching sampling, expanding the exploration manifold beyond deterministic ODE trajectories.
2. **Step-wise Supervision**: Instead of optimizing the entire denoising chain, NFT targets the immediate next denoising step with a noise-aware regression signal, providing finer-grained learning.
3. **DPO-style Preference Loss**: A logistic contrastive ranking loss implements push-pull dynamics — promoting velocity predictions from successful trajectories while suppressing those from failures.
4. **Lightweight Training**: Only one forward pass per optimization step. No critic network or likelihood computation is needed, making it significantly more efficient than PPO-based approaches.

Environment
-----------

**LIBERO Environment**

- **Environment**: LIBERO Goal benchmark (also supports Spatial, Object, Long)
- **Task**: Tabletop manipulation tasks requiring goal-directed reasoning
- **Observation**: Robot proprioception + RGB images (224 x 224)
- **Action Space**: 7-dimensional continuous actions (3D position + 3D rotation + gripper)

Algorithm
---------

**NFT Pipeline**

1. **Rollout with SDE Sampling**: The π₀ policy generates action trajectories using multi-step flow-matching denoising. At a randomly sampled denoising step per trajectory, the intermediate state ``x_t``, velocity ``v_t``, and next state ``x_{t+1}`` are recorded as NFT traces.

2. **Advantage Computation**: Raw advantages are computed from environment rewards (sparse success/failure signals).

3. **DPO Preference Signal**: Advantages are mapped to preference labels ``y ∈ [-1, 1]``, where positive advantages indicate preferred trajectories and negative advantages indicate dispreferred ones, clipped by ``adv_clip_max``.

4. **Velocity Drift Loss**: The energy-based NFT loss computes the drift between the current velocity prediction ``v_θ`` and the old velocity ``v_old``, incorporating the SDE noise scale ``σ_i``. The velocity change ``δv`` is clipped by ``max_drift`` to stabilize training.

5. **Single Forward Pass**: Only one forward pass through the action expert is needed per training step — the frozen VLM prefix is cached and reused.

Installation
------------

NFT uses the same environment and model dependencies as π₀. Please refer to :doc:`pi0` for the full installation guide, including Docker image setup, dependency installation, and model download.

Running Scripts
---------------

**1. Configuration File**

- **NFT Training**: ``examples/embodiment/config/libero_goal_nft_openpi.yaml``

**2. Key Parameter Configuration**

**2.1 NFT Model Parameters**

.. code:: yaml

   actor:
     model:
       add_value_head: False        # No critic needed
       openpi:
         is_nft: True               # Enable NFT trace collection
         noise_level: 0.2           # SDE noise intensity

**2.2 Algorithm Parameters**

.. code:: yaml

   algorithm:
     adv_type: raw                  # Raw advantage (no GAE)
     loss_type: embodied_nft        # NFT loss type
     update_epoch: 4                # Training epochs per rollout
     nft_beta: 1.0                  # Velocity scaling coefficient
     adv_clip_max: 1.0              # Advantage clipping bound for DPO signal
     dpo_beta: 1.0                  # DPO loss temperature
     max_drift: 0.5                 # Max velocity drift norm for clipping

**2.3 Environment Parameters**

.. code:: yaml

   env:
     train:
       total_num_envs: 64
       max_episode_steps: 320
     eval:
       total_num_envs: 500

**3. Launch Command**

::

   bash examples/embodiment/run_embodiment.sh libero_goal_nft_openpi

Visualization and Results
-------------------------

**1. TensorBoard Logs**

.. code-block:: bash

   tensorboard --logdir ./logs

**2. Key Monitoring Metrics**

- **Environment Metrics**:

  - ``env/episode_len``: Average episode length
  - ``env/success_once``: Task success rate (0 or 1 per episode)

- **Training Metrics**:

  - ``train/actor/nft_loss``: NFT preference loss
  - ``train/actor/E_pos_mean``: Energy of preferred (successful) trajectories
  - ``train/actor/E_neg_mean``: Energy of dispreferred (failed) trajectories
  - ``train/actor/delta_v_norm``: Velocity drift norm (before clipping)
  - ``train/actor/grad_norm``: Gradient norm

Citation
--------

.. code-block:: bibtex

   @misc{wang2026pistepnft,
     title={$\pi$-StepNFT: Wider Space Needs Finer Steps in Online RL for Flow-based VLAs},
     author={Wang, Siting and Wang, Xiaofeng and Zhu, Zheng and Pei, Minnan and Cui, Xinyu and Deng, Cheng and Zhao, Jian and Huang, Guan and Zhang, Haifeng and Wang, Jun},
     year={2026},
     eprint={2603.02083},
     archivePrefix={arXiv},
     primaryClass={cs.RO},
     url={https://arxiv.org/abs/2603.02083},
   }
