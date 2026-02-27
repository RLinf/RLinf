MLP Policy Reinforcement Learning Training
==========================================

This example demonstrates the complete workflow for training Reinforcement Learning (RL) agents using **MLP (Multi-Layer Perceptron)** policy networks within the RLinf framework.

The MLP policy is primarily designed for robotics control tasks utilizing **low-dimensional state inputs**. It supports training across various simulation environments, including **ManiSkill3**, **FrankaSim**, and **Libero-Spatial**.

The current configuration covers **PPO-MLP**, **SAC-MLP**, and **GRPO-MLP** algorithm setups, enabling rapid validation of environments, training pipelines, and network architectures.

The primary goal is to equip the model with the following capabilities:

1.  **State Understanding**: Process low-dimensional proprioceptive data from the environment (joint angles, end-effector pose, object states, etc.).
2.  **Action Generation**: Produce continuous control actions (end-effector position deltas, joint targets, gripper commands, etc.).
3.  **Reinforcement Learning**: Optimize policies using PPO or SAC based on environmental feedback.

Environments
------------

RLinf currently supports a diverse range of embodied intelligence environments. You can select different environment configurations via the **defaults** list using ``env/<env_name>@env.train`` and ``env/<env_name>@env.eval``.

Specific parameters such as parallel environment count, episode length, reset protocols, and video recording can be overridden under the ``env.train`` / ``env.eval`` nodes.

Currently supported environments (covered in this example) include:

-   ``maniskill_pick_cube`` (ManiSkill3)
-   ``libero_spatial`` (LIBERO Spatial)
-   ``frankasim_pickcube_state`` (Mujoco / FrankaSim)

You can also train on custom tasks by referencing specific environment configurations:

1.  Reference the environment in the configuration file via defaults (training and evaluation can be specified separately).

.. code:: yaml

   defaults:
     - env/maniskill_pick_cube@env.train
     - env/maniskill_pick_cube@env.eval

   defaults:
     - env/libero_spatial@env.train
     - env/libero_spatial@env.eval

   defaults:
     - env/frankasim_pickcube_state@env.train
     - env/frankasim_pickcube_state@env.eval

Algorithms
----------

**Core Algorithm Components**

1.  **PPO (Proximal Policy Optimization)**

    -   Adopts an on-policy Actor-Critic framework.
    -   Uses **GAE (Generalized Advantage Estimation)** for advantage function estimation: ``adv_type: gae``.
    -   Utilizes ratio clipping to constrain policy updates, with optional KL divergence constraints.

2.  **SAC (Soft Actor-Critic)**

    -   Learns Q-values via Bellman backups and entropy regularization (off-policy).
    -   Uses an MLP as the Actor policy network; ensure Q-related heads/structures are enabled in the configuration (``add_q_head: True``).
    -   Supports **Automatic Entropy Tuning** via ``entropy_tuning`` (e.g., ``alpha_type: softplus``) to balance exploration and exploitation.

3.  **GRPO (Group Relative Policy Optimization)**

    -   For each state/prompt, the policy generates *G* independent actions.
    -   Uses the group average reward as a baseline to calculate the relative advantage of each action.

Installation & Dependencies
---------------------------

For running in simulation environments, please refer to :doc:`../../start/installation` for installation instructions.

This configuration series uses Hydra's ``searchpath`` to load external configuration directories via environment variables:

-   ``hydra.searchpath: file://${oc.env:EMBODIED_PATH}/config/``

Please ensure that ``EMBODIED_PATH`` is correctly set and that dependencies/resources for ManiSkill3 / FrankaSim are installed.

Running Scripts
---------------

**1. Configuration Files**

RLinf provides several default MLP configurations covering different environments and algorithm settings:

-   **ManiSkill + PPO + MLP**: ``maniskill_ppo_mlp``
-   **ManiSkill + SAC + MLP**: ``maniskill_sac_mlp``
-   **FrankaSim + PPO + MLP**: ``franka_sim_ppo_mlp``

**2. Key Parameter Configuration**

**2.1 Model Parameters (Model)**

The MLP model is introduced via ``model/mlp_policy@actor.model`` and can be overridden in different configurations. Key fields include:

.. code:: yaml

   model_type: "mlp_policy"                # Use MLP policy network as actor (Multi-Layer Perceptron; fits low-dim state inputs)

   model_path: ""

   policy_setup: "panda-qpos"              # Select action semantics and control mode; 'panda-qpos' usually implies joint space control (e.g., qpos/joint targets or deltas)

   obs_dim: 42                             # Input dimension of the state vector (must match environment state output)

   action_dim: 8                           # Output dimension of the action vector (must match environment action space)

   num_action_chunks: 1                    # Number of action chunks generated per forward pass

   hidden_dim: 256                         # Width/Channel size of MLP hidden layers

   precision: "32"                         # Model parameter and computation precision

   add_value_head: True                    # Whether to attach an additional value head to the policy network

   is_lora: False                          # Whether to enable LoRA

   lora_rank: 32                           # LoRA rank dimension 'r'; only effective when is_lora=True

**2.2 Cluster & Hardware Configuration (Cluster)**

For real-robot training, a multi-node configuration is used, deploying the Actor/Policy on GPU servers and the Env/Robot on control machines (NUC/Industrial PC). For specific configurations, please refer to :doc:`franka`.

**3. Launch Commands**

**ManiSkill (PPO-MLP)**

::

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp

**ManiSkill (SAC-MLP)**

::

   bash examples/embodiment/run_embodiment.sh maniskill_sac_mlp

**Libero-Spatial (GRPO-MLP)**

::

   bash examples/embodiment/run_embodiment.sh libero_spatial_0_grpo_mlp

**FrankaSim (PPO-MLP)**

::

   bash examples/embodiment/run_embodiment.sh franka_sim_ppo_mlp

Visualization & Results
-----------------------

**1. TensorBoard Logs**

.. code-block:: bash

   # Launch TensorBoard
   tensorboard --logdir ../results

**2. Key Monitoring Metrics**

-   **Environment Metrics**:

    -   ``env/episode_len``: Actual environment steps taken in an episode (Unit: step).
    -   ``env/return``: Total cumulative return of the episode.
    -   ``env/reward``: Step-level reward signal.
    -   ``env/success_once``: Flag indicating if success was achieved at least once in the episode (if provided by environment).

-   **Training Metrics (SAC)**:

    -   ``train/sac/critic_loss``: Q-function loss.
    -   ``train/sac/actor_loss``: Policy loss.
    -   ``train/sac/alpha_loss``: Temperature parameter loss.
    -   ``train/sac/alpha``: Temperature parameter value.
    -   ``train/replay_buffer/size``: Replay buffer size.

-   **Training Metrics (PPO)**:

    -   Policy Loss
    -   Value Loss
    -   Approx KL / KL (Estimated KL Divergence)
    -   Clip Frac (Ratio clipping proportion)
    -   Entropy (Policy entropy)