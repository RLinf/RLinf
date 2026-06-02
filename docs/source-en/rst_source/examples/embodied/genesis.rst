RL with Genesis Benchmark
================================================================

This document provides a guide for launching **MLP Policy** training tasks using the Genesis simulation environment within the RLinf framework.

Genesis is a physics-realistic multi-physics simulation platform that supports high-performance GPU parallel computing for precise contact dynamics, making it ideal for complex robotic manipulation tasks.

Environment
-----------

**Genesis Environment**

- **Environment**: Genesis Simulation Platform
- **Task**: Controlling a Franka Panda robotic arm to pick up a cube.
- **Observation**:
  - **Images**: Third-person view RGB images (256×256).
  - **States**: 16-dimensional vector (7-dim end-effector pose + 2-dim gripper + 7-dim cube pose).
- **Action Space**: 9-dimensional continuous action space.
  - 7-DOF arm joint position control.
  - 2-DOF gripper position control.

Dependency Installation
-----------------------

1. Clone the RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # Use install.sh to install Genesis-related dependencies
   bash requirements/install.sh embodied --env genesis
   source .venv/bin/activate

Running Scripts
---------------

**1. Configuration Files**

- State-only baseline: ``examples/embodiment/config/genesis_cubepick_ppo_mlp.yaml``
- Image experiment: ``examples/embodiment/config/genesis_cubepick_ppo_cnn.yaml``

**2. Launch Commands**

.. code-block:: bash

   # A) State-only baseline (MLP policy)
   bash examples/embodiment/run_embodiment.sh genesis_cubepick_ppo_mlp

   # B) Image experiment (CNN policy)
   # Note: actor.model.model_path must contain resnet10_pretrained.pt
   bash examples/embodiment/run_embodiment.sh genesis_cubepick_ppo_cnn

**Get ``resnet10_pretrained.pt`` and set ``actor.model.model_path``**

.. code:: bash

   # Method 1: git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-ResNet10-pretrained

   # Method 2: huggingface-hub (mainland China: export HF_ENDPOINT=https://hf-mirror.com)
   pip install huggingface-hub
   hf download RLinf/RLinf-ResNet10-pretrained --local-dir RLinf-ResNet10-pretrained

Point ``actor.model.model_path`` and ``rollout.model.model_path`` in the YAML
at the downloaded directory.

Visualization and Results
-------------------------

**1. TensorBoard Logs**

.. code-block:: bash

   # Launch TensorBoard
   tensorboard --logdir ./logs

**2. Key Monitoring Metrics**

- **Training Metrics**:

  - ``train/actor/policy_loss``: PPO policy loss.
  - ``train/actor/clip_fraction``: Fraction of samples triggering PPO clipping, reflecting the difference between new and old policies.
  - ``train/actor/approx_kl``: Approximate KL divergence. Monitors the magnitude of policy updates to prevent instability.
  - ``train/actor/grad_norm``: Gradient norm. Used to monitor training stability; typically increases as the model converges.
  - ``train/critic/value_loss``: Value function loss. Measures the accuracy of the Critic's state-value estimation.
  - ``train/critic/explained_variance``: Measures the fit of the value function. The closer to 1, the better.
  - ``train/actor/total_loss``: Total loss (sum of actor loss, critic loss, and entropy regularization).

- **Rollout Metrics**:

  - ``rollout/returns_mean``: Mean of the return during rollout.
  - ``rollout/advantages_max/mean/min``: Maximum, mean, and minimum values of the advantage function.
  - ``rollout/rewards``: Rewards obtained per action chunk.

- **Environment Metrics**:

  - ``env/success_once``: **Core metric**. Indicates whether the cube was successfully picked up and lifted within an episode. Success rate is expected to reach over 90% within 400 epochs.
  - ``env/episode_len``: Actual environment steps taken per episode.
  - ``env/return``: Total cumulative return per episode.
  - ``env/reward``: Step-level mean reward.

**3. Video Generation**

.. code-block:: yaml

   env:
      eval:
         video_cfg:
            save_video: True
            video_base_dir: ${runner.logger.log_path}/video/eval

**4. Logger Integration**

.. code-block:: yaml

    runner:
        task_type: embodied
        logger:
            log_path: "../results"
            project_name: rlinf
            experiment_name: "genesis_cubepick_ppo_mlp"
            logger_backends: ["tensorboard"]

Genesis Results
~~~~~~~~~~~~~~~

When training with the default parameters in ``/examples/embodiment/config/genesis_cubepick_ppo_mlp.yaml``, the ``env/success_once`` metric can reach approximately **80%**.
