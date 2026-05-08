ManiSkill PPO with ResNet Reward Model
======================================

This document provides a complete guide for running ManiSkill PPO in RLinf with an **MLP policy + ResNet reward model**.
The main reference config is ``examples/embodiment/config/maniskill_ppo_mlp_resnet_reward.yaml``.

The main goals of this setup are:

1. **State-based policy learning**: the actor uses ``mlp_policy`` over ``states``.
2. **Image-based reward judgement**: the reward worker uses a ResNet model on RGB observations.
3. **Reward fusion**: learned reward is mixed with environment reward through explicit weights.
4. **RL optimization**: PPO continuously updates the policy.

Environment
-----------

**ManiSkill3 Environment**

- **Environment**: ManiSkill3 simulation platform
- **Task**: robotic manipulation tasks such as ``PickCube-v1``
- **Policy Observation**: ``states``
- **Reward Observation**: ``main_images``
- **Action Space**: 7-dimensional continuous actions

**Data Structure**

- **States**: state vectors for ``mlp_policy``
- **Main Images**: image input for the ResNet reward model
- **Rewards**: final rewards are computed from env reward and reward-model output

Algorithm
-----------------------------------------

**Core Algorithm Components**

1. **PPO (Proximal Policy Optimization)**

   - GAE-based advantage estimation
   - clipped policy update
   - optional value clipping and entropy regularization

2. **MLP Policy**

   - the actor only consumes ``states``
   - the policy branch stays lightweight

3. **ResNet Reward Model**

   - the reward worker uses ``ResNetRewardModel``
   - input is ``main_images``
   - output is a sigmoid probability used in final reward computation

Dependency Installation
-----------------------

1. Clone RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Docker Image**

Use Docker image for the experiment.

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

For ResNet reward experiments, you can directly switch to an image-bundled environment with ManiSkill dependencies:

.. code:: bash

   source switch_env openvla

**Option 2: Custom Environment**

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.
   bash requirements/install.sh embodied --model openvla --env maniskill_libero
   source .venv/bin/activate

Assets Download
----------------

Download the ManiSkill assets by running the following command:

.. code:: bash

   cd <path_to_RLinf>/rlinf/envs/maniskill
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets

Model Download
--------------

Before starting training, prepare the ResNet checkpoint used by the reward worker.
If you do not have a trained reward model yet, first follow
:doc:`/rst_source/tutorials/extend/reward_model` for offline reward-data preprocessing and training.

Make sure the config yaml correctly sets:

- ``reward.model.model_path``

Running the Script
-------------------

**1. Key Parameters Configuration**

.. code-block:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         actor: 0-0
         env: 0-0
         rollout: 0-0
         reward: 0-0

   env:
      train:
         wrap_obs_mode: simple
         use_full_state: True
         init_params:
            id: PickCube-v1
            obs_mode: rgb

   reward:
      use_reward_model: True
      reward_mode: terminal
      reward_weight: 1.0
      env_reward_weight: 0.0
      model:
         model_type: resnet
         arch: resnet18
         model_path: /path/to/reward_model_checkpoint

These parameters matter because:

- ``component_placement.reward`` places the online reward worker.
- ``obs_mode: rgb`` makes the env expose ``main_images`` for ResNet reward inference.
- ``wrap_obs_mode: simple`` and ``use_full_state: True`` still keep ``states`` for ``mlp_policy``.
- ``reward_mode: terminal`` means learned reward is only written at done steps, which is usually safer for sparse-success tasks.

**2. Configuration Files**

You can directly refer to:

- ``examples/embodiment/config/maniskill_ppo_mlp_resnet_reward.yaml``

**3. Launch Commands**

After choosing a config, start training with:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_resnet_reward

Visualization and Results
-------------------------

**1. TensorBoard Logging**

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

**2. Key Metrics Tracked**

- ``env/success_once``: the most direct success-rate metric to watch.
- ``env/reward``: raw environment step reward.
- ``rollout/rewards``: mixed rollout reward after reward-model integration.
- ``train/actor/policy_loss``: policy optimization status.

Online Call Chain
-----------------

At runtime, the reward path is:

.. code-block:: text

   train_embodied_agent.py
      -> EmbodiedRunner
      -> EnvWorker + MultiStepRolloutWorker + EmbodiedRewardWorker
      -> ResNetRewardModel

More concretely:

1. ``train_embodied_agent.py`` creates ``EmbodiedRewardWorker`` when ``reward.use_reward_model=True``.
2. ``EnvWorker`` sends observations to the rollout worker for action generation, and sends reward inputs to the reward worker.
3. ``EmbodiedRewardWorker`` resolves ``reward.model.model_type="resnet"`` through ``rlinf.models.embodiment.reward.get_reward_model_class`` and instantiates ``ResNetRewardModel``.
4. ``ResNetRewardModel.compute_reward`` reads ``main_images``, applies image preprocessing, runs the ResNet head, and returns sigmoid probabilities.
5. ``EnvWorker.compute_bootstrap_rewards`` combines env reward and reward-model output as:

   .. code-block:: python

      reward = env_reward_weight * env_reward + reward_weight * reward_model_output

6. When ``reward_mode="terminal"``, ``EnvWorker`` uses ``_scatter_terminal_reward_output`` so only done steps receive reward-model output. When ``reward_mode="per_step"``, every step receives reward-model output directly.

Current Implementation Notes
----------------------------

- ``cluster.component_placement.reward`` is required for online reward inference. Without it, the reward worker group cannot be launched.
- ``reward.reward_weight`` and ``reward.env_reward_weight`` control the mixture between learned reward and env reward. The example sets ``env_reward_weight: 0.0``.
- ``reward_threshold`` is kept at the top-level ``reward`` section in the example, but the current embodied reward worker only passes ``reward.model`` into ``ResNetRewardModel``. As a result, this top-level threshold is not actually consumed by the current online path.
- ``ResNetRewardModel.compute_reward`` currently expects an observation dict containing ``main_images``. Passing a raw image tensor or ndarray directly is not part of the current interface.
