GR00T Model Reinforcement Learning Training
==================================================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This example provides a complete guide to fine-tune GR00T models with reinforcement learning in the **LIBERO** environment using the **RLinf** framework. It covers the entire process—from environment setup and core algorithm design to training configuration, evaluation, and visualization—along with reproducible commands and configuration snippets.

.. note::

   RLinf supports both GR00T-N1.5 and GR00T-N1.6. N1.6 has significant upgrades in model architecture (Flow-Matching Action Head), distributed training (FSDP), and cross-embodiment generalization. Tabs below differentiate version-specific details.

Environment
-----------

**LIBERO Environment**

- **Environment**: LIBERO simulation benchmark built on top of *robosuite* (MuJoCo).
- **Task**: Command a 7-DoF robotic arm to perform a variety of household manipulation skills (pick-and-place, stacking, opening drawers, spatial rearrangement).

.. tab-set::

    .. tab-item:: N1.5

        - **Observation**: RGB images (typical resolutions 128 × 128 or 224 × 224) captured by off-screen cameras placed around the workspace.
        - **Action Space**: 7-dimensional continuous actions — 3D end-effector position control (x, y, z), 3D rotation control (roll, pitch, yaw), gripper control (open / close).

    .. tab-item:: N1.6

        - **Observation**: RGB images (typical resolutions 128×128, 224×224, or 256×256) captured by off-screen cameras placed around the workspace.
        - **Action Space**: 7-dimensional continuous actions. *Note: GR00T-N1.6 zero-pads these 7-dim actions to a 128-dim cross-embodiment universal action space via embodiment tags.*

**Task Description Format**

.. tab-set::

    .. tab-item:: N1.5

        GR00T-N1.5 directly uses the environment-provided natural-language task description as the language model input.

        **Data Structure**

        - **Images**: Main-view and wrist-view RGB tensors, named "main_images" and "wrist_images" with shape ``[batch_size, 224, 224, 3]``.
        - **States**: End-effector position, orientation, and gripper state.
        - **Task Descriptions**: Natural-language instructions.
        - **Rewards**: Sparse success/failure rewards.

    .. tab-item:: N1.6

        GR00T-N1.6 directly uses the natural language task descriptions provided by the environment as input to the language model, utilizing the frozen backbone network to extract semantic features.

        **Data Structure**

        - **Images**: Continuous RGB video frames from the main view and wrist view, typically named ``main_images`` and ``wrist_images``. Considering timestep history, the shape is usually ``[batch_size, seq_len, 224, 224, 3]``.
        - **State**: End-effector position, pose, and gripper state (concatenated with visual features at the network bottom as state representation).
        - **Task Description**: Natural language instructions.
        - **Rewards**: Sparse rewards for PPO reinforcement (1 for success, 0 for failure).

Algorithm
---------

**Core Algorithm Components**

1. **PPO (Proximal Policy Optimization)**

   - GAE (Generalized Advantage Estimation) for advantage estimation.
   - Policy clipping with ratio limits.
   - Value function clipping.
   - Entropy regularization.

Dependency Installation
-----------------------

1. Clone the RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # For faster downloads in China, you can use:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Docker Image**

Use the Docker image to run the experiments.

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # For faster image downloads in China, you can use:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

Use the built-in ``switch_env`` tool to switch to the corresponding virtual environment:

.. tab-set::

    .. tab-item:: N1.5

        .. code:: bash

           source switch_env gr00t

    .. tab-item:: N1.6

        .. code:: bash

           source switch_env gr00t_n1d6

**Option 2: Custom Environment**

.. tab-set::

    .. tab-item:: N1.5

        .. code:: bash

           # You can add ``--use-mirror`` for faster downloads in China
           bash requirements/install.sh embodied --model gr00t --env maniskill_libero
           source .venv/bin/activate

    .. tab-item:: N1.6

        .. code:: bash

           # You can add ``--use-mirror`` for faster downloads in China
           bash requirements/install.sh embodied --model gr00t_n1d6 --env maniskill_libero
           source .venv/bin/activate

Model Download
--------------

Before starting training, you need to download the corresponding pre-trained model.

.. tab-set::

    .. tab-item:: N1.5

        **GR00T-N1.5 Few-Shot SFT Model Download**

        We currently support four LIBERO tasks: Spatial, Object, Goal, and Long.

        .. code:: bash

           # Method 1: Using git clone
           git lfs install
           git clone https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Spatial

           # Method 2: Using huggingface-hub
           # For faster downloads in China, set:
           # export HF_ENDPOINT=https://hf-mirror.com
           pip install huggingface-hub
           hf download RLinf/RLinf-Gr00t-SFT-Spatial --local-dir RLinf-Gr00t-SFT-Spatial

        SFT model downloads for other tasks:
        - `Libero-Object <https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Object>`_
        - `Libero-Goal <https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Goal>`_
        - `Libero-Long <https://huggingface.co/RLinf/RLinf-Gr00t-SFT-10>`_

    .. tab-item:: N1.6

        **GR00T-N1.6 SFT Model**

        You need to run the RLinf-provided GR00T-N1.6 SFT first, obtain the format-converted model, and configure the model path in the designated YAML file.

        RLinf SFT models will be released soon — stay tuned!

        Currently supports four LIBERO tasks: Spatial, Object, Goal, 10.

--------------

GR00T Core Design Concepts
-----------------------------

.. tab-set::

    .. tab-item:: N1.5

        **1. Modality Config**

        Modality Config is a critical design feature in GR00T-N1.5.
        By defining a unified dataset interface, it enables different robot configurations to utilize the same dataset. For example, a dual-arm dataset can be used to train a single-arm model through this innovative design.

        **1.1 Enhanced LeRobot Dataset**

        The LeRobot dataset contains a ``meta`` folder that records all dataset metadata.
        GR00T-N1.5 further defines a ``modality.json`` file to determine the data interface of the dataset.

        **1.2 DataConfig Class**

        GR00T-N1.5 introduces the ``DataConfig`` class to describe all information needed for model training.
        It decouples datasets from robot configurations, enabling model training across different robots without modifying data processing code.

        **1.3 Embodiment Tag**

        The Embodiment Tag is an enum value that determines which ``DataConfig`` to use during training. The model also adopts different state and action encoders/decoders based on this tag.

        **2. Fine-Tuning Guide**

        Based on the above design, before deploying GR00T-N1.5 in new environments beyond LIBERO, users need to fine-tune it.
        The fine-tuning guide can be found at `GR00T official repo's getting_started/finetune_new_embodiment.md <https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md>`_.

        After fine-tuning, GR00T-N1.5 generates an ``experiment_cfg/metadata.json`` file containing all modality configs and fine-tuned dataset statistics.
        This file is essential for GR00T-N1.5 inference and RL post-training.
        For more details, see `GR00T official repo's getting_started/GR00T_inference.ipynb <https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/GR00T_inference.ipynb>`__.

    .. tab-item:: N1.6

        **1. Two-Stage Decoupled Training Paradigm**

        RLinf adopts a highly decoupled two-stage training architecture for GR00T-N1.6:

        - **Stage 1 (Pure SFT)**: Uses ``Pure SFT Model`` mode. The model is completely detached from the physical simulation environment, relying solely on offline expert datasets for supervised fine-tuning.
        - **Stage 2 (PPO RL Alignment)**: Based on SFT convergence, loads the model into a FSDP-based distributed Actor for real-time interaction with the simulation environment.

        **2. Head-Only Fine-Tuning**

        To save memory while preventing "catastrophic forgetting", the framework adopts a backbone-freezing strategy:

        - **Backbone Freezing**: Vision-language backbone parameters are strictly locked (``requires_grad=False``).
        - **Action Head Focus**: Only the action output head participates in gradient updates.

        **3. Flow-Matching Action Generation**

        - The model generates high-frequency action chunks directly in continuous space through noise-adding and denoising flow-matching mechanisms (Flow-SDE / Diffusion).
        - Key configurations: ``num_action_chunks`` controls prediction step length, ``denoising_steps`` controls denoising depth.

        **4. Cross-Embodiment Generalization**

        - **Embodiment Tag**: Through configuration tags (e.g., ``ROBOCASA_PANDA_OMRON``), the system dynamically adapts the corresponding state encoder and action space. Both single-arm manipulators and quadruped robots can reuse the same architecture.

        **5. FSDP Distributed Parallel Architecture**

        - The underlying system has been restructured for the Actor node (``EmbodiedFSDPActor``), which shards model weights, gradients, and optimizer states across GPU nodes.
        - Given the significant increase in GR00T-N1.6 parameter scale, the Actor node has been fully restructured to break through the single-GPU memory bottleneck of traditional DDP.

        After fine-tuning, the system generates ``metadata.json`` and other statistical files in the output directory, preserving key modality information for inference and deployment.

---------------

Running Scripts
---------------

**1. Key Cluster Configuration**

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

You can configure the placement to share all GPUs among env, rollout, and actor components.

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-3
         rollout: 4-7
         actor: 0-7

   rollout:
      pipeline_stage_num: 2

You can flexibly configure GPU counts for env, rollout, and actor components, and enable pipelining between rollout and env via ``pipeline_stage_num``.

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-1
         rollout: 2-5
         actor: 6-7

You can also fully separate components, each using dedicated GPUs without offloading.

--------------

**2. Key Model Parameters**

.. tab-set::

    .. tab-item:: N1.5

        .. code:: yaml

           model:
              num_action_chunks: 5
              denoising_steps: 4
              rl_head_config:
                noise_method: "flow_sde"
                noise_level: 0.5
                disable_dropout: True

        You can adjust ``noise_level`` and ``denoising_steps`` to control noise intensity and flow-matching steps.
        ``num_action_chunks`` determines the number of future steps to use for forward simulation.
        GR00T-N1.5's action head contains dropout layers that interfere with log-probability calculations, so ``disable_dropout`` must be set to True to replace them with identity layers.
        Use ``noise_method`` to select different noise injection methods. Two options are available:
        `flow-sde <https://arxiv.org/abs/2505.05470>`__ and
        `flow-noise <https://arxiv.org/abs/2505.22094>`__.

    .. tab-item:: N1.6

        **Actor Model & Action Head Configuration**

        .. code:: yaml

           model:
              model_type: "gr00t_n1d6"
              add_value_head: True          # RL critical: dynamically inject value network for advantage prediction
              num_action_chunks: 1          # Set to 1 during PPO to improve real-time physical interaction
              denoising_steps: 1            # Controls flow-matching denoising steps

        **FSDP Sharding Strategy**

        .. code:: yaml

           fsdp_config:
             wrap_policy:
               transformer_layer_cls_to_wrap:
                 - "Qwen3DecoderLayer"
                 - "Siglip2EncoderLayer"

        **PPO & Optimizer Hyperparameters**

        .. code:: yaml

           algorithm:
              adv_type: gae
              clip_ratio_high: 0.2
              gamma: 0.99
              gae_lambda: 0.95

           optim:
              lr: 5.0e-6
              value_lr: 1.0e-4
              clip_grad: 1.0

**3. Configuration Files**

.. tab-set::

    .. tab-item:: N1.5

        - GR00T-N1.5 + PPO + Libero-Spatial:
          ``examples/embodiment/config/libero_spatial_ppo_gr00t.yaml``

        - GR00T-N1.5 + PPO + Libero-Object:
          ``examples/embodiment/config/libero_object_ppo_gr00t.yaml``

        - GR00T-N1.5 + PPO + Libero-Goal:
          ``examples/embodiment/config/libero_goal_ppo_gr00t.yaml``

        - GR00T-N1.5 + PPO + Libero-Long:
          ``examples/embodiment/config/libero_10_ppo_gr00t.yaml``

    .. tab-item:: N1.6

        - GR00T-N1.6 + PPO + Libero-Spatial:
          ``examples/embodiment/config/libero_spatial_ppo_gr00t_n1d6.yaml``

        Update the SFT model path:

        .. code:: yaml

           model:
              model_path: "/path/to/GR00T-N1.6-libero-spatial-ppo"

--------------

**4. Launch Commands**

.. tab-set::

    .. tab-item:: N1.5

        ::

           bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_gr00t
           bash examples/embodiment/run_embodiment.sh libero_object_ppo_gr00t
           bash examples/embodiment/run_embodiment.sh libero_goal_ppo_gr00t
           bash examples/embodiment/run_embodiment.sh libero_10_ppo_gr00t

    .. tab-item:: N1.6

        .. code:: bash

           bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_gr00t_n1d6

--------------

Visualization & Results
-------------------------

**1. TensorBoard Logs**

.. code:: bash

   # Launch TensorBoard
   tensorboard --logdir ./logs --port 6006

--------------

**2. Key Monitoring Metrics**

- **Training Metrics**

  - ``actor/loss``: Policy loss
  - ``actor/value_loss``: Value function loss (PPO)
  - ``actor/grad_norm``: Gradient norm
  - ``actor/approx_kl``: KL divergence between old and new policy

- **Rollout Metrics**

  - ``rollout/returns_mean``: Average episode returns
  - ``rollout/advantages_mean``: Average advantage values

- **Environment Metrics**

  - ``env/episode_len``: Average episode length
  - ``env/success_once``: Task success rate

--------------

**3. Video Generation**

.. code:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

--------------

**4. WandB Integration**

.. code:: yaml

   runner:
     task_type: embodied
     logger:
       log_path: "../results"
       project_name: rlinf
       experiment_name: "libero_spatial_ppo_gr00t"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab

--------------

**LIBERO Results**

.. tab-set::

    .. tab-item:: N1.5

        .. list-table:: **GR00T-N1.5 Model Results with Flow-SDE on LIBERO**
           :header-rows: 1

           * - Model
             - Spatial
             - Object
             - Goal
             - Long
             - Average
             - Δ Avg.

           * - GR00T (few-shot)
             - |huggingface| `41.4% <https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Spatial>`_
             - |huggingface| `58.6% <https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Object>`_
             - |huggingface| `48.2% <https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Goal>`_
             - |huggingface| `61.9% <https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Long>`_
             - 52.5%
             - ---

           * - +PPO
             - |huggingface| `92.5% <https://huggingface.co/RLinf/RLinf-Gr00t-RL-Spatial-Step400>`_
             - |huggingface| `95.0% <https://huggingface.co/RLinf/RLinf-Gr00t-RL-Object-Step400>`_
             - |huggingface| `84.3% <https://huggingface.co/RLinf/RLinf-Gr00t-RL-Goal-Step500>`_
             - |huggingface| `86.3% <https://huggingface.co/RLinf/RLinf-Gr00t-RL-Long-Step300>`_
             - **89.5%**
             - **+37.0%**

    .. tab-item:: N1.6

        .. raw:: html

           <div style="display: flex; justify-content: center; margin: 20px 0;">
             <div style="flex: 0.5; text-align: center;">
               <img src="https://github.com/RLinf/misc/blob/main/pic/gr00t_1.6_ppo_success_rate.png?raw=true" style="width: 100%;"/>
               <p><em>GR00T-N1.6 SFT + PPO Accuracy Curve on LIBERO_Spatial</em></p>
             </div>
           </div>

        .. list-table:: **GR00T-N1.6 Results with Flow-SDE on LIBERO Spatial**
           :header-rows: 1

           * - Model
             - Spatial

           * - GR00T-N1.6 SFT
             - |huggingface| `70% <https://huggingface.co/RLinf/RLinf-Gr00t-N1.6-RL-Spatial>`_

           * - +PPO
             - |huggingface| `82% <https://huggingface.co/RLinf/RLinf-Gr00t-N1.6-RL-Spatial-Step500>`_
