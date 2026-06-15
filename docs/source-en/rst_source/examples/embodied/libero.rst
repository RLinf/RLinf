RL with LIBERO Benchmarks
=========================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This page documents two related families of LIBERO RL recipes in RLinf:

- :ref:`libero-benchmark` — the original LIBERO suites (Spatial / Goal / Object / Long / 90 / 130) with OpenVLA-OFT + PPO/GRPO.
- :ref:`liberopro-plus-benchmark` — the harder LIBERO-Pro and LIBERO-Plus evaluation suites that stress generalization with anti-memorization perturbations.

For LIBERO setup on **AMD ROCm** or **Ascend CANN** accelerators, see the :doc:`Supported Accelerators <../../tutorials/accelerators/index>` tutorial.

.. _libero-benchmark:

LIBERO Benchmark
----------------

This section provides a comprehensive guide to launching and managing the
Vision-Language-Action Models (VLAs) training task within the RLinf framework,
focusing on finetuning a VLA model for robotic manipulation in the LIBERO environment.

The primary objective is to develop a model capable of performing robotic manipulation by:

1. **Visual Understanding**: Processing RGB images from the robot's camera.
2. **Language Comprehension**: Interpreting natural-language task descriptions.
3. **Action Generation**: Producing precise robotic actions (position, rotation, gripper control).
4. **Reinforcement Learning**: Optimizing the policy via the PPO with environment feedback.

Environment
~~~~~~~~~~~

**LIBERO Environment**

- **Environment**: LIBERO simulation benchmark built on top of *robosuite* (MuJoCo).
- **Task**: Command a 7-DoF robotic arm to perform a variety of household manipulation skills (pick-and-place, stacking, opening drawers, spatial rearrangement).
- **Observation**: RGB images (typical resolutions 128 × 128 or 224 × 224) captured by off-screen cameras placed around the workspace.
- **Action Space**: 7-dimensional continuous actions
  - 3D end-effector position control (x, y, z)
  - 3D rotation control (roll, pitch, yaw)
  - Gripper control (open / close)

**Task Description Format**

.. code-block:: text

   In: What action should the robot take to [task_description]?
   Out:

**Data Structure**

- **Images**: RGB tensors ``[batch_size, 224, 224, 3]``
- **Task Descriptions**: Natural-language instructions
- **Actions**: Normalized continuous values converted to discrete tokens
- **Rewards**: Step-level rewards based on task completion

Algorithm
~~~~~~~~~

**Core Algorithm Components**

1. **PPO (Proximal Policy Optimization)**

   - Advantage estimation using GAE (Generalized Advantage Estimation)

   - Policy clipping with ratio limits

   - Value function clipping

   - Entropy regularization

2. **GRPO (Group Relative Policy Optimization)**

   - For every state / prompt the policy generates *G* independent actions

   - Compute the advantage of each action by subtracting the group's mean reward.


3. **Vision-Language-Action Model**

   - OpenVLA architecture with multimodal fusion

   - Action tokenization and de-tokenization

   - Value head for critic function

Dependency Installation
~~~~~~~~~~~~~~~~~~~~~~~

1. Clone RLinf Repository
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install Dependencies
^^^^^^^^^^^^^^^^^^^^^^^

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

Please switch to the corresponding virtual environment via the built-in `switch_env` utility in the image:

.. code:: bash

   source switch_env openvla-oft

**Option 2: Custom Environment**

Install dependencies directly in your environment by running the following command:

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.

   bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
   source .venv/bin/activate

Model Download
~~~~~~~~~~~~~~

Before starting training, you need to download the corresponding pretrained model:

.. code:: bash

   # Download the model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

   # Method 2: Using huggingface-hub
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora --local-dir RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
   hf download RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora --local-dir RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

After downloading, make sure to correctly specify the model path in the configuration yaml file.

.. code:: yaml

   rollout:
      model:
         model_path: Pathto/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
   actor:
      model:
         model_path: Pathto/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora

Running the Script
~~~~~~~~~~~~~~~~~~

**1. Key Parameters Configuration**

.. code-block:: yaml

   cluster:
      num_nodes: 2
      component_placement:
         env: 0-7
         rollout: 8-15
         actor: 0-15

   rollout:
      pipeline_stage_num: 2

Here you can flexibly configure the GPU count for env, rollout, and actor components.
Additionally, by setting `pipeline_stage_num = 2` in the configuration, you can achieve pipeline overlap between rollout and env, improving rollout efficiency.

.. code-block:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

You can also reconfigure the placement to achieve complete sharing, where env, rollout, and actor components all share all GPUs.

.. code-block:: yaml

   cluster:
      num_nodes: 2
      component_placement:
         env: 0-3
         rollout: 4-7
         actor: 8-15

You can also reconfigure the placement to achieve complete separation, where env, rollout, and actor components each use their own GPUs without interference, eliminating the need for offload functionality.

**2. Configuration Files**

We currently support training in two environments: **ManiSkill3** and **LIBERO**.

We support the **OpenVLA-OFT** model with both **PPO** and **GRPO** algorithms.
The corresponding configuration files are:

- **OpenVLA-OFT + PPO**: ``examples/embodiment/config/libero_10_ppo_openvlaoft.yaml``
- **OpenVLA-OFT + GRPO**: ``examples/embodiment/config/libero_10_grpo_openvlaoft.yaml``

**3. Launch Commands**

To start training with a chosen configuration, run the following command:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

For example, to train the OpenVLA-OFT model using the GRPO algorithm in the LIBERO environment, run:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft


Visualization and Results
~~~~~~~~~~~~~~~~~~~~~~~~~

**1. TensorBoard Logging**

.. code-block:: bash

   # Start TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. Key Metrics Tracked**

- **Training Metrics**:

  - ``train/actor/approx_kl``: Approximate KL divergence between old and new policies.
  - ``train/actor/clip_fraction``: Fraction of updates where the probability ratio was clipped.
  - ``train/actor/clipped_ratio``: Mean of the clipped probability ratios.
  - ``train/actor/grad_norm``: Gradient norm.
  - ``train/actor/lr``: Learning rate.
  - ``train/actor/policy_loss``: PPO/GRPO policy loss.
  - ``train/critic/value_loss``: Value function loss.
  - ``train/critic/value_clip_ratio``: Fraction of value targets whose update was clipped.
  - ``train/critic/explained_variance``: Explained variance of the value function predictions.
  - ``train/entropy_loss``: Policy entropy.
  - ``train/loss``: Total training loss (actor_loss + critic_loss + entropy_loss regularization).

- **Rollout Metrics**:

  - ``rollout/advantages_max``: the max of the advantage.
  - ``rollout/advantages_mean``: the mean of the advantage.
  - ``rollout/advantages_min``: the min of the advantage.
  - ``rollout/rewards``: chunk of reward (refer to L414 in libero_env.py).

- **Environment Metrics**:

  - ``env/episode_len``: Number of environment steps elapsed in the episode (unit: step).
  - ``env/return``: Episode return. In LIBERO's sparse-reward setting this metric is not informative, since the reward is almost always 0 until the terminal success step.
  - ``env/reward``: Step-level reward (0 for all intermediate steps and 1 only at successful termination).
    The logged value is normalized by the number of episode steps, which makes it difficult to interpret as real task performance during training.
  - ``env/success_once``: Recommended metric to monitor training performance. It directly reflects the unnormalized episodic success rate.

**3. Video Generation**

.. code-block:: yaml

   env:
      eval:
         video_cfg:
            save_video: True
            video_base_dir: ${runner.logger.log_path}/video/eval

**4. Train Log Tool Integration**

.. code-block:: yaml

   runner:
      task_type: embodied
      logger:
         log_path: "../results"
         project_name: rlinf
         experiment_name: "libero_10_grpo_openvlaoft"
         logger_backends: ["tensorboard"] # wandb, swanlab


LIBERO Results
^^^^^^^^^^^^^^

In order to show the RLinf's capability for large-scale multi-task RL. We train a single unified model on all 130 tasks in LIBERO and evaluate its performance across the five LIBERO task suites: LIBERO-Spatial, LIBERO-Goal, LIBERO-Object, LIBERO-Long, and LIBERO-90.

For each LIBERO suite, we evaluate every combination of task_id and trial_id.
For the Object, Spatial, Goal, and Long suites, we evaluate 500 environments in total (10 tasks × 50 trials).
For LIBERO-90 and LIBERO-130, we evaluate 4,500 (90 tasks × 50 trials) and 6,500 environments respectively (130 tasks × 50 trials).

We evaluate each model according to its training configuration.
For the SFT-trained (LoRA-base) models, we set `do_sample = False`.
For the RL-trained models, we set `do_sample = True`, `temperature = 1.6`, and enable `rollout_epoch=2` to elicit the best performance of the RL-tuned policy.

.. note::

   This unified base model is fine-tuned by ourselves. For more details, please refer to paper https://arxiv.org/abs/2510.06710.

.. list-table:: **Evaluation results of the unified model on the five LIBERO task groups**
   :header-rows: 1

   * - Model
     - Object
     - Spatial
     - Goal
     - Long
     - 90
     - 130
   * - |huggingface| `OpenVLA-OFT (LoRA-base) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora>`_
     - 50.20%
     - 51.61%
     - 49.40%
     - 11.90%
     - 42.67%
     - 42.09%
   * - |huggingface| `OpenVLA-OFT (RLinf-GRPO) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130>`_
     - **99.60%**
     - **98.69%**
     - **98.09%**
     - **93.45%**
     - **98.02%**
     - **97.85%**
   * - Improvement
     - +49.40%
     - +47.08%
     - +48.69%
     - +81.55%
     - +55.35%
     - +55.76%

.. _liberopro-plus-benchmark:

LIBERO-Pro & LIBERO-Plus Benchmark
----------------------------------

This section introduces full support for the LIBERO-Pro and LIBERO-Plus evaluation suites within the RLinf framework. By incorporating more complex task scenarios and longer manipulation horizons, these suites further challenge the generalization capabilities of VLA models (such as OpenVLA-OFT).

The primary objective is to develop a model capable of performing robotic manipulation by:

1. **Visual Understanding**: Processing RGB images from robot cameras.
2. **Language Comprehension**: Interpreting natural-language task descriptions under perturbations.
3. **Action Generation**: Producing precise robotic actions (position, rotation, gripper control).
4. **Reinforcement Learning**: Optimizing the policy via PPO/GRPO with environment feedback.

Environment
~~~~~~~~~~~

**Base Simulation Setup**

* **Environment:** Simulation benchmarks built on top of robosuite (MuJoCo), heavily extending the original LIBERO suites with rigorous perturbation tests.
* **Observation:** RGB images captured by both third-person and wrist-mounted cameras.
* **Action Space:** 7-dimensional continuous actions (3D position, 3D rotation, and 1D gripper control).

**LIBERO-Pro: Anti-Memorization Perturbations**
LIBERO-Pro systematically evaluates model robustness across four orthogonal dimensions to prevent rote memorization:

* **Object Attribute Perturbations:** Modifies non-essential attributes of target objects (e.g., color, texture, size) while preserving semantic equivalence.
* **Initial Position Perturbations:** Alters the absolute and relative spatial arrangements of objects at the start of the episode.
* **Instruction Perturbations:** Introduces semantic paraphrasing (e.g., "grab" instead of "pick up") and task-level modifications (e.g., replacing the target object in the instruction).
* **Environment Perturbations:** Randomly substitutes the background workspace/scene appearance.

**LIBERO-Plus: In-depth Robustness Perturbations**
LIBERO-Plus expands the evaluation into a massive suite of 10,030 tasks across 5 difficulty levels, applying perturbations across 7 physical and semantic dimensions:

* **Objects Layout:** Injects confounding distractor objects and shifts the target object's position/pose.
* **Camera Viewpoints:** Shifts the 3rd-person camera's distance, spherical position (azimuth/elevation), and orientation.
* **Robot Initial States:** Applies random perturbations to the robot arm's initial joint angles (qpos).
* **Language Instructions:** Rewrites task instructions using LLMs to add conversational distractions, common-sense reasoning, or complex reasoning chains.
* **Light Conditions:** Alters diffuse color, light direction, specular highlights, and shadow casting.
* **Background Textures:** Modifies scene themes (e.g., brick walls) and surface materials.
* **Sensor Noise:** Simulates real-world degradation by injecting motion blur, Gaussian blur, zoom blur, fog, and glass refraction distortions.

Algorithm
~~~~~~~~~

**Core Algorithm Components**

* **PPO (Proximal Policy Optimization)**

  * Advantage estimation using GAE (Generalized Advantage Estimation).
  * Policy clipping with ratio limits.
  * Value function clipping.
  * Entropy regularization.

* **GRPO (Group Relative Policy Optimization)**

  * For every state / prompt the policy generates *G* independent actions.
  * Compute the advantage of each action by subtracting the group's mean reward.

**Vision-Language-Action Model**

* OpenVLA architecture with multimodal fusion.
* Action tokenization and de-tokenization.
* Value head for critic function.

Dependency Installation
~~~~~~~~~~~~~~~~~~~~~~~

To ensure full compatibility with the RLinf framework, please install the designated forks maintained under the RLinf organization.

1. Clone RLinf Repository
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install Dependencies
^^^^^^^^^^^^^^^^^^^^^^^

**Option 1: Docker Image**

Use Docker image for the experiment.

.. code:: bash

   # LIBERO-Pro
   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-liberopro

   # LIBERO-Plus
   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-liberoplus

**Option 2: Custom Environment**

.. code:: bash

    # For mainland China users, you can add the `--use-mirror` flag for better download speed.

    # Create an embodied environment with LIBERO-Pro support
    bash requirements/install.sh embodied --model openvla-oft --env liberopro

    # Create an embodied environment with LIBERO-Plus support
    bash requirements/install.sh embodied --model openvla-oft --env liberoplus

    # Activate the virtual environment
    source .venv/bin/activate

LIBERO-Plus Assets Download
~~~~~~~~~~~~~~~~~~~~~~~~~~~

LIBERO-Plus requires hundreds of new objects, textures, and other assets to function correctly. Download the ``assets.zip`` archive from the Hugging Face dataset ``Sylvest/LIBERO-plus`` and extract it into the installed ``liberoplus.liberoplus`` package directory.

.. code-block:: bash

    # Resolve the installed liberoplus package directory
    LIBERO_PLUS_PACKAGE_DIR=$(python -c "import pathlib; import liberoplus.liberoplus as l_plus; print(pathlib.Path(l_plus.__file__).resolve().parent)")

    # For mainland China users, you can use the following for better download speed:
    # export HF_ENDPOINT=https://hf-mirror.com

    # Download the assets archive from the Hugging Face dataset repo
    hf download --repo-type dataset Sylvest/LIBERO-plus assets.zip \
        --local-dir "${LIBERO_PLUS_PACKAGE_DIR}"

    # Extract assets in place
    unzip -o "${LIBERO_PLUS_PACKAGE_DIR}/assets.zip" -d "${LIBERO_PLUS_PACKAGE_DIR}"

After extraction, ensure your directory structure matches the following layout:

.. code-block:: text

    <installed liberoplus package dir>/
    └── assets/
        ├── articulated_objects/
        ├── new_objects/
        ├── scenes/
        ├── stable_hope_objects/
        ├── stable_scanned_objects/
        ├── textures/
        ├── turbosquid_objects/
        ├── serving_region.xml
        ├── wall_frames.stl
        └── wall.xml

Model Download
~~~~~~~~~~~~~~

For OpenVLA-OFT-based experiments on LIBERO-Pro and LIBERO-Plus, you can start from the same pretrained checkpoints used for standard LIBERO training:

.. code-block:: bash

    # Download the model (choose either method)
    # Method 1: Using git clone
    git lfs install
    git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
    git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

    # Method 2: Using huggingface-hub
    # For mainland China users, you can use the following for better download speed:
    # export HF_ENDPOINT=https://hf-mirror.com
    pip install huggingface-hub
    hf download RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora --local-dir RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
    hf download RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora --local-dir RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

After downloading, make sure to correctly specify the model path in the configuration yaml file.

.. code-block:: yaml

    rollout:
       model:
          model_path: Pathto/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
    actor:
       model:
          model_path: Pathto/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora

Running the Script
~~~~~~~~~~~~~~~~~~

**1. Configuration Files**

The LIBERO-Pro and LIBERO-Plus suites reuse the standard LIBERO config family and switch the suite through the additional ``LIBERO_TYPE`` argument:

- **OpenVLA-OFT + GRPO**: ``examples/embodiment/config/libero_10_grpo_openvlaoft.yaml``

**2. Launch Commands**

**Training**

To start training a model on the newly integrated suites, use the ``run_embodiment.sh`` script:

.. code-block:: bash

    # Train on LIBERO-Pro
    export LIBERO_TYPE=pro
    bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft

    # Train on LIBERO-Plus
    export LIBERO_TYPE=plus
    bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft

**Evaluation**

To evaluate the trained models, use the ``eval_embodiment.sh`` script:

.. code-block:: bash

    # Evaluate on LIBERO-Pro
    export LIBERO_TYPE=pro
    bash examples/embodiment/eval_embodiment.sh libero_10_grpo_openvlaoft

    # Evaluate on LIBERO-Plus
    export LIBERO_TYPE=plus
    bash examples/embodiment/eval_embodiment.sh libero_10_grpo_openvlaoft
