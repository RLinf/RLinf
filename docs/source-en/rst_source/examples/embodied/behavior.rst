RL with Behavior Benchmark
==========================

This example provides a complete guide to fine-tuning the 
Behavior algorithms with reinforcement learning in the `Behavior <https://behavior.stanford.edu/index.html>`_ environment
using the **RLinf** framework. It covers the entire process—from
environment setup and core algorithm design to training configuration,
evaluation, and visualization—along with reproducible commands and
configuration snippets.

The primary objective is to develop a model capable of performing
robotic manipulation by:

1. **Visual Understanding**: Processing RGB images from the robot's
   camera.
2. **Language Comprehension**: Interpreting natural-language task
   descriptions.
3. **Action Generation**: Producing precise robotic actions (position,
   rotation, gripper control).
4. **Reinforcement Learning**: Optimizing the policy via the PPO with
   environment feedback.

--------------

Environment
-----------

**Behavior Environment**

- **Environment**: Behavior simulation benchmark built on top of *IsaacSim*.
- **Task**: Command a dual-arm R1 Pro robot to perform a variety of household manipulation skills (pick-and-place, stacking, opening drawers, spatial rearrangement).
- **Observation**: Multi-camera RGB images captured by robot-mounted sensors:
  - **Head Camera**: head camera providing 224×224 RGB images for global scene understanding
  - **Wrist Cameras**: Left and right RealSense cameras providing 224×224 RGB images for precise manipulation
- **Action Space**: 23-dimensional continuous actions (a 3-DOF (x,y,rz) set of joints, 4-DOF torso, x2 7-DOF arm, and x2 1-DOF parallel jaw grippers.)

**Data Structure**

- **Task_descriptions**: select from `behavoir-1k` tasks
- **Images**: Multi-camera RGB tensors
  - Head images: ``[batch_size, 224, 224, 3]``
  - Wrist images: ``[batch_size, 2, 224, 224, 3]`` (left and right cameras)


Algorithm
---------

**Core Algorithm Components**

1. **PPO (Proximal Policy Optimization)**

   - Advantage estimation using GAE (Generalized Advantage Estimation)

   - Policy clipping with ratio limits

   - Value function clipping

   - Entropy regularization

2. **GRPO (Group Relative Policy Optimization)**

   - For every state / prompt the policy generates *G* independent actions

   - Compute the advantage of each action by subtracting the group’s mean reward.

Dependency Installation
------------------------

.. warning::

   Please refer to the following ISAAC-SIM software and hardware dependency documentation to ensure your environment meets the requirements.

   https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html

   https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html

   In particular, if your GPU is of Hopper architecture or above, please follow the instructions for NVIDIA driver version 570 or above.

   Additionally, if your GPU lacks Ray Tracing capabilities (e.g., A100, H100), the rendering quality of BEHAVIOR will be very poor, and the visuals may suffer from severe artifacts or blurriness.

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
      rlinf/rlinf:agentic-rlinf0.2-behavior
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-behavior

**Option 2: Custom Environment**

Install dependencies directly in your environment by running the following command:

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.

   # Install openvla-oft environment
   bash requirements/install.sh embodied --model openvla-oft --env behavior
   source .venv/bin/activate

   # Install openpi environment
   bash requirements/install.sh embodied --model openpi --env behavior
   source .venv/bin/activate

Assets Download
-----------------

* ISAAC-SIM 4.5 Download

.. warning::

   The `ISAAC_PATH` environment variable must be set every time you run the experiment.

.. code:: bash

   export ISAAC_PATH=/path/to/isaac-sim
   mkdir -p $ISAAC_PATH && cd $ISAAC_PATH
   curl https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip -o isaac-sim.zip
   unzip isaac-sim.zip && rm isaac-sim.zip

* BEHAVIOR Datasets and Assets Download

.. warning::

   The `OMNIGIBSON_DATA_PATH` environment variable must be set every time you run the experiment.

.. code:: bash

   # Change to the directory you wish to put the assets and datasets
   # Beware, the datasets occupy more than 30GB of space
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   mkdir -p $OMNIGIBSON_DATA_PATH

   # Make sure you are inside the correct Python virtual environment (venv) before running below commands
   # For our Docker image, you need to switch to the `openvla-oft` venv via `source switch_env openvla-oft`
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
   python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)" 
   python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"


Model Download
---------------

Before starting training, you need to download the corresponding pretrained models. Based on the algorithm type you want to use, we provide different model options:

**OpenVLA-OFT Model Download**

OpenVLA-OFT provides a unified model that is suitable for all task types in the Behavior environment.

.. code:: bash

   # Download the model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-Behavior

   # Method 2: Using huggingface-hub
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-Behavior --local-dir RLinf-OpenVLAOFT-Behavior

**OpenPI Model Download**

.. code:: bash

   # Download the model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-Behavior

   # Method 2: Using huggingface-hub
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-Behavior --local-dir RLinf-Pi0-Behavior


After downloading, please make sure to specify the model path correctly in your configuration yaml file.

Running Scripts
---------------

**1. Key Cluster Configuration**

.. warning::

   Beware, due to the special behavior of ISAAC-SIM, please try to place the env on GPUs starting from 0.
   Otherwise, ISAAC-SIM may get stuck on certain GPUs.

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-3
         rollout: 4-7
         actor: 0-7

   rollout:
      pipeline_stage_num: 2

Here you can flexibly configure the GPU count for env, rollout, and
actor components.
Additionally, by setting ``pipeline_stage_num = 2`` in the
configuration, you can achieve pipeline overlap between rollout and
env, improving rollout efficiency.

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

You can also reconfigure the placement to achieve complete sharing,
where env, rollout, and actor components all share all GPUs.

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-1
         rollout: 2-5
         actor: 6-7

You can also reconfigure the placement to achieve complete separation,
where env, rollout, and actor components each use their own GPUs without
interference, eliminating the need for offload functionality.

--------------

**2. Configuration Files**

Using behavior as an example:

- OpenVLA-OFT + PPO:
  ``examples/embodiment/config/behavior_ppo_openvlaoft.yaml``
- OpenVLA-OFT + GRPO:
  ``examples/embodiment/config/behavior_grpo_openvlaoft.yaml``
- OpenPI (Pi0) + PPO:
  ``examples/embodiment/config/behavior_ppo_openpi.yaml``
- OpenPI (Pi0.5) + PPO:
  ``examples/embodiment/config/behavior_ppo_openpi_pi05.yaml``

.. warning::

   Known issue: under the current Behavior setup, training success rate
   (``env/success_once``) may stay at 0 for ``OpenVLA-OFT`` / ``OpenPI (Pi0)``.
   This issue will be fixed in a later release.

.. note::

   The Behavior configs above all load
   ``examples/embodiment/config/env/behavior_r1pro.yaml`` via ``defaults``
   (for both ``env.train`` and ``env.eval``). This file defines the base R1 Pro
   environment settings, including ``task_idx``, ``max_episode_steps``,
   ``max_steps_per_rollout_epoch``, camera resolution, and ``omni_config``.
   You can override these defaults in each concrete config under
   ``env.train`` / ``env.eval``.

**Key Settings in behavior_r1pro.yaml**

- ``base_config_name: r1pro_behavior``:
  RLinf first loads OmniGibson's base ``r1pro_behavior.yaml`` and then applies
  overrides from ``omni_config`` (see ``setup_omni_cfg`` in
  ``rlinf/envs/behavior/utils.py``).
- ``omni_config.task.type: RLinfBehaviorTask`` and
  ``omni_config.scene.type: RLinfInteractiveTraversableScene``:
  RLinf ships a lightweight BEHAVIOR compatibility patch for
  ``omnigibson==3.7.1``. Keep these two types in
  ``examples/embodiment/config/env/behavior_r1pro.yaml`` when using RLinf's
  BEHAVIOR setup. ``install_patch()`` is still called automatically by
  ``rlinf/envs/behavior/behavior_env.py`` before ``VectorEnvironment`` is
  created, but it only registers the RLinf classes and applies monkey patches.
  It does not rewrite ``task.type`` or ``scene.type`` anymore, so these two
  YAML entries must be set explicitly.
- RLinf BEHAVIOR patch contents:
  The patch fixes several multi-environment issues observed with
  OmniGibson 3.7.1, including cross-scene ``BehaviorTask`` callbacks,
  presampled robot poses being applied in world frame instead of scene frame,
  cross-scene shared control views for the same robot type, and RLinf's
  missing ``scene`` sub-config override in ``setup_omni_cfg``.
- Version note:
  The current patch is only tested and supported on ``omnigibson==3.7.1``.
  RLinf raises an error during environment initialization if a different
  OmniGibson version is detected.
- ``task_idx``:
  Current task id (0-49). RLinf maps it to the concrete task name and writes it
  into ``task.activity_name`` (see ``rlinf/envs/behavior/behavior_env.py``).
- ``omni_config.task.instance_resample_mode``:
  Controls reset-time instance switching. Supported modes are
  ``disabled``, ``offline``, and ``online``.
  In ``offline`` mode, RLinf scans ``omni_config.task.activity_instance_dir``
  once at startup, parses cached instance ids from
  filenames in that directory, and samples one cached offline instance before
  each ``env.reset()``. ``*_template.json`` files are treated as full cached
  templates and are reloaded through the heavier scene-reload path, while
  ``*_template-tro_state.json`` files are treated as task-relevant-only cached
  states and are applied through the lighter in-place path. This is useful
  when you want more reset-time diversity than a fixed
  ``activity_instance_id`` but lower overhead than ``online_object_sampling``.
  In ``online`` mode, RLinf reuses the online task-resampling path and requires
  ``online_object_sampling: True`` plus ``use_presampled_robot_pose: False``.
  In ``disabled`` mode, if ``activity_instance_dir`` is set RLinf loads the
  configured ``activity_instance_id`` from that directory before each reset.
- ``omni_config.task.activity_instance_dir``:
  Optional directory containing cached task instance JSON files. RLinf
  recognizes official ``*_template.json`` instances and
  ``*_template-tro_state.json`` files. Used by
  ``instance_resample_mode: offline`` and by fixed ``activity_instance_id``
  loading when the mode is ``disabled``.
- ``omni_config.task.instance_file_format``:
  Optional cached-instance format selector. Supported values are ``template``
  and ``tro_state``. Use ``template`` to force full cached-template reloads, or
  ``tro_state`` to force light-weight task-relevant-only reloads. RLinf also
  accepts official ``tro_state`` files that do not include ``robot_poses``; in
  that case, RLinf clears any stale cached robot-pose metadata and the
  subsequent reset uses the task's default robot reset pose instead of a
  presampled pose override. When converting from
  ``template.json``, omitting ``robot_poses`` is usually safer than writing the
  current simulator robot pose into the cache.
- Generating cached instances with RLinf's generator:
  RLinf provides ``rlinf/envs/behavior/instance_generator.py`` to
  generate ``*_template.json`` and ``*_template-tro_state.json`` files directly
  from ``examples/embodiment/config/env/behavior_r1pro.yaml``.
  The script reads ``omni_config.scene.scene_model``,
  ``omni_config.task.activity_name``,
  ``omni_config.task.activity_definition_id``, the robot config, and room
  loading settings from the yaml, then temporarily switches the task to online
  object sampling for cached-instance generation.
  It writes into ``omni_config.task.activity_instance_dir`` when that field is
  set; otherwise it falls back to ``OMNIGIBSON_DATA_PATH``'s default
  ``2025-challenge-task-instances`` directory. Use ``--output-dir`` to override
  either behavior.

  .. code-block:: bash

     cd /path/to/RLinf

     python rlinf/envs/behavior/instance_generator.py \
       --config examples/embodiment/config/env/behavior_r1pro.yaml \
       --output-format template \
       --start-idx 1 \
       --end-idx 50

     python rlinf/envs/behavior/instance_generator.py \
       --config examples/embodiment/config/env/behavior_r1pro.yaml \
       --output-format tro_state \
       --start-idx 1 \
       --end-idx 50

  The generated filenames follow
  ``<scene_model>_task_<activity_name>_<activity_definition_id>_<activity_instance_id>_template(.json|-tro_state.json)``.
  ``--start-idx`` and ``--end-idx`` therefore control the generated
  ``activity_instance_id`` range. ``tro_state`` outputs include top-level
  ``robot_poses`` when the task metadata provides them; otherwise the key is
  omitted so RLinf reset falls back to the task's default robot reset pose.
  BEHAVIOR-1K's upstream
  ``OmniGibson/omnigibson/sampling/multiply_b1k_tasks.py`` is still usable, but
  RLinf's generator is the recommended path because it reads the RLinf yaml
  directly and preserves ``activity_definition_id`` from that config.
- ``camera.head_resolution`` / ``camera.wrist_resolution``:
  Head / wrist camera resolutions. RLinf overrides default values in
  ``omnigibson.learning.utils.eval_utils`` (default 720x720 and 480x480), then
  applies them through the environment wrapper to R1Pro sensors.
- ``omni_config.env.action_frequency / rendering_frequency / physics_frequency``:
  Controls action stepping, rendering, and physics frequency respectively
  (common default: 30 / 30 / 120). Higher frequencies are usually slower.
- ``omni_config.env.automatic_reset: False``:
  Do not auto-reset when an episode ends; reset is explicitly controlled by the
  RLinf training / evaluation loop.
- ``omni_config.env.flatten_obs_space: False`` and ``flatten_action_space: False``:
  Keep structured observation / action spaces instead of flattening to 1D.
- ``omni_config.macro.use_gpu_dynamics: False``:
  Disables GPU dynamics and usually improves performance; enable it only when
  advanced features like particles / fluids are required.
- ``omni_config.macro.enable_flatcache: True``:
  Enables flatcache, which generally improves performance for large scenes.
- ``omni_config.macro.enable_object_states: True``:
  BehaviorTask depends on object states, so this should stay enabled.
- ``omni_config.macro.enable_transition_rules: True``:
  Enables transition-rule-based state changes (e.g., slicing, cooking-related
  transitions).
- ``omni_config.macro.use_numpy_controller_backend: True``:
  Uses the numpy controller backend, which is usually faster in single-process
  or moderate-parallel settings.

--------------

**3. Launch Command**

To start training with a chosen configuration, run the following
command:

.. code:: bash

   export ISAAC_PATH=/path/to/isaac-sim
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

For example, to train the OpenVLA-OFT model using the PPO algorithm in
the Behavior environment, run:

.. code:: bash

   export ISAAC_PATH=/path/to/isaac-sim
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   bash examples/embodiment/run_embodiment.sh behavior_ppo_openvlaoft

--------------

**4. Evaluate with behavior_ppo_openpi_pi05.yaml**

In principle, any ``pi05`` checkpoint that has non-zero success rate on
Behavior and has been converted to PyTorch format can be used for evaluation
with this config. We use OpenPI-Comet only as an example source:

- https://huggingface.co/sunshk/openpi_comet/tree/main

After download, you can use the following repository to convert weights to
PyTorch format:

- https://github.com/mli0603/openpi-comet

Thanks to the OpenPI-Comet authors for open-sourcing the model and tools, which
helps reproducibility and evaluation in RLinf.

After conversion, update ``behavior_ppo_openpi_pi05.yaml`` as follows:

1. Set ``actor.model.model_path`` and ``rollout.model.model_path`` to the converted model directory.
2. Increase ``max_episode_steps`` and ``max_steps_per_rollout_epoch`` in both
   ``env.train`` and ``env.eval`` (for example, ``4096``).

.. code:: yaml

   env:
     train:
       max_episode_steps: 4096
       max_steps_per_rollout_epoch: 4096
     eval:
       max_episode_steps: 4096
       max_steps_per_rollout_epoch: 4096

Run evaluation with:

.. code:: bash

   export ISAAC_PATH=/path/to/isaac-sim
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05


Visualization and Results
-------------------------

**1. TensorBoard Logging**

.. code:: bash

   # Launch TensorBoard
   tensorboard --logdir ./logs --port 6006

--------------

**2. Key Monitoring Metrics**

-  **Training Metrics**

   -  ``actor/loss``: Policy loss
   -  ``actor/value_loss``: Value function loss (PPO)
   -  ``actor/grad_norm``: Gradient norm
   -  ``actor/approx_kl``: KL divergence between old and new policies
   -  ``actor/pg_clipfrac``: Policy clipping ratio
   -  ``actor/value_clip_ratio``: Value loss clipping ratio (PPO)

-  **Rollout Metrics**

   -  ``rollout/returns_mean``: Average episode return
   -  ``rollout/advantages_mean``: Mean advantage value

-  **Environment Metrics**

   -  ``env/episode_len``: Average episode length
   -  ``env/success_once``: Task success rate

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
       experiment_name: "behavior_ppo_openvlaoft"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab


For the Behavior experiment, we were inspired by 
`Behavior-1K baselines <https://github.com/StanfordVL/b1k-baselines.git>`_, 
with only minor modifications. We thank the authors for releasing their open-source code.
