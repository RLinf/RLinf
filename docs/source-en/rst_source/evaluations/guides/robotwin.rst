RoboTwin Evaluation
===================

RoboTwin is a bimanual manipulation simulation platform with tasks such as placing cups, adjusting bottles, and clicking bells. RLinf supports parallel VLA policy evaluation on RoboTwin and reports metrics such as ``eval/success_once``.

Related training doc: :doc:`../../examples/embodied/robotwin`

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``.

.. tab-set::

   .. tab-item:: OpenVLA-OFT (place_empty_cup)

      .. code-block:: bash

         # 1. Install and download RoboTwin assets
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support ../RoboTwin
         (cd ../RoboTwin && bash script/_download_assets.sh)

         # 2. Download model
         hf download RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup --local-dir ./RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup

         # 3. Launch
         export ROBOTWIN_PATH="$(cd ../RoboTwin && pwd)"
         export ROBOT_PLATFORM=ALOHA
         bash evaluations/run_eval.sh robotwin robotwin_place_empty_cup_openvlaoft_eval \
           rollout.model.model_path=./RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup \
           env.eval.assets_path="${ROBOTWIN_PATH}"

   .. tab-item:: OpenPI π₀.₅ (adjust_bottle)

      .. code-block:: bash

         # 1. Install and download RoboTwin assets
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support ../RoboTwin
         (cd ../RoboTwin && bash script/_download_assets.sh)

         # 2. Download model
         hf download RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle --local-dir ./RLinf-Pi05-RoboTwin-SFT-adjust_bottle

         # 3. Launch
         export ROBOTWIN_PATH="$(cd ../RoboTwin && pwd)"
         export ROBOT_PLATFORM=ALOHA
         bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_pi05_eval \
           rollout.model.model_path=./RLinf-Pi05-RoboTwin-SFT-adjust_bottle \
           env.eval.assets_path="${ROBOTWIN_PATH}"

   .. tab-item:: OpenPI π₀ (adjust_bottle)

      .. code-block:: bash

         # 1. Install and download RoboTwin assets
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support ../RoboTwin
         (cd ../RoboTwin && bash script/_download_assets.sh)

         # 2. Download model
         hf download RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle --local-dir ./RLinf-Pi0-RoboTwin-SFT-adjust_bottle

         # 3. Launch
         export ROBOTWIN_PATH="$(cd ../RoboTwin && pwd)"
         export ROBOT_PLATFORM=ALOHA
         bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_eval \
           rollout.model.model_path=./RLinf-Pi0-RoboTwin-SFT-adjust_bottle \
           env.eval.assets_path="${ROBOTWIN_PATH}"

Supported models also include ``lingbotvla`` and ``lingbotvlav2`` — replace ``--model`` accordingly during installation. On AMD ROCm, install with ``--platform`` and set ``planner_backend: mplib`` in the task config; see :ref:`Run on Different Hardware Backends <robotwin-hardware>`.

``run_eval.sh`` adds ``ROBOTWIN_PATH`` to ``PYTHONPATH``, and ``env.eval.assets_path`` points to the parent folder of ``assets/`` (``ASSETS_PATH`` is populated automatically at env init). RLinf creates RoboTwin's cameras with a 0.01 m near plane instead of RoboTwin's 0.1 m: SAPIEN 3.0.1 clips everything closer than the near plane, and 0.1 m hides the gripper fingers from the wrist cameras. Set ``env.eval.camera_near`` to change it, or to ``null`` to keep RoboTwin's own value.

The terminal prints ``eval/success_once``; see :doc:`../reference/results` for logs and videos.

**Docker (optional)**

You can also run evaluation with the official Docker image ``rlinf/rlinf:agentic-rlinf0.4-robotwin``, which includes RoboTwin dependencies and compatibility patches. Inside the container, switch environments by model type:

- OpenVLA-OFT: ``source switch_env openvla-oft``
- OpenPI (π\ :sub:`0`\ / π\ :sub:`0.5`\ ): ``source switch_env openpi``

Example Configs
---------------

Available under ``evaluations/robotwin/``:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Config file
     - Task
     - Model
   * - ``robotwin_place_empty_cup_openvlaoft_eval.yaml``
     - place_empty_cup
     - OpenVLA-OFT
   * - ``robotwin_place_empty_cup_openpi_eval.yaml``
     - place_empty_cup
     - π₀
   * - ``robotwin_adjust_bottle_openpi_pi05_eval.yaml``
     - adjust_bottle
     - π₀.₅
   * - ``robotwin_adjust_bottle_openpi_eval.yaml``
     - adjust_bottle
     - π₀
   * - ``robotwin_place_shoe_lingbotvla_eval.yaml``
     - place_shoe
     - LingbotVLA 1.0
   * - ``robotwin_click_bell_lingbotvla_eval.yaml``
     - click_bell
     - LingbotVLA 1.0
   * - ``robotwin_click_bell_lingbotvlav2_eval.yaml``
     - click_bell
     - LingbotVLA 2.0

If ``evaluations/robotwin/<config>.yaml`` does not exist, ``run_eval.sh`` falls back to the same name under ``examples/embodiment/config/`` (set ``runner.only_eval: True`` and ``runner.task_type: embodied_eval``). The bundled evaluation seed file contains eval seeds for **22 tasks**; other tasks can be derived from training configs (see :doc:`../reference/configuration`).

.. _robotwin-eval-config:

Evaluation Configuration
------------------------

RoboTwin evaluation runs one trajectory per **success seed** in ``eval_seeds.json`` for each task and reports ``eval/success_once`` (fraction of trajectories with at least one success). The fields below are all under ``env.eval`` and together control **parallelism**, **trajectory length**, and **test-set coverage**.

Evaluation Protocol
~~~~~~~~~~~~~~~~~~~

RoboTwin evaluation uses pre-filtered **success seeds** as the random seed for each trajectory, fixing the initial scene and language instruction. The bundled ``eval_seeds.json`` file indexes seeds by ``task_name`` and covers **22 tasks** (150–320 seeds each).

In RLinf's RoboTwin environment:

- On startup, ``success_seeds`` for the task are loaded from ``seeds_path``, globally shuffled, and partitioned across workers so each env worker gets a non-overlapping subset;
- Each trajectory is uniquely determined by its assigned **seed** (initial scene and language instruction);
- When ``is_eval: True`` and ``auto_reset`` fires, completed envs receive the next seed (only when ``use_fixed_reset_state_ids: False``).

With the default example settings (``total_num_envs: 128``, ``rollout_epoch: 1``, ``use_fixed_reset_state_ids: True``), each parallel env evaluates one fixed-seed trajectory per epoch. With 8 GPUs (``component_placement: 0-7``), one epoch covers about 128 trajectories, which may not cover all seeds for the task.

Seed counts and step limits for example tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 22 22 28

   * - Task
     - Total seeds
     - Example ``max_episode_steps``
     - Example config
   * - ``adjust_bottle``
     - 150
     - 200
     - ``robotwin_adjust_bottle_openpi_eval``
   * - ``place_empty_cup``
     - 260
     - 200
     - ``robotwin_place_empty_cup_openvlaoft_eval``
   * - ``click_bell``
     - 150
     - 400
     - ``robotwin_click_bell_lingbotvla_eval``
   * - ``place_shoe``
     - 320
     - 400
     - ``robotwin_place_shoe_lingbotvla_eval``

``max_episode_steps`` should match training and ``task_config.step_lim``. LingBotVLA example tasks typically use 400 steps; OpenVLA-OFT / OpenPI examples mostly use 200.

See :ref:`env-eval-fields` in :doc:`../reference/configuration` for detailed ``env.eval`` field descriptions.

Model-specific settings
~~~~~~~~~~~~~~~~~~~~~~~

Different VLAs use different robot embodiments, cameras, and domain randomization on RoboTwin. Eval settings must match the training protocol for comparable results:

OpenVLA-OFT (demo_randomized protocol)

- Use env preset default: ``task_config.embodiment: [piper, piper, 0.6]``
- ``center_crop: True``; set ``rollout.model.center_crop: True`` on the model side
- Keep domain randomization enabled (training preset default)
- ``rollout.model.num_action_chunks: 25``; ``unnorm_key`` must match SFT, e.g. ``place_empty_cup_1k``
- ``rollout.model.implement_version: "official"``

OpenPI (π\ :sub:`0`\ / π\ :sub:`0.5`\ , demo_clean protocol)

- ``task_config.embodiment: [aloha-agilex]``
- ``center_crop: False``
- ``task_config.camera.collect_wrist_camera: true``
- Disable all ``task_config.domain_randomization`` fields: set ``random_background``, ``cluttered_table``, ``random_light``, etc. to ``false``
- ``rollout.model.num_action_chunks: 50``
- ``rollout.model.openpi.config_name``: ``pi0_aloha_robotwin`` or ``pi05_aloha_robotwin``
- Recommend ``env.enable_offload: True`` and ``rollout.enable_offload: True`` to reduce GPU memory use

LingbotVLA 1.0

- Besides ``rollout.model.model_path``, also set ``tokenizer_path`` and ``rollout.model.lingbotvla.config_path``
- ``rollout.model.num_action_chunks: 50``; ``max_episode_steps: 400`` (e.g. ``click_bell``, ``place_shoe``)
- ``use_custom_reward: False`` (disable custom reward during evaluation)

LingbotVLA 2.0

- Fill in the paths in ``examples/embodiment/config/model/lingbotvlav2.yaml``, which the eval config shares with training
- ``rollout.model.precision: bf16``; fp32 weights belong only to the PPO actor
- ``task_config.embodiment: [aloha-agilex]`` and ``task_config.camera.collect_wrist_camera: true``; the model reads both wrist cameras
- ``rollout.model.num_action_chunks: 50``; ``max_episode_steps: 400``

Covering the full test set
~~~~~~~~~~~~~~~~~~~~~~~~~~

Let ``S`` be the total number of seeds for a task, ``E`` the number of parallel envs, and ``T`` the per-trajectory step limit (``max_episode_steps``).

**Option 1: High parallelism**

.. code-block:: yaml

   env:
     eval:
       total_num_envs: 260        # S, e.g. place_empty_cup
       max_episode_steps: 200
       max_steps_per_rollout_epoch: 200   # equals max_episode_steps
       use_fixed_reset_state_ids: True
       rollout_epoch: 1

**Option 2: Dynamic seeds with auto reset (recommended when resources are limited)**

.. code-block:: yaml

   env:
     eval:
       total_num_envs: 128
       max_episode_steps: 200
       # N = ceil(S / E); place_empty_cup: ceil(260/128) = 3
       max_steps_per_rollout_epoch: 600   # N * max_episode_steps = 3 * 200
       auto_reset: True
       ignore_terminations: True
       use_fixed_reset_state_ids: False   # allow seed rotation on auto_reset
       is_eval: True
       rollout_epoch: 1

**Multi-epoch averaging**

.. code-block:: yaml

   env:
     eval:
       rollout_epoch: 2
       use_fixed_reset_state_ids: False   # required when rollout_epoch > 1

Notes
~~~~~

- ``max_steps_per_rollout_epoch`` must be divisible by ``rollout.model.num_action_chunks``, or startup validation will fail.
- ``env.eval.seeds_path`` defaults to ``eval_seeds.json``; custom seed files must include a ``success_seeds`` list for the target ``task_name``.
- OpenVLA-OFT is trained/evaluated under **demo_randomized**; OpenPI under **demo_clean**. Mixing domain randomization settings makes metrics incomparable.
- These tasks are not yet supported: ``place_fan``, ``open_laptop``, ``place_object_scale``, ``put_object_cabinet``.

Advanced Usage
--------------

**Adjust parallelism**

.. code-block:: bash

   bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_eval \
     env.eval.total_num_envs=64 \
     rollout.model.model_path=/path/to/model

**Derive configs for other tasks**

``eval_seeds.json`` also lists seeds for ``beat_block_hammer``, ``handover_block``, ``lift_pot``, and more. Copy a structurally similar YAML from ``evaluations/robotwin/``, change the env preset in ``defaults`` to ``env/robotwin_<task>@env.eval`` (under ``examples/embodiment/config/env/``), and adjust ``rollout.model`` fields such as ``unnorm_key`` or ``openpi.config_name``.

**Load an RL checkpoint**

.. code-block:: bash

   bash evaluations/run_eval.sh robotwin robotwin_place_empty_cup_openvlaoft_eval \
     runner.ckpt_path=/path/to/checkpoint.pt

FAQ
---

- **ROBOTWIN_PATH not set:** Export ``ROBOTWIN_PATH`` to a RoboTwin repo root (``RLinf_support`` branch). ``run_eval.sh`` adds it to ``PYTHONPATH`` only when it is set.
- **Wrong assets_path:** The env loads assets via ``ASSETS_PATH``; an invalid path causes startup failure or missing scenes.
- **Robot platform:** Set ``ROBOT_PLATFORM=ALOHA`` to select the platform variant.
- **GPU OOM:** Set ``env.enable_offload: True`` and ``rollout.enable_offload: True`` in the YAML, or reduce ``env.eval.total_num_envs``.
- **Eval coverage:** See :ref:`robotwin-eval-config` above; default 128 parallel envs with ``use_fixed_reset_state_ids: True`` only covers a subset of seeds.
- **Rendering issues:** ``run_eval.sh`` renders with EGL by default; on hosts without EGL, ``export MUJOCO_GL=osmesa`` and ``export PYOPENGL_PLATFORM=osmesa``.
