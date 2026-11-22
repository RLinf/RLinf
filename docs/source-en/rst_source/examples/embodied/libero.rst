LIBERO RL
=========

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://libero-project.github.io/assets/img/libero/fig1.png
   :align: center
   :width: 90%

   An overview of the LIBERO benchmark (image: `LIBERO project <https://libero-project.github.io>`__).

Use this page to install and configure LIBERO suites for VLA training and
evaluation in RLinf. A 7-DoF Franka arm performs language-conditioned
manipulation in `robosuite <https://robosuite.ai>`__ / MuJoCo. Choose the
suite below, then follow its installation and configuration steps:

- :ref:`Standard LIBERO suites <libero-benchmark>` — train OpenVLA-OFT and other VLAs with PPO/GRPO.
- :ref:`LIBERO-Pro / LIBERO-Plus <liberopro-plus-benchmark>` — harder suites that stress generalization with anti-memorization perturbations.
- :ref:`LIBERO-Mem <libero-mem-benchmark>` — ordered manipulation tasks with explicit observation history for OpenPI short memory.

For backend setup, open the model instructions: :ref:`OpenVLA-OFT on AMD / Ascend <openvla-oft-hardware>`, :ref:`GR00T N1.5 on Ascend <gr00t-hardware>`, or :ref:`π₀.₅ on Moore Threads MUSA <pi0-hardware>`. These combinations cover standard LIBERO suites; LIBERO-Pro, LIBERO-Plus, and LIBERO-Mem require separate validation.

Overview
--------

RL-finetune a VLA on the LIBERO suites; OpenVLA-OFT + GRPO reaches ~98–99% success on standard LIBERO.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      OpenVLA-OFT · π₀ / π₀.₅ · GR00T · Dexbotic · ABot-M0 · StarVLA · MLP

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO · GRPO · DSRL · DAgger

   .. grid-item-card:: Tasks
      :text-align: center

      130 across 5 suites + Pro / Plus / Mem

   .. grid-item-card:: Hardware
      :text-align: center

      NVIDIA CUDA · AMD ROCm · Huawei Ascend CANN · Moore Threads MUSA

| **You'll do:** install deps → download the base model → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · a downloaded base checkpoint (steps below).

Tasks
~~~~~

Standard LIBERO ships **five task suites covering 130 tasks**, from single-step pick-and-place
to long-horizon, multi-step scenarios. Pick a suite through the config name; ``libero_130``
trains one unified policy across all of them.

.. list-table::
   :header-rows: 1
   :widths: 24 20 10 46

   * - Suite
     - Config id
     - Tasks
     - Focus
   * - LIBERO-Spatial
     - ``libero_spatial``
     - 10
     - Same objects, different spatial arrangements — tests spatial reasoning.
   * - LIBERO-Object
     - ``libero_object``
     - 10
     - Same layout, different objects — tests object grounding.
   * - LIBERO-Goal
     - ``libero_goal``
     - 10
     - Same objects and layout, different goals — tests goal conditioning.
   * - LIBERO-Long
     - ``libero_10``
     - 10
     - Long-horizon, multi-step tasks from LIBERO-100.
   * - LIBERO-90
     - ``libero_90``
     - 90
     - Short-horizon tasks from LIBERO-100.
   * - LIBERO-130
     - ``libero_130``
     - 130
     - All suites combined, for large-scale multi-task RL.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - RGB from a third-person (agentview) and a wrist camera — typically 128×128 or 224×224 — plus 8-dim proprioception (end-effector pose and gripper).
   * - Action
     - 7-dim continuous, ``Box(-1, 1)``: a 6-DoF end-effector delta (3D position + 3D rotation) and 1-D gripper open/close.
   * - Reward
     - Sparse — ``0`` on every step and ``1`` only when the task succeeds at episode termination.
   * - Task prompt
     - ``In: What action should the robot take to [task_description]? Out:``

.. _libero-benchmark:

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: Standard LIBERO
      :sync: libero-standard

      The walkthrough below uses **OpenVLA-OFT** with **PPO/GRPO**; switch the config to use another supported model.

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero && source .venv/bin/activate

      **2. Download**

      Download the pretrained base checkpoint for your task suite (for LIBERO-90 only, use ``RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora`` instead):

      .. code-block:: bash

         hf download RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora --local-dir ./RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

      **3. Launch**

      Available OpenVLA-OFT + GRPO configs under ``examples/embodiment/config/``:

      .. list-table::
         :header-rows: 1
         :widths: 30 70

         * - Suite
           - Config
         * - LIBERO-Long
           - ``libero_10_grpo_openvlaoft.yaml``
         * - LIBERO-90
           - ``libero_90_grpo_openvlaoft.yaml``
         * - LIBERO-130
           - ``libero_130_grpo_openvlaoft.yaml``

      Launch training with ``run_embodiment.sh``, passing ``actor.model.model_path`` on the command line (rollout inherits it automatically):

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft \
           actor.model.model_path=./RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

   .. tab-item:: LIBERO-Pro
      :sync: libero-pro

      .. _liberopro-plus-benchmark:

      LIBERO-Pro shares the robosuite/MuJoCo setup and 7-DoF action space of standard LIBERO while applying four orthogonal anti-memorization perturbations:

      .. list-table::
         :header-rows: 1
         :widths: 26 74

         * - Perturbation
           - What it changes
         * - Object attributes
           - Non-essential attributes of target objects (color, texture, size), preserving semantics.
         * - Initial positions
           - Absolute and relative spatial arrangements of objects at episode start.
         * - Instructions
           - Semantic paraphrasing (e.g. "grab" vs "pick up") and target-object swaps.
         * - Environment
           - Background workspace / scene appearance.

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env liberopro && source .venv/bin/activate

      **2. Download**

      Download a standard LIBERO base checkpoint (for LIBERO-90 only, use ``RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora`` instead):

      .. code-block:: bash

         hf download RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora --local-dir ./RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

      **3. Launch**

      LIBERO-Pro reuses the standard LIBERO configs and selects the suite via ``LIBERO_TYPE=pro``. Launch training with ``run_embodiment.sh`` (for standalone evaluation such as ``libero_10_openvlaoft_eval``, see :doc:`LIBERO evaluation guide <../../evaluations/guides/libero>`):

      .. code-block:: bash

         export LIBERO_TYPE=pro
         bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft \
           actor.model.model_path=./RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

   .. tab-item:: LIBERO-Plus
      :sync: libero-plus

      LIBERO-Plus expands to **10,030 tasks across 5 difficulty levels**, perturbing seven physical and semantic dimensions:

      .. list-table::
         :header-rows: 1
         :widths: 26 74

         * - Perturbation
           - What it changes
         * - Objects layout
           - Injects distractor objects and shifts the target's position/pose.
         * - Camera viewpoints
           - Third-person camera distance, spherical position (azimuth/elevation), and orientation.
         * - Robot initial states
           - Random perturbations to the arm's initial joint angles (qpos).
         * - Language instructions
           - LLM rewrites adding conversational distractions, common-sense or complex reasoning.
         * - Light conditions
           - Diffuse color, light direction, specular highlights, and shadow casting.
         * - Background textures
           - Scene themes (e.g. brick walls) and surface materials.
         * - Sensor noise
           - Motion/Gaussian/zoom blur, fog, and glass-refraction distortions.

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env liberoplus && source .venv/bin/activate

      **2. Download**

      Extract ``assets.zip`` from ``Sylvest/LIBERO-plus`` into the installed ``liberoplus.liberoplus`` package and download a standard LIBERO base checkpoint:

      .. code-block:: bash

         LIBERO_PLUS_PACKAGE_DIR=$(python -c "import pathlib; import liberoplus.liberoplus as l_plus; print(pathlib.Path(l_plus.__file__).resolve().parent)")
         hf download --repo-type dataset Sylvest/LIBERO-plus assets.zip --local-dir "${LIBERO_PLUS_PACKAGE_DIR}"
         unzip -o "${LIBERO_PLUS_PACKAGE_DIR}/assets.zip" -d "${LIBERO_PLUS_PACKAGE_DIR}"
         hf download RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora --local-dir ./RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

      **3. Launch**

      LIBERO-Plus reuses the standard LIBERO configs and selects the suite via ``LIBERO_TYPE=plus``. Launch training with ``run_embodiment.sh`` (for standalone evaluation such as ``libero_10_openvlaoft_eval``, see :doc:`LIBERO evaluation guide <../../evaluations/guides/libero>`):

      .. code-block:: bash

         export LIBERO_TYPE=plus
         bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft \
           actor.model.model_path=./RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

   .. tab-item:: LIBERO-Mem
      :sync: libero-mem

      .. _libero-mem-benchmark:

      Use LIBERO-Mem to evaluate 10 ordered-subgoal tasks that require remembering previous interactions. It reuses RLinf's ``libero`` environment, camera observations, and 7-dimensional action space, with a 700-control-step episode limit for OpenPI. LIBERO-Mem reads task instructions from BDDL and vertically flips both cameras to match its demonstrations, while standard LIBERO keeps its existing camera convention.

      **1. Install**

      Install the OpenPI environment and replace the standard LIBERO package with the pinned `LIBERO-Mem fork <https://github.com/libero-mem/libero-mem>`__ (both share the ``libero.libero`` namespace). Answer ``N`` when the final command prompts for a dataset path, and keep ``LIBERO_CONFIG_PATH`` exported in the launch shell:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env libero && source .venv/bin/activate
         git clone https://github.com/libero-mem/libero-mem.git ../LIBERO-Mem
         git -C ../LIBERO-Mem checkout 0eee0defa4024e51511b6e15a79f30e9620209de
         uv pip uninstall rlinf-libero
         uv pip install --no-deps -e ../LIBERO-Mem
         export LIBERO_CONFIG_PATH="$PWD/.libero-mem"
         python -c "import libero.libero"

      **2. Copy Initial States**

      Copy the bundled 200 initial states (10 tasks × 20 states from demonstrations 101–120 in `LIBERO-Mem-Raw <https://huggingface.co/datasets/libero-mem/LIBERO-Mem-Raw>`__, stored under ``evaluations/libero/init_states/libero_mem/``) into the LIBERO-Mem checkout:

      .. code-block:: bash

         mkdir -p ../LIBERO-Mem/libero/libero/init_files/libero_mem
         cp evaluations/libero/init_states/libero_mem/*.pruned_init ../LIBERO-Mem/libero/libero/init_files/libero_mem/

      **3. Launch**

      On PyTorch 2.6 and later, export ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`` in the LIBERO-Mem launch shell because the pinned fork calls ``torch.load`` without specifying ``weights_only``.

      The shared config ``examples/embodiment/config/env/libero_mem.yaml`` selects the memory suite:

      .. code-block:: yaml

         env_type: libero
         task_suite_name: libero_mem
         max_episode_steps: 700
         max_steps_per_rollout_epoch: 700
         short_memory:
           num_frames: 8
           stride: 70
           fps: 20.0

      For OpenPI evaluation, start from ``evaluations/libero/libero_mem_openpi_pi05_eval.yaml`` (which sets ``ignore_terminations: True`` to run until the episode limit and report the final state). See :doc:`OpenPI Short Memory </rst_source/examples/embodied/pi0/short_memory>` for checkpoint paths and launch commands:

      .. code-block:: bash

         export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
         bash evaluations/run_eval.sh libero libero_mem_openpi_pi05_eval

      .. dropdown:: Observation history and success metrics

         The environment records both cameras after every control step, including each step inside an action chunk. ``num_frames`` includes the current frame, ``stride`` counts control steps between sampled frames, and ``fps`` converts step offsets to seconds (must match the control frequency). With the defaults, eight frames span up to 490 steps (24.5 seconds). History runs from oldest to current, left-pads with zero images marked invalid before enough frames exist, and resets per environment. Keep ``skip_intermediate_renders: False`` when history is enabled.

         The observation groups past frames under ``history`` (``B`` = environments, ``K`` = ``num_frames``):

         .. list-table::
            :header-rows: 1
            :widths: 35 20 45

            * - Field
              - Shape
              - Meaning
            * - ``history.main_images``
              - ``[B, K, H, W, 3]``
              - Third-person RGB frames.
            * - ``history.wrist_images``
              - ``[B, K, H, W, 3]``
              - Wrist RGB frames.
            * - ``history.frame_mask``
              - ``[B, K]``
              - ``True`` for observed frames; ``False`` for padding.
            * - ``history.time_offsets``
              - ``[B, K]``
              - Seconds relative to the current frame, whose offset is zero. Padding also uses zero and is excluded by the mask.

         **Interpreting success:** completing the ordered sequence and retaining the final goal are both required. ``strict_success`` reports that condition at the current step (including the fork's overshoot check), and per-task evaluation statistics use ``strict_success`` at episode completion (``success_at_end`` records the same condition). ``success_once`` stays latched after any successful step even if the final condition is later lost. For diagnostics, inspect ``subgoals_completed``, ``subgoals_total``, ``subgoal_progress`` (completed / total), and ``overshot`` in episode info; progress advances once per control step and resets on episode reset (reset settling steps do not count).

**Other Supported Models on LIBERO:** :doc:`OpenPI π₀ / π₀.₅ <pi0>` · :doc:`GR00T <gr00t>` · :doc:`Dexbotic <dexbotic>` · :doc:`ABot-M0 <abot_m0>` · :doc:`StarVLA <starvla>`

.. admonition:: Configure further
   :class: note

   - Placement and throughput → :doc:`Placement <../../concepts/placement>` and :doc:`Execution modes <../../concepts/execution_modes>`
   - Hiding inference latency during evaluation → :doc:`RTC <../../guides/rtc>`
   - All config keys → :doc:`Configuration <../../guides/index>`
   - Metric definitions and logging backends → :doc:`Training metrics <../../reference/metrics>`
   - Resuming from a checkpoint → :doc:`Resume <../../guides/resume>`
   - Stuck or hitting OOM? → :doc:`FAQ <../../resources/faq>`

Visualization and Results
-------------------------

Watch ``env/success_once`` in TensorBoard (``tensorboard --logdir ./logs``); set ``env.eval.video_cfg.save_video=True`` to record videos, or see :doc:`Training metrics <../../reference/metrics>` for W&B/SwanLab and all metrics.

LIBERO Results
~~~~~~~~~~~~~~

To show RLinf's large-scale multi-task RL, we train a single unified model on all 130
LIBERO tasks and evaluate across the five suites. We evaluate every ``task_id`` × ``trial_id``
combination: 500 environments each for Object/Spatial/Goal/Long (10 tasks × 50 trials),
4,500 for LIBERO-90, and 6,500 for LIBERO-130. SFT (LoRA-base) models use ``do_sample = False``;
RL models use ``do_sample = True`` and ``temperature_train = 1.6`` in ``rollout.sampling_params``, and ``env.train.rollout_epoch = 2``.

.. note::

   This unified base model is fine-tuned by ourselves. For details, see the `paper <https://arxiv.org/abs/2510.06710>`__.

.. list-table:: **Unified model evaluated on the five LIBERO task groups**
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
