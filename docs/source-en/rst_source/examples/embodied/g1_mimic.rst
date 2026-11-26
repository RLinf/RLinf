G1 Motion Tracking Environment
==============================

Train and evaluate whole-body motion-tracking policies for the 29-DoF Unitree G1 humanoid in MuJoCo Warp (via mjlab, ``env_type: g1_mimic``). The environment runs batched GPU physics for LAFAN1 or custom clip tracking and standing push recovery, and pairs with the :doc:`SuperMimic <supermimic>` recipe.

.. toctree::
   :hidden:

   Motion Data <supermimic_motion>

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      SuperMimic Policy (Unitree G1)

   .. grid-item-card:: Algorithms
      :text-align: center

      SuperMimic

   .. grid-item-card:: Tasks
      :text-align: center

      LAFAN1 motion tracking · Standing push recovery

   .. grid-item-card:: Hardware
      :text-align: center

      1–4 NVIDIA GPUs · :ref:`AMD ROCm (experimental) <supermimic-hardware>`

| **You'll do:** review the G1 task, observation, reward, and config specs → launch training or evaluation with :doc:`SuperMimic <supermimic>`.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · SuperMimic environment and prepared LAFAN1 clips from :doc:`supermimic`.

Tasks
~~~~~

The environment supports two task modes selected by the training or evaluation config:

.. list-table::
   :header-rows: 1
   :widths: 20 26 24 30

   * - Task
     - Reference Input
     - Episode Limit & Failure
     - Default Config
   * - Motion tracking (``tracking``)
     - 40 LAFAN1 clips (or custom ``clip_*.pt`` clips)
     - 10 s in training (full clip in eval); ends on clip end or fall
     - ``supermimic_lafan1_gencoord`` / ``supermimic_lafan1_eval``
   * - Standing push recovery (``push_recovery``)
     - Built-in mjlab knees-bent standing pose (no clips required)
     - 1 s settle + up to 5 s after horizontal velocity push (0.5–4.0 m/s)
     - ``supermimic_push_recovery``

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - Current robot state paired with a 10-frame reference motion window starting from the current frame and sampled every 5 control steps (0.1 s spacing; clamped at the clip's last frame). State encoding uses generalized coordinates (``gencoord``, 68-dim) or tracked-body poses and velocities in the pelvis frame (``maxcoord``, 214-dim).
   * - Action
     - 29-dim target joint angles at 50 Hz, tracked by a PD controller over four 5 ms physics substeps (200 Hz physics). Training adds Gaussian noise (``action_noise_std: 0.1``); evaluation runs the policy deterministically.
   * - Reward
     - Per-step tracking reward over the pelvis and 14 bodies minus action-rate and soft joint-limit penalties (ceiling ``0.1`` per step; see `Reward and Termination`_ below).
   * - Prompt
     - Not used (proprioceptive control only).

Environment Details
-------------------

Tracked Bodies and Randomization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tracking scores 14 bodies across the humanoid:

* **Root & torso:** ``pelvis``, ``torso_link``
* **Legs:** ``left/right_hip_roll_link``, ``left/right_knee_link``, ``left/right_ankle_roll_link``
* **Arms:** ``left/right_shoulder_roll_link``, ``left/right_elbow_link``, ``left/right_wrist_yaw_link``

During training, each environment draws domain randomization at initialization (foot friction in ``[0.3, 1.2]``, torso center-of-mass offsets, and 29 joint encoder biases in ``[-0.01, 0.01]`` rad) and adds reset noise to the sampled reference pose and velocities. Evaluation disables domain randomization, reset noise, and action noise.

Reward and Termination
~~~~~~~~~~~~~~~~~~~~~~

Before scoring body position and orientation, the environment aligns the reference pose to the robot pelvis's horizontal position (xy) and yaw so those terms measure local posture; root terms compare directly against the reference pelvis.

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Term
     - Weight
     - What It Measures
   * - Root position & orientation
     - 0.5, 0.5
     - World pelvis position and geodesic orientation error against the reference
   * - Body position & orientation
     - 1.0, 1.0
     - Mean position and orientation error across the 14 tracked bodies (pelvis-anchored)
   * - Body linear & angular velocity
     - 1.0, 1.0
     - Mean world linear and angular velocity error across the 14 tracked bodies
   * - Action-rate penalty
     - -0.1
     - Squared change between consecutive raw policy actions
   * - Joint-limit penalty
     - -2.0
     - Sum of joint-angle violations outside soft limits

The sum of these terms is multiplied by ``0.02 s``, giving a tracking ceiling of ``0.1`` per step. An episode terminates as a **fall** when the pelvis height error exceeds ``0.25 m`` or the pelvis tilt error (difference in projected gravity z-component between robot and reference pelvis frames) exceeds ``0.8``. Reaching the 10-second training limit or the end of the clip ends the episode without counting as a fall.

Environment Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

The default preset lives in ``examples/embodiment/config/env/g1_mimic.yaml``. Commonly adjusted keys:

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Key
     - Default
     - Meaning
   * - ``motion_path``
     - ``${oc.env:SUPERMIMIC_MOTION_PATH,""}``
     - Directory containing converted ``clip_*.pt`` files
   * - ``motion_format``
     - ``lafan1``
     - Dataset loader format (``lafan1``)
   * - ``num_clips``
     - ``null`` (``40`` in recipe)
     - Number of clips to load; ``null`` loads every clip in ``motion_path``
   * - ``motion_on_cpu``
     - ``False``
     - Keep the motion library in host RAM and transfer only queried frames to GPU
   * - ``max_episode_seconds``
     - ``10.0``
     - Maximum training episode duration before truncation
   * - ``action_noise_std``
     - ``0.1``
     - Standard deviation of Gaussian action noise during training

Setup and Run
-------------

Follow :ref:`Setup and Run <supermimic-hardware>` on the :doc:`SuperMimic <supermimic>` page to install the ``.venv-supermimic`` environment (NVIDIA CUDA or AMD ROCm) and prepare the 40 LAFAN1 clips into ``./data/lafan1_g1`` (or prepare custom ``clip_*.pt`` files via :doc:`supermimic_motion`).

.. tab-set::

   .. tab-item:: SuperMimic (GenCoord)

      Activate ``.venv-supermimic`` from the repository root and launch single-GPU training on ``g1_mimic`` (starts 4096 parallel G1 environments on GPU 0 for 6000 updates, saving checkpoints every 2000 updates):

      .. code-block:: bash

         source .venv-supermimic/bin/activate
         export MUJOCO_GL=disabled
         export SUPERMIMIC_MOTION_PATH=./data/lafan1_g1
         bash examples/embodiment/run_embodiment.sh supermimic_lafan1_gencoord

Visualization and Results
-------------------------

Watch training curves in TensorBoard (``tensorboard --logdir ./logs``); see :doc:`SuperMimic <supermimic>` for 40-clip LAFAN1 tracking results and multi-GPU throughput benchmarks.

Next Steps
----------

* **Algorithm guide, multi-GPU scaling, MaxCoord, and results:** :doc:`supermimic`
* **Standalone evaluation (motion tracking & standing push recovery):** :doc:`../../evaluations/guides/supermimic`
* **Custom motion clips and offline conversion:** :doc:`supermimic_motion`
* **Algorithm equations and implementation details:** :doc:`../../reference/algorithms/supermimic`
