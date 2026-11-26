G1 Motion Data Preparation
==========================

Prepare reference motion clips for the :doc:`G1 Motion Tracking Environment <g1_mimic>` and :doc:`SuperMimic <supermimic>`, including offline LAFAN1 conversion and custom ``clip_*.pt`` files.

**Prerequisites:** :doc:`Installation </rst_source/start/installation>` · the ``.venv-supermimic`` environment from :doc:`supermimic`.

Setup and Run
-------------

Activate ``.venv-supermimic`` and run all commands from the RLinf repository root.

.. tab-set::

   .. tab-item:: Offline LAFAN1 Conversion

      If the training machine cannot reach Hugging Face, convert LAFAN1 on an internet-connected machine that has an RLinf checkout and Python with ``torch``, ``numpy``, and ``omegaconf``. Because conversion checks against mjlab's G1 default joint pose, export that pose from the training machine first:

      **1. Export the default joint pose on the training machine**

      .. code-block:: bash

         python toolkits/supermimic/prepare_lafan1.py --export-q0 ./g1_q0.pt

      **2. Copy ``g1_q0.pt`` and convert the 40 clips on the internet-connected machine**

      .. code-block:: bash

         python toolkits/supermimic/prepare_lafan1.py --out ./data/lafan1_g1 \
             --q0-file ./g1_q0.pt --accept-license

      **3. Copy ``./data/lafan1_g1`` (including ``manifest.json``) back to the training machine**

      Point ``SUPERMIMIC_MOTION_PATH`` to ``./data/lafan1_g1`` and launch training as shown in :doc:`supermimic`.

   .. tab-item:: Custom Clips

      Save each custom clip in the ``clip_*.pt`` format below inside one folder (loaded in file-name order), point ``SUPERMIMIC_MOTION_PATH`` to it, and pass ``env.train.num_clips=null`` (or ``env.eval.num_clips=null`` for evaluation) to load every clip:

      .. code-block:: bash

         export MUJOCO_GL=disabled
         export SUPERMIMIC_MOTION_PATH=./data/my_clips
         bash examples/embodiment/run_embodiment.sh supermimic_lafan1_gencoord \
             env.train.num_clips=null

      To evaluate more than 40 clips, adjust the evaluation capacity settings in :doc:`../../evaluations/guides/supermimic`. If the directory includes a ``manifest.json``, the loader verifies every clip's SHA-256 and rejects unlisted files; without ``manifest.json``, it validates tensor shapes and values only.

Clip File Format
----------------

Each ``clip_*.pt`` file is a Python dictionary saved with ``torch.save``:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Key
     - Contents
   * - ``fps``
     - ``50`` or ``30``. A 30 Hz clip is resampled to the 50 Hz control grid and its velocities are recomputed by central difference. A 50 Hz clip is loaded as stored, so **its velocity columns must be accurate**. Other rates are rejected.
   * - ``joint_names``
     - List of 29 joint names in the exact mjlab Unitree G1 order (left leg, right leg, waist, left arm, right arm).
   * - ``q0``
     - 29-dim default joint pose in radians, matching mjlab's standing default pose.
   * - ``state``
     - Float tensor of shape ``[T, 71]`` with the column layout below, finite values only, and unit-norm root quaternions.

Column layout of ``state`` (``[T, 71]``):

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Columns
     - Contents
     - Unit
   * - ``0:3``
     - Root (pelvis) position, world frame
     - m
   * - ``3:7``
     - Root orientation quaternion (``wxyz``)
     - --
   * - ``7:36``
     - Joint positions, in ``joint_names`` order
     - rad
   * - ``36:39``
     - Root linear velocity, world frame
     - m/s
   * - ``39:42``
     - Root angular velocity, root body frame
     - rad/s
   * - ``42:71``
     - Joint velocities
     - rad/s

Export ``g1_q0.pt`` once with ``python toolkits/supermimic/prepare_lafan1.py --export-q0 ./g1_q0.pt``, then save each custom clip dictionary with ``torch.save`` (for example ``clip_000.pt``). The loader validates every clip's keys, joint order, ``q0`` match, quaternion norms, and finite values on startup.
