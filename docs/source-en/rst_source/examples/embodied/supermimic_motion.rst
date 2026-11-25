G1 Motion Data Preparation
==========================

Prepare reference motion clips for the :doc:`G1 Motion Tracking Environment <g1_mimic>` and :doc:`SuperMimic <supermimic>`, including offline LAFAN1 conversion and custom ``clip_*.pt`` files.

Installation
------------

Activate the SuperMimic virtual environment from :doc:`g1_mimic` and run all commands from the RLinf repository root.

Offline LAFAN1 Conversion
-------------------------

If the training machine cannot reach Hugging Face, convert LAFAN1 on another machine that has network access, an RLinf checkout, and Python with ``torch``, ``numpy``, and ``omegaconf``. Because conversion checks against mjlab's G1 default joint pose, export that pose from the training machine first:

1. On the training machine, export the default joint pose to ``g1_q0.pt``:

   .. code-block:: bash

      python toolkits/supermimic/prepare_lafan1.py --export-q0 g1_q0.pt

2. Copy ``g1_q0.pt`` to the conversion machine.
3. On the conversion machine, download and convert the 40 clips using that pose:

   .. code-block:: bash

      python toolkits/supermimic/prepare_lafan1.py --out /data/lafan1_g1 \
          --q0-file /path/to/g1_q0.pt --accept-license

4. Copy the entire ``/data/lafan1_g1`` directory (including ``manifest.json``) to the training machine and point ``SUPERMIMIC_MOTION_PATH`` to it.

Run It
------

Save each custom clip in the ``clip_*.pt`` format below inside one directory (loaded in file-name order), point ``SUPERMIMIC_MOTION_PATH`` to that directory, and pass ``env.train.num_clips=null`` (or ``env.eval.num_clips=null`` for evaluation) to load every clip in the folder:

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/my_clips
   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_gencoord \
       env.train.num_clips=null \
       runner.logger.log_path=/data/runs/supermimic_custom

To evaluate more than 40 clips, adjust the evaluation capacity settings described in :doc:`../../evaluations/guides/supermimic`. If the directory includes a ``manifest.json``, the loader verifies every clip's SHA-256 and rejects unlisted files; without ``manifest.json``, it checks tensor shapes and values only.

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

Export ``g1_q0.pt`` once with ``python toolkits/supermimic/prepare_lafan1.py --export-q0 g1_q0.pt``, then save each custom clip dictionary with ``torch.save`` (for example ``clip_000.pt``). The training and evaluation loader validates every clip's keys, joint order, ``q0`` match, quaternion norms, and finite values on startup.
