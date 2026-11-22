SuperMimic Motion Data
======================

The main guide, :doc:`supermimic`, trains on the paper's 40 LAFAN1 clips.
This page covers training on other motion data: the experimental BONES-SEED
converter, and the clip file format for clips of your own.

Installation
------------

Use the environment from :doc:`supermimic`. Every command runs from the
RLinf repository root with that environment activated.

Run It
------

A clip folder holds ``clip_*.pt`` files, loaded in file-name order. Run the
setup block of the main guide first, then point ``SUPERMIMIC_MOTION_PATH`` at
the folder, and train and evaluate with the commands in the main guide plus
the overrides in the sections below.

Motion-tracking evaluation plays each clip in its own environment and is
sized for the 40 LAFAN1 clips. With more than 40 loaded clips,
``env.eval.num_clips=null`` alone fails a capacity check at startup. Also
raise ``env.eval.total_num_envs``, or ``env.eval.rollout_epoch`` to play the
clips in several rounds, so that their product covers the clip count. If a
clip runs longer than ``env.eval.max_steps_per_rollout_epoch`` control steps
(13656 by default), raise that too.

BONES-SEED (Experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~

The official release is
`bones-studio/seed <https://huggingface.co/datasets/bones-studio/seed>`_ on
Hugging Face, with G1 trajectories in ``g1/csv/{date}/{motion}.csv``. Read
``LICENSE.md`` in the download first; the script does not download the data.
The converter follows the public header in NVIDIA ProtoMotions'
`SEED G1 CSV preparation <https://github.com/NVlabs/ProtoMotions/blob/main/docs/source/getting_started/seed_g1_csv_preparation.rst>`_:
120 Hz, ``root_translateX/Y/Z`` in centimeters, ``root_rotateX/Y/Z`` as
extrinsic XYZ Euler angles in degrees, and 29 ``<joint>_dof`` columns in
degrees named after the mjlab G1 joints. This convention has not been checked
against an official CSV.

.. code-block:: bash

   python toolkits/supermimic/convert_bones_seed.py \
       --input /data/bones-seed/g1/csv \
       --output /data/bones_g1
   python toolkits/supermimic/bones_feasibility.py --clips /data/bones_g1 \
       --out /data/bones_g1_annotated
   export SUPERMIMIC_MOTION_PATH=/data/bones_g1_annotated

``--input`` is one CSV or a directory scanned recursively. Clips are
resampled to 50 Hz, existing outputs are kept unless ``--overwrite`` is set,
and a broken CSV is reported without stopping the rest. The converter is
tested on synthetic CSVs only.

``bones_feasibility.py`` writes every clip to the ``--out`` folder with its
geometry annotations and drops none. Training and motion-tracking evaluation
filter the clips by motion name and by those annotations when they load
them. To train or evaluate on these clips, append these overrides to the
launch command:

.. code-block:: bash

   env.train.motion_format=bones_seed env.train.num_clips=null \
       env.eval.motion_format=bones_seed env.eval.num_clips=null

The BONES-SEED loader keeps the library in host memory and moves only the
queried rows to the GPU.

Clips of Your Own
~~~~~~~~~~~~~~~~~

Write each clip in the format below, put them in one folder, and train with
``env.train.num_clips=null`` (and ``env.eval.num_clips=null`` for
evaluation), which loads every clip in the folder.

The folder may also hold a ``manifest.json``. With the manifest present, the
loader refuses a clip whose SHA-256 differs or that the manifest does not
list. Without it, the loader checks the format only. The BONES-SEED format
does not read a manifest.

Clip File Format
----------------

Each ``clip_*.pt`` is a dictionary saved with ``torch.save``:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Key
     - Contents
   * - ``fps``
     - 50 or 30. A 30 Hz clip is resampled onto the 50 Hz control grid and
       its velocities are recomputed by central difference. A 50 Hz clip is
       used as stored, so **its velocity columns must be correct**. Other
       rates are refused.
   * - ``joint_names``
     - 29 joint names, in the same order as
       ``rlinf.envs.sim.g1_mimic.constants.G1_JOINT_NAMES`` (the mjlab G1
       order).
   * - ``q0``
     - The 29-d default joint pose in radians. It must equal mjlab's default,
       ``asset_default_state()[7:36]``.
   * - ``state``
     - A float tensor ``[T, 71]`` with the columns below. No NaN or Inf, and
       the root quaternion is normalized.
   * - ``name``
     - Optional. The BONES-SEED converter writes it.

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
     - Root quaternion, wxyz
     - --
   * - ``7:36``
     - Joint positions, in ``joint_names`` order
     - rad
   * - ``36:39``
     - Root linear velocity, world frame
     - m/s
   * - ``39:42``
     - Root angular velocity, root frame
     - rad/s
   * - ``42:71``
     - Joint velocities
     - rad/s

This example writes a 5 s clip of standing still and runs the loader's own
checks on it:

.. code-block:: python

   from pathlib import Path

   import torch

   from rlinf.envs.sim.g1_mimic.constants import G1_JOINT_NAMES
   from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state
   from rlinf.envs.sim.g1_mimic.motion import decode_motion_payload

   default = asset_default_state().float()  # mjlab standing keyframe, [71]
   T = 250                                  # 5 s at 50 Hz
   state = torch.zeros(T, 71)
   state[:, :36] = default[:36]             # root pose and joint positions
   payload = {
       "fps": 50,
       "joint_names": list(G1_JOINT_NAMES),
       "q0": default[7:36],
       "state": state,
   }
   decode_motion_payload(payload, "clip_000.pt")  # raises ValueError if invalid
   out = Path("/data/my_clips")
   out.mkdir(parents=True, exist_ok=True)
   torch.save(payload, out / "clip_000.pt")
