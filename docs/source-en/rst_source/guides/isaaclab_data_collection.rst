Collect Isaac Lab Demonstrations
================================

Generate Panda grasping demonstrations with GraspGen and cuRobo, execute them
in Isaac Lab, and export verified successes as a local LeRobot dataset.
The task is grasp, lift, and hold for four bundled objects: a block, a slender
bar, a cylinder, and a disk. Grasp generation uses known meshes and simulation
object poses. Two RGB cameras record training observations.

Install the Runtime
-------------------

Use a Linux NVIDIA workstation with Isaac Sim 5.1.0, Isaac Lab 2.3.0,
cuRobo 0.7.8, and the GraspGen Panda model. Simulation, planning, and grasp
generation run in separate Python 3.11 environments. Run the RLinf entrypoint
in an environment with RLinf, NumPy, and LeRobot. The dataset version follows
the installed LeRobot version.

Clone the external collector and select the validated revision:

.. code-block:: bash

   git clone https://github.com/juchuanqi/graspgen-curobo-collect.git
   cd graspgen-curobo-collect
   git checkout 9f1feae9aa0a66a31053b4e237b29cf388a45637

Follow the collector's `installation instructions <https://github.com/juchuanqi/graspgen-curobo-collect/blob/9f1feae9aa0a66a31053b4e237b29cf388a45637/docs/INSTALL.md>`__
to install its environments and download model weights. Keep tracked runtime
files unchanged; candidates, caches, and local settings stay outside version control.

Configure a Small Batch
-----------------------

Return to the RLinf repository root and copy the example:

.. code-block:: bash

   cp examples/embodiment/config/isaaclab_graspgen_curobo_collect.yaml collection.yaml

Fill in the runtime checkout, three Python interpreters, and model configuration
path. Choose new output and candidate directories. Start with two block grasp attempts:

.. code-block:: yaml

   runtime:
     root: /path/to/graspgen-curobo-collect
     sim_python: /path/to/isaaclab/bin/python
     planner_python: /path/to/curobo/bin/python
     graspgen_python: /path/to/graspgen/bin/python
     graspgen_config: /path/to/models/checkpoints/graspgen_franka_panda.yml
   output: /path/to/new_collection
   candidates:
     root: /path/to/candidates_200
   collection:
     objects: [block55]
     episodes_per_object: 2
     workers: 1
     gpus: "0"

GPU indices are logical indices within ``CUDA_VISIBLE_DEVICES``. Each requested
initial state gets one physical attempt. Failed grasps remain in the raw run
and count toward the attempted batch.

Run Collection and Export
-------------------------

Check the runtime, model files, and CPU dependencies:

.. code-block:: bash

   python -m toolkits.isaaclab.collect_graspgen_curobo --config collection.yaml --stage check

Generate candidates, collect the fixed batch, and export qualified episodes:

.. code-block:: bash

   python -m toolkits.isaaclab.collect_graspgen_curobo --config collection.yaml

Collection records RGB and absolute joint-position targets at 60 Hz, with
480 Hz physics. Export keeps episodes that pass the collector's grasp-success
and data checks. Every exported row is read back with the official LeRobot
loader and compared with its source, including both images. The completed
export directory appears after this comparison passes.

Inspect the configured output directory:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Path
     - Contents
   * - ``raw/REPORT.md``
     - Attempts, successes, and failures.
   * - ``SELECTION.json``
     - Included episodes and exclusion reasons for other attempts.
   * - ``export/dataset/``
     - LeRobot data with RLinf's image, wrist_image, state, and actions fields.
   * - ``export/sidecars/``
     - Physical evidence, raw joints, and terminal RGB observations.
   * - ``export/AUDIT.json``
     - Readback result and exported episode/frame counts.

State and action contain seven arm joints in radians followed by total finger
opening in metres. Actions are absolute position targets. Observation at row t
precedes action t; the extra terminal observation is kept alongside the dataset.
Downstream policies must use this joint-position convention and matching field
mapping. See :doc:`data_collection` for RLinf data formats.

Continue a Stopped Run
----------------------

Resume interrupted raw collection with the original configuration:

.. code-block:: bash

   python -m toolkits.isaaclab.collect_graspgen_curobo --config collection.yaml --stage collect --resume

Completed attempt slots, including physical failures, retain their results.
After collection finishes, export without executing grasps again:

.. code-block:: bash

   python -m toolkits.isaaclab.collect_graspgen_curobo --config collection.yaml --stage export

Export failures preserve ``export.partial/`` for diagnosis. Completed exports
are never overwritten.
