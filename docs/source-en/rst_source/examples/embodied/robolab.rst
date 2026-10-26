Evaluation on RoboLab
=====================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/robolab.png
   :align: center
   :width: 90%
   :alt: RoboLab manipulation benchmark

   RoboLab (`NVlabs/RoboLab <https://github.com/NVlabs/RoboLab>`__).

Evaluate RoboLab's 120 DROID joint-position manipulation tasks with RLinf's π₀.₅ policy. This example covers installation, checkpoint download, task discovery, and success and normalized-score reporting through RLinf's environment interface and evaluation runner.

Overview
--------

Evaluate from two RGB views, robot state, and a natural-language instruction.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀.₅ / OpenPI

   .. grid-item-card:: Algorithms
      :text-align: center

      Evaluation only

   .. grid-item-card:: Tasks
      :text-align: center

      120 manipulation tasks

   .. grid-item-card:: Hardware
      :text-align: center

      NVIDIA GPU / Isaac Sim

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Group
     - Coverage
   * - Difficulty
     - simple / moderate / complex
   * - Procedural
     - affordance / reorientation / stacking
   * - Relational
     - conjunction / counting / spatial
   * - Visual
     - color / semantics / size

Task metadata provides names, instructions, difficulty, capability attributes, durations, and source paths. Use the export command in the :doc:`evaluation guide <../../evaluations/guides/robolab>` to inspect all tasks in your installed version.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - Specification
   * - Observation
     - Two 224×224 RGB views; seven joint positions and one gripper state.
   * - Action
     - Eight dimensions: seven joint-position targets and one gripper target; 15 actions per chunk.
   * - Reward
     - Terminal reward 1 on success, otherwise 0.
   * - Prompt
     - Natural-language instruction from the task metadata.

Installation
------------

Install one uv environment with ``bash requirements/install.sh embodied --model openpi --env robolab``. See the :doc:`evaluation guide <../../evaluations/guides/robolab>` for hardware requirements, installation validation limits, and checkout reuse.

Download the Model
------------------

Use the ``RLinf/RLinf-Pi05-RoboLab-droid-jointpos`` checkpoint and point ``rollout.model.model_path`` at its local directory. The :doc:`evaluation guide <../../evaluations/guides/robolab>` provides the download command and required files.

Run It
------

The configuration is ``evaluations/robolab/pi05_droid_jointpos.yaml``. Launch it with ``bash evaluations/run_eval.sh robolab pi05_droid_jointpos`` after setting the model path and task limits. Follow the :doc:`evaluation guide <../../evaluations/guides/robolab>` for one task or the full benchmark.

Visualization and Results
-------------------------

Watch ``eval/success_once`` and ``eval/score`` in the terminal and TensorBoard. The following runs use the same 120 tasks and ten episodes per task; see the :doc:`evaluation guide <../../evaluations/guides/robolab>` for metrics and logs.


.. list-table::
   :header-rows: 1

   * - Evaluation
     - Success rate
     - Normalized score
   * - RoboLab paper π₀.₅
     - 28.00%
     - 0.4300
   * - Native RoboLab + JAX
     - 27.83%
     - 0.4198
   * - Native RoboLab + Torch (compiled)
     - 26.50%
     - 0.4134
   * - RLinf + Torch (aligned, eager)
     - 27.00%
     - 0.4228

The RLinf run used eight A800 GPUs, 15 tasks per GPU, ten environments per task, and policy microbatches of one observation. Native Torch used compilation; RLinf used eager execution. Random streams were not paired episode by episode, so similar aggregate results do not establish identical actions.
