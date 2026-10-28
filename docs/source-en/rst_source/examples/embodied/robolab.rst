Evaluation on RoboLab
=====================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/robolab.png
   :align: center
   :width: 90%
   :alt: RoboLab manipulation benchmark

   RoboLab (`NVlabs/RoboLab <https://github.com/NVlabs/RoboLab>`__).

RoboLab is a manipulation benchmark built on Isaac Sim, with 120 tasks covering procedural, relational, and visual capabilities across three difficulty levels. RLinf supports evaluating the OpenPI π₀.₅ policy on its DROID joint-position tasks.

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

Run It
------

Follow the :doc:`evaluation guide <../../evaluations/guides/robolab>` to install the environment, download the checkpoint, configure tasks, and run the evaluation.

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
