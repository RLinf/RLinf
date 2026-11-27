RoboLab Training and Evaluation
===============================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/robolab.png
   :align: center
   :width: 90%
   :alt: RoboLab manipulation benchmark

   RoboLab (`NVlabs/RoboLab <https://github.com/NVlabs/RoboLab>`__).

RoboLab is a manipulation benchmark built on Isaac Sim, with 120 tasks covering procedural, relational, and visual capabilities across three difficulty levels. RLinf supports PPO training and evaluation of the OpenPI π₀.₅ policy on its DROID joint-position tasks.

Overview
--------

Train or evaluate a policy from two RGB views, robot state, and a natural-language instruction.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀.₅ / OpenPI

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO and evaluation

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

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: OpenPI π₀.₅

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env robolab && source .venv/bin/activate

      **2. Download**

      Download the converted PyTorch checkpoint for RoboLab DROID joint-position tasks:

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-RoboLab-droid-jointpos --local-dir ./RLinf-Pi05-RoboLab-droid-jointpos

      **3. Launch**

      .. list-table::
         :header-rows: 1
         :widths: 24 46 30

         * - Recipe
           - Config
           - Command suffix
         * - OpenPI π₀.₅ (PPO)
           - ``examples/embodiment/config/robolab_banana_in_bowl_ppo_openpi_pi05.yaml``
           - ``robolab_banana_in_bowl_ppo_openpi_pi05``
         * - OpenPI π₀.₅ (eval)
           - ``evaluations/robolab/robolab_pi05_droid_jointpos.yaml``
           - ``robolab_pi05_droid_jointpos``

      Launch PPO training on ``BananaInBowlTask`` (with per-environment auto-reset):

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh \
           robolab_banana_in_bowl_ppo_openpi_pi05 \
           actor.model.model_path=./RLinf-Pi05-RoboLab-droid-jointpos

      Or run standalone evaluation on the default task (``BowlStackingLeftOnRightTask``, 10 episodes):

      .. code-block:: bash

         bash evaluations/run_eval.sh robolab robolab_pi05_droid_jointpos \
           rollout.model.model_path=./RLinf-Pi05-RoboLab-droid-jointpos

      Logs and TensorBoard events are written to ``logs/<timestamp>-<config_name>/``.

Follow the :doc:`evaluation guide <../../evaluations/guides/robolab>` to export all 120 task definitions, configure task step limits, and run the full benchmark.

.. warning::

   Keep ``env.train.enable_offload: false``. RoboLab does not support suspending and resuming a live Isaac Sim environment between rollout and training.

Visualization and Results
-------------------------

Watch ``env/success_once`` and ``env/return`` during PPO training, and ``eval/success_once`` and ``eval/score`` during validation and evaluation, in the terminal or TensorBoard (``tensorboard --logdir ./logs --port 6006``). See :doc:`Training metrics <../../reference/metrics>` for training metric definitions. The following benchmark evaluation runs use the same 120 tasks and ten episodes per task; see the :doc:`evaluation guide <../../evaluations/guides/robolab>` for evaluation metrics and logs.


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
