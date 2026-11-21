RL on Dexbotic Models
=====================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/dexmal/dexbotic/main/resources/intro.png
   :align: center
   :width: 90%

   Dexbotic model overview (image: `Dexbotic <https://github.com/dexmal/dexbotic>`__).

`Dexbotic <https://github.com/dexmal/dexbotic>`__ is an open-source VLA toolbox from Dexmal.
RLinf uses the Dexbotic π\ :sub:`0`\ and DM0 policies as LIBERO action-generation models, then fine-tunes them online with PPO.

Overview
--------

Fine-tune Dexbotic π\ :sub:`0`\ or DM0 on LIBERO with PPO.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      LIBERO

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      LIBERO Spatial · Object · Goal · 10

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 8 GPUs

| **You'll do:** install deps → download a Dexbotic checkpoint → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · a LIBERO-compatible Dexbotic checkpoint (steps below).

Tasks
~~~~~

Select the model page by matching the environment, task family, and config or checkpoint artifact.

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - Environment
     - Task / Suite
     - Config / Weights
     - Focus
   * - LIBERO
     - LIBERO-Spatial
     - ``libero_spatial_ppo_dexbotic_*``
     - Dexbotic pi0/dm0 policies on spatial manipulation tasks.
   * - LIBERO
     - LIBERO-Object
     - ``libero_object_ppo_dexbotic_pi0``
     - Dexbotic pi0 on object manipulation tasks.
   * - LIBERO
     - LIBERO-Goal / LIBERO-10
     - ``libero_goal_ppo_dexbotic_pi0`` / ``libero_10_ppo_dexbotic_pi0``
     - Goal-conditioned and long-horizon LIBERO suites.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - Field
     - Description
   * - Observation
     - LIBERO camera streams and proprioception packaged for Dexbotic policies.
   * - Action
     - Chunked continuous actions produced by the selected Dexbotic policy backend, including flow-matching / flow-SDE settings.
   * - Reward
     - LIBERO success signal or simulator reward used for PPO updates.
   * - Prompt
     - Natural-language LIBERO instruction consumed by the policy processor.

Installation
------------

.. include:: _setup_common.rst

.. code-block:: bash

   bash requirements/install.sh embodied --model dexbotic --env maniskill_libero && source .venv/bin/activate

Download the Model
------------------

Download the checkpoint for the Dexbotic model you plan to train:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model
     - Hugging Face Repo
   * - Dexbotic π₀
     - ``Dexmal/libero-db-pi0``
   * - DM0
     - ``Dexmal/DM0-libero``

.. code-block:: bash

   hf download <repo> --local-dir <dir>

.. include:: _model_path.rst

Run It
------

Each recipe is a YAML config under ``examples/embodiment/config/``:

.. list-table::
   :header-rows: 1
   :widths: 30 26 44

   * - Task suite
     - Model
     - Config
   * - LIBERO Spatial
     - Dexbotic π₀
     - ``libero_spatial_ppo_dexbotic_pi0.yaml``
   * - LIBERO Spatial
     - DM0
     - ``libero_spatial_ppo_dexbotic_dm0.yaml``
   * - LIBERO Object
     - Dexbotic π₀
     - ``libero_object_ppo_dexbotic_pi0.yaml``
   * - LIBERO Goal
     - Dexbotic π₀
     - ``libero_goal_ppo_dexbotic_pi0.yaml``
   * - LIBERO 10
     - Dexbotic π₀
     - ``libero_10_ppo_dexbotic_pi0.yaml``

Launch a config with ``run_embodiment.sh``:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_dexbotic_pi0

For π₀, ``actor.model.precision`` controls the loaded parameter dtype. Set it to
``bf16`` for BF16 execution, and use the same precision in ``rollout.model``.
Leaving precision at ``null`` preserves the checkpoint's default dtype.

.. admonition:: Configure further
   :class: note

   - π₀ checkpoint path → set ``actor.model.model_path`` and ``rollout.model.model_path`` to ``libero-db-pi0``.
   - DM0 checkpoint path → set both model paths to ``DM0-libero`` in ``libero_spatial_ppo_dexbotic_dm0.yaml``.
   - Action chunks → π₀ uses ``num_action_chunks: 5``; DM0 uses ``num_action_chunks: 10``.
   - Metric definitions and logging backends → :doc:`Training metrics <../../reference/metrics>`
   - Placement and throughput → :doc:`Placement <../../concepts/placement>` and :doc:`Execution modes <../../concepts/execution_modes>`

Standalone Evaluation
---------------------

Run Dexbotic's LIBERO evaluator for a trained checkpoint:

.. code-block:: bash

   python toolkits/standalone_eval_scripts/dexbotic/libero_eval.py \
      --config_name db_pi0_libero \
      --pretrained_path /path/to/checkpoint \
      --task_suite_name libero_spatial \
      --num_trials_per_task 50 \
      --action_chunk 5 \
      --num_steps 10

For DM0, switch the evaluator config and action chunk:

.. code-block:: bash

   python toolkits/standalone_eval_scripts/dexbotic/libero_eval.py \
      --config_name dm0_libero \
      --pretrained_path /path/to/checkpoint \
      --task_suite_name libero_spatial \
      --num_trials_per_task 50 \
      --action_chunk 10 \
      --num_steps 10

You can also use RLinf's unified VLA evaluation flow. See :doc:`evaluation <../../evaluations/index>`.

Visualization and Results
-------------------------

Launch TensorBoard to watch training live:

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

The key signal to watch is **``env/success_once``** — the episodic success rate. For every logged metric, see :doc:`Training metrics <../../reference/metrics>`.
