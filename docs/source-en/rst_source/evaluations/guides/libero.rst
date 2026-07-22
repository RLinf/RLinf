LIBERO Evaluation
=================

LIBERO is a robotic manipulation simulation benchmark built on robosuite (MuJoCo), with suites including Spatial, Object, Goal, and Long. RLinf supports parallel VLA policy evaluation on LIBERO with task-level success metrics.

Related training docs: :doc:`../../examples/embodied/libero`, :ref:`LIBERO-Pro & LIBERO-Plus <liberopro-plus-benchmark>`

Environment Setup
-----------------

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env libero
   source .venv/bin/activate

With ``--env libero``, the installer clones LIBERO into ``.venv/libero`` (or reuses an existing checkout when ``LIBERO_PATH`` is set) and appends it to ``PYTHONPATH`` in ``.venv/bin/activate``.

Supported models include ``openpi``, ``openvla-oft``, ``starvla``, and ``dreamzero`` — replace ``--model`` accordingly during installation.

Example Configs
---------------

Available under ``evaluations/libero/``:

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Config file
     - Task suite
     - Model
   * - ``libero_spatial_openpi_pi05_eval.yaml``
     - Spatial
     - π₀.₅
   * - ``libero_spatial_starvla_eval.yaml``
     - Spatial
     - StarVLA
   * - ``libero_spatial_dreamzero_eval.yaml``
     - Spatial
     - DreamZero
   * - ``libero_spatial_dreamzero_eval_sglang.yaml``
     - Spatial
     - DreamZero (SGLang backend)
   * - ``libero_object_openpi_pi05_eval.yaml``
     - Object
     - π₀.₅
   * - ``libero_object_openvlaoft_eval.yaml``
     - Object
     - OpenVLA-OFT
   * - ``libero_goal_openpi_eval.yaml``
     - Goal
     - π₀
   * - ``libero_goal_openvlaoft_eval.yaml``
     - Goal
     - OpenVLA-OFT
   * - ``libero_10_openpi_pi05_eval.yaml``
     - Long (libero_10)
     - π₀.₅
   * - ``libero_10_openvlaoft_eval.yaml``
     - Long (libero_10)
     - OpenVLA-OFT

For the DreamZero SGLang backend, see :doc:`dreamzero_sglang`.

End-to-End Workflow
-------------------

**Step 1: Activate the environment**

.. code-block:: bash

   source .venv/bin/activate

**Step 2: Edit the config**

Copy or edit the target YAML and set at least ``rollout.model.model_path``. See :doc:`../reference/configuration` (:ref:`env-eval-fields`) for ``env.eval`` field descriptions; see :ref:`libero-eval-config` below for the LIBERO eval protocol and suite-specific settings.

**Step 3: Launch evaluation**

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model

**Step 4: Check results**

The terminal prints ``eval/success_once``; see :doc:`../reference/results` for logs.

.. _libero-eval-config:

Evaluation Configuration
------------------------

LIBERO evaluation runs one trajectory per ``(task_id, trial_id)`` pair in the suite and reports ``eval/success_once`` (fraction of trajectories with at least one success). The fields below are all under ``env.eval`` and together control **parallelism**, **trajectory length**, and **test-set coverage**.

Evaluation Protocol
~~~~~~~~~~~~~~~~~~~

LIBERO provides a fixed set of initial states per task (``task_suite.get_task_init_states(task_id)``, loaded from ``.pruned_init`` files).
The `official repo <https://github.com/Lifelong-Robot-Learning/LIBERO>`_ defines four standard eval suites: **LIBERO-Spatial**, **LIBERO-Object**, **LIBERO-Goal**, and **LIBERO-Long** (``libero_10``), each with 10 tasks and ~50 initial states per task—**~500** trajectories to fully evaluate one suite.
RLinf ``evaluations/libero/`` examples cover these four ``task_suite_name`` values: ``libero_spatial``, ``libero_object``, ``libero_goal``, and ``libero_10``.

In RLinf's ``LiberoEnv``, each eval trajectory is uniquely identified by ``(task_id, trial_id)``:

- ``task_id``: task index within the current ``task_suite_name`` (``0 … n_tasks-1``), determining the language instruction and BDDL scene;
- ``trial_id``: initial-state index for that task, loaded via ``get_task_init_states(task_id)[trial_id]`` into the MuJoCo configuration.

Internally, trials from all tasks are concatenated into a global ``reset_state_id``, which is decoded back into ``task_id`` and ``trial_id``.
In eval mode (``is_eval=True``), all ``reset_state_id`` values are traversed in interleaved order—``(task0, trial0), (task1, trial0), …, (task0, trial1), …``—so parallel envs advance trials evenly across tasks; on ``auto_reset``, the next ``reset_state_id`` is assigned in order.

One ``rollout_epoch`` should cover every ``(task_id, trial_id)`` pair in the suite. Two approaches:

1. **High parallelism:** Set ``total_num_envs`` ≥ total init states and ``max_steps_per_rollout_epoch = max_episode_steps`` so each parallel env runs exactly one trajectory (see ``libero_spatial_openpi_pi05_eval.yaml``).
2. **Auto-reset:** With ``auto_reset=True``, finished episodes immediately load the next init state. Set ``max_steps_per_rollout_epoch`` to **N** × ``max_episode_steps`` to evaluate roughly ``N × total_num_envs`` trajectories per epoch (see ``libero_spatial_dreamzero_eval.yaml``).

Per-Suite ``max_episode_steps`` Reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The longest training demo per suite sets a lower bound for the step limit. RLinf eval YAML values vary by model action frequency but should be **≥** this bound and match the training config:

.. list-table::
   :header-rows: 1
   :widths: 22 22 28 28

   * - Suite
     - Lower bound
     - RLinf example values
     - Example config
   * - ``libero_spatial``
     - 220
     - 240 / 480
     - ``libero_spatial_openpi_pi05_eval`` / ``libero_spatial_dreamzero_eval``
   * - ``libero_object``
     - 280
     - 280 / 512
     - ``libero_object_openpi_pi05_eval`` / ``libero_object_openvlaoft_eval``
   * - ``libero_goal``
     - 300
     - 320 / 512
     - ``libero_goal_openpi_eval`` / ``libero_goal_openvlaoft_eval``
   * - ``libero_10``
     - 520
     - 520
     - ``libero_10_openpi_pi05_eval`` / ``libero_10_openvlaoft_eval``

See :ref:`env-eval-fields` in :doc:`../reference/configuration` for detailed ``env.eval`` field descriptions.

Covering the Full Test Set
~~~~~~~~~~~~~~~~~~~~~~~~~~

Let ``S`` = total init states in the suite, ``E`` = ``total_num_envs``, ``T`` = ``max_episode_steps``.

Option 1: High parallelism (``auto_reset`` optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   env:
     eval:
       total_num_envs: 500        # S for Spatial / Object / Goal / Long
       max_episode_steps: 240
       max_steps_per_rollout_epoch: 240   # equals max_episode_steps
       auto_reset: True           # optional when E >= S
       rollout_epoch: 1

**Option 2: Auto-reset (recommended when memory is limited)**

.. code-block:: yaml

   env:
     eval:
       total_num_envs: 128
       max_episode_steps: 480
       # N = ceil(S / E); Spatial: ceil(500/128) = 4
       max_steps_per_rollout_epoch: 1920   # N * max_episode_steps = 4 * 480
       auto_reset: True
       ignore_terminations: True
       use_fixed_reset_state_ids: True
       use_ordered_reset_state_ids: True
       rollout_epoch: 1

Trajectories per ``rollout_epoch`` ≈ ``N × total_num_envs`` where ``N = max_steps_per_rollout_epoch / max_episode_steps``. For example, Spatial (``S=500``) with ``E=128`` requires ``N = ceil(500/128) = 4``.

**Multi-epoch averaging**

.. code-block:: yaml

   env:
     eval:
       rollout_epoch: 2           # same seed, two passes, metrics averaged

Notes
~~~~~

- ``max_steps_per_rollout_epoch`` must be divisible by ``rollout.model.num_action_chunks``; startup validation will fail otherwise.
- Env workers use seed offset ``seed + rank × stage_num + stage_id`` so each worker receives a distinct init-state subset.
- ``eval/success_once`` in the terminal is the success rate over completed trajectories; with ``auto_reset``, metrics are recorded only when a new episode finishes, avoiding double counting.

Advanced Usage
--------------

**LIBERO-PRO**

.. code-block:: bash

   export LIBERO_TYPE=pro
   export LIBERO_PERTURBATION=all
   bash evaluations/run_eval.sh libero libero_10_openvlaoft_eval

**LIBERO-PLUS**

.. code-block:: bash

   export LIBERO_TYPE=plus
   export LIBERO_SUFFIX=all
   bash evaluations/run_eval.sh libero libero_10_openvlaoft_eval

**Adjust parallel scale**

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval \
     env.eval.total_num_envs=64 \
     rollout.model.model_path=/path/to/model

FAQ
---

- **Rendering issues:** On headless systems, try ``export MUJOCO_GL=osmesa`` and ``export PYOPENGL_PLATFORM=osmesa`` (``run_eval.sh`` sets these by default).
- **Test coverage:** See :ref:`libero-eval-config` above; the key is coordinating ``total_num_envs``, ``auto_reset``, and ``max_steps_per_rollout_epoch``.

.. toctree::
   :hidden:
   :maxdepth: 1

   dreamzero_sglang
