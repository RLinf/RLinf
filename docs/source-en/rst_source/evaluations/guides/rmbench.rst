RMBench Evaluation
==================

Evaluate a fine-tuned π₀.₅ Short Memory checkpoint on the Aloha robot in
`RMBench <https://rmbench.github.io/>`__. Use the RMBench checkout, SFT checkpoint,
and normalization file from the :doc:`training example <../../examples/embodied/rmbench>`.

Installation
------------

From the RLinf root, install the simulator in a separate environment. The
simulator requires an NVIDIA GPU and a CUDA toolkit for building CuRobo.
Use the RMBench checkout from the training recipe and download its assets
from inside that checkout.

.. code-block:: bash

   bash requirements/install.sh embodied --env rmbench --venv .rmbench-venv
   source .rmbench-venv/bin/activate
   (cd RMBench && bash scripts/_download_assets.sh)
   source .venv/bin/activate

The launcher uses the active OpenPI environment for inference and
``.rmbench-venv`` for simulation.

Run Evaluation
--------------

In ``evaluations/rmbench/rmbench_openpi_pi05_eval.yaml``, set these paths.

.. list-table::
   :header-rows: 1

   * - Setting
     - Value
   * - ``env.eval.init_params.rmbench_path``
     - The RMBench checkout with the downloaded assets.
   * - ``env.eval.init_params.sim_python``
     - ``.rmbench-venv/bin/python``.
   * - ``rollout.model.model_path``
     - The SFT checkpoint directory ``global_step_<N>``.
   * - ``rollout.model.openpi_data.norm_stats_path``
     - The ``norm_stats.json`` used for SFT.

The configuration uses the ``seen`` instruction split for ``observe_and_pickup``.
RMBench screens the seeds with its expert and runs the policy on 100 accepted
episodes. Set ``env.eval.rollout_epoch: 1`` to check your setup.
The OpenPI policy uses ``rollout.model.openpi.eval_seed`` for action sampling.
Start evaluation from the RLinf root.

.. code-block:: bash

   bash evaluations/run_eval.sh rmbench rmbench_openpi_pi05_eval

With Short Memory enabled, the simulator records observations at every control
step, including steps within an action chunk. The policy receives eight frames,
spaced 53 steps apart, with time offsets based on the demonstrations' nominal
15 Hz rate.

To switch tasks, set ``env.eval.init_params.task_name``. Use the task's limit
from ``env_cfg/task_config/_eval_step_limit.yml`` in the RMBench checkout for
``env.eval.max_episode_steps``. Round that limit up to a multiple of 32 for
``env.eval.max_steps_per_rollout_epoch`` so the rollout covers the final chunk.

To use a current-frame checkpoint, set ``rollout.model.openpi.short_memory: false``
and ``env.eval.short_memory: null``.

Results
-------

Find the success rate (``eval/success_once``) in
``logs/<run>/eval_embodiment.log`` or TensorBoard.
Set ``env.eval.video_cfg.save_video: true`` to record videos in the run directory.
