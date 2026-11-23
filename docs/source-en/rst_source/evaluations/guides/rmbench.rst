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
   * - ``rmbench_path``
     - The RMBench checkout with the downloaded assets.
   * - ``sim_python``
     - ``.rmbench-venv/bin/python``.
   * - ``rollout.model.model_path``
     - The SFT checkpoint directory ``global_step_<N>``.
   * - ``rollout.model.openpi_data.norm_stats_path``
     - The ``norm_stats.json`` used for SFT.

The configuration uses the ``seen`` instruction split for ``observe_and_pickup``.
RMBench screens the seeds with its expert and runs the policy on 100 accepted
episodes. Set ``episodes: 1`` to check your setup, then restore 100 for evaluation.
Start evaluation from the RLinf root.

.. code-block:: bash

   bash evaluations/run_eval.sh rmbench rmbench_openpi_pi05_eval

During rollout the policy receives the same eight-frame history used in
training. The simulator records each control step, including those within a
32-action chunk, and samples every 53 steps. The model encodes the offsets
using the demonstrations' nominal 15 Hz rate.

To evaluate with current camera frames only, set
``rollout.model.openpi.short_memory: false`` and use a matching checkpoint.

Results
-------

RMBench prints ``Success rate`` in the terminal. The launcher saves the same
output to ``logs/<run>/eval_embodiment.log``. For the one-episode check, look for
a completed episode and its result. Set ``save_video: true`` to save videos in
the run directory. Video recording requires ``ffmpeg`` on the executable search path.
