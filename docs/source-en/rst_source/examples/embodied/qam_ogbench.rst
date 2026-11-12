Train offline QAM on OGBench
============================

RLinf provides a native PyTorch QAM implementation for offline OGBench training. The example task is ``antmaze-large-navigate-singletask-task1-v0``. Offline updates use ``OfflineRunner``; evaluation uses the standard EnvWorker and RolloutWorker. Online fine-tuning should use an online runner and is outside this example.

Setup
-----

From the RLinf repository root:

.. code-block:: bash

   bash requirements/install.sh embodied --env ogbench
   source .venv/bin/activate
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl
   export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

Train and evaluate
------------------

``ogbench_qam_offline.yaml`` trains one seed for one million gradient updates with batch size 256. It evaluates 50 episodes every 50,000 updates. Choose a free GPU via ``cluster.component_placement``. Run each seed in a separate process with a unique experiment name:

.. code-block:: bash

   for seed in 10001 20002 30003; do
     python examples/embodiment/train_offline_rl.py \
       --config-name ogbench_qam_offline \
       actor.seed="$seed" \
       runner.logger.experiment_name="qam-antmaze-offline-$seed"
   done

This runs the seeds sequentially. For parallel runs, assign separate ``cluster.component_placement.actor``, ``env``, and ``rollout`` GPU indices to the three processes. Checkpoints are under ``logs/<experiment>/checkpoints/global_step_<step>/actor/``. Training metrics are under ``logs/<experiment>/seed_<seed>/train.csv``; evaluation metrics are saved by RLinf's metric logger.

Resume from a saved update:

.. code-block:: bash

   python examples/embodiment/train_offline_rl.py \
     --config-name ogbench_qam_offline \
     actor.seed=10001 \
     runner.resume_dir=logs/qam-antmaze-offline-10001/checkpoints/global_step_250000 \
     runner.logger.experiment_name=qam-antmaze-offline-10001-resumed

Add ``runner.only_eval=true`` to evaluate the checkpoint without further updates. The checkpoint stores the policy, optimizer, dataset sampler, and random generator state for exact offline continuation.

Checks
------

Unit tests cover sampling, QAM updates, and checkpoint state. The optional parity test compares losses, gradients, and one update against official JAX QAM commit ``2726d767c9a0a7a46d49693f0391f73dc2cf58ac``. Training uses RLinf's PyTorch implementation and does not import the reference checkout.

.. code-block:: bash

   python -m pytest tests/unit_tests/test_qam.py
   QAM_REFERENCE_ROOT=/path/to/pinned/QAM \
     JAX_PLATFORMS=cpu python -m pytest tests/parity_tests/test_qam_reference.py
