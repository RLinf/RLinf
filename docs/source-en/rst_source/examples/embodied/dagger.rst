DAgger for Embodied Policies
============================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dagger.jpg
   :align: center
   :width: 75%

   A DAgger training loop.

**DAgger** (Dataset Aggregation) is an imitation-learning algorithm that lets
the student policy interact with the environment, asks an expert policy to
relabel the visited states, and aggregates those expert-labeled trajectories
for further training. This page documents RLinf's simulator-based embodied DAgger workflow. Current DAgger support covers MLP and Pi0 models, and both
**sync** and **async** training pipelines.

For the real-world Franka pipeline, see :doc:`hg-dagger`.

Overview
--------

DAgger-finetune an embodied policy: the student acts, an expert relabels visited states, and the aggregated expert data trains the student.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      DAgger

   .. grid-item-card:: Models
      :text-align: center

      MLP · π₀

   .. grid-item-card:: Environments / Data
      :text-align: center

      ManiSkill · LIBERO · RoboTwin

   .. grid-item-card:: Training
      :text-align: center

      Sync · Async

| **You'll do:** install → set student/expert checkpoints → launch ``run_embodiment.sh`` (or ``run_async.sh``) → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · a student and an expert checkpoint (steps below).

Supported Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 26 58

   * - Model
     - Environment
     - Config
   * - MLP
     - ManiSkill (pick-cube)
     - ``maniskill_dagger_mlp.yaml``
   * - π₀
     - LIBERO-Spatial
     - ``libero_spatial_dagger_openpi.yaml``
   * - π₀
     - RoboTwin (adjust-bottle)
     - ``robotwin_adjust_bottle_dagger_openpi.yaml``
   * - π₀
     - LIBERO-Spatial (online LeRobot)
     - ``libero_spatial_dagger_openpi_lerobot.yaml``

How DAgger Works
----------------

**DAgger Pipeline**

1. **Mixed Rollout Policy**

   - During training, the rollout worker chooses the expert action with
     probability ``beta``.
   - During evaluation, RLinf always uses the student policy.

2. **Expert Relabeling**

   - If the student acts in the environment, RLinf runs an extra expert
     forward pass on the same observation.
   - The expert action is stored as the supervision target for that step.

3. **Replay-Buffer Training**

   - Expert-labeled trajectories are written into the replay buffer.
   - The actor then optimizes the ``embodied_dagger`` loss on replayed samples.

4. **Beta Scheduling**

   - ``init_beta`` controls the initial expert-action probability.
   - ``beta_schedule`` and ``beta_decay`` control how quickly execution shifts
     from expert to student.
   - ``beta_min`` is optional and sets the lower bound of ``beta``.

Online LeRobot DAgger
---------------------

The classic configs above store expert-labeled trajectories in an in-memory
**replay buffer**. ``libero_spatial_dagger_openpi_lerobot.yaml`` adds an
**online LeRobot** path: the env worker collects completed episodes in LeRobot
format, sends them to the actor in memory, and the actor trains from a rolling
window via :class:`~rlinf.data.datasets.dagger.RollingLeRobotDataset`.

**Online LeRobot pipeline**

1. **Mixed rollout and expert relabeling** — same ``beta`` scheduling and
   expert relabeling as classic DAgger.
2. **Episode collection** — when ``algorithm.dagger.online_lerobot.enabled`` is
   ``True``, EnvWorker uses :class:`~rlinf.data.embodied_io_struct.EmbodiedLerobotRolloutResult`
   to accumulate per-env frames and export completed episodes.
3. **Actor ingestion** — completed episodes flow to the actor through
   ``recv_lerobot_rollout_trajectories`` and append to the rolling dataset.
4. **Rolling-window training** — the actor samples from
   ``RollingLeRobotDataset`` (optionally with a decoded frame cache) and
   optimizes ``embodied_dagger`` loss.

**Offline vs online collection**

- **Offline disk export** — ``env.train.data_collection.enabled`` writes LeRobot
  shards to disk for later SFT or HG-DAgger. See :doc:`../../guides/data_collection`
  and :doc:`hg-dagger`.
- **Online in-memory collection** — ``algorithm.dagger.online_lerobot.enabled``
  streams episodes directly to the actor during DAgger training. Do **not** enable
  ``env.train.data_collection`` on the same train env when online LeRobot is on.

Installation
------------

For installation details, please first refer to :doc:`../../start/installation`.
The DAgger examples below use the embodied image or the equivalent local environment.

**Option 1: Docker Image**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.3-maniskill_libero
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.3-maniskill_libero

Please switch to the corresponding virtual environment via the built-in
``switch_env`` utility in the image:

.. code:: bash

   source switch_env openpi

**Option 2: Custom Environment**

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag for better download speed.
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   # For robotwin environment, please use the following command:
   # bash requirements/install.sh embodied --model openpi --env robotwin
   source .venv/bin/activate

Checkpoint Setup
----------------

Before launch, fill in the student and expert paths in the chosen YAML file.

**1. ManiSkill + MLP**

The MLP DAgger config uses a student checkpoint and an expert checkpoint under
``runner``:

.. code:: yaml

   runner:
     ckpt_path: null                       # Optional student warm start
     expert_ckpt_path: /path/to/expert_ckpt

The expert model in ``expert_ckpt_path`` could be produced by a PPO run in
:doc:`mlp`.

**2. LIBERO Spatial + Pi0**

The Pi0 DAgger config uses separate student and expert model paths:

.. code:: yaml

   actor:
     model:
       model_path: /path/to/student_model

   rollout:
     model:
       model_path: /path/to/student_model
     expert_model:
       model_path: /path/to/expert_model

You can find pretrained Pi0 checkpoints on Hugging Face for student initialization. For example:

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT --local-dir /path/to/model

The expert model checkpoint could also come from a PPO run in :doc:`pi0`.

**3. RoboTwin + Pi0**

The Pi0 DAgger config uses separate student and expert model paths:

.. code:: yaml

   actor:
     model:
       model_path: /path/to/student_model

   rollout:
     model:
       model_path: /path/to/student_model
     expert_model:
       model_path: /path/to/expert_model

In the same way, ou can find pretrained Pi0 checkpoints on Hugging Face for student initialization. For example:

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle --local-dir /path/to/model

The expert model checkpoint could also come from a PPO run in :doc:`pi0`.

In addition, the RoboTwin environment requires separate configuration of the RoboTwin code and corresponding Assets. Refer to :doc:`robotwin` for details, and then configure the corresponding paths in your YAML file.

.. code:: yaml

   env:
     train:
       assets_path: /path/to/robotwin_assets
     eval:
       assets_path: /path/to/robotwin_assets

**4. LIBERO Spatial + Pi0 (online LeRobot)**

Use the same student and expert ``model_path`` fields as the classic LIBERO Pi0
DAgger config above. Rolling-window and cache settings live under
``algorithm.dagger.online_lerobot``:

.. code:: yaml

   algorithm:
     dagger:
       online_lerobot:
         enabled: True
         only_success: True
         robot_type: "panda"
         fps: 10
         finalize_interval: 8
         data_path: ${runner.logger.log_path}/physical-intelligence/libero
         rolling_lerobot_window_size: 50000
         enable_decoded_cache: true
         decoded_cache_capacity: 25000
         cache_ingest_mode: new_shards   # or last_n / both
         lerobot_num_workers: 0          # recommended when cache is on

See ``examples/embodiment/config/libero_spatial_dagger_openpi_lerobot.yaml`` for
the full reference config.


Run It
------

**1. Configuration files**

We currently support DAgger training with the following configs:

- **MLP + ManiSkill**: ``examples/embodiment/config/maniskill_dagger_mlp.yaml``
- **Pi0 + LIBERO**: ``examples/embodiment/config/libero_spatial_dagger_openpi.yaml``
- **Pi0 + LIBERO (online LeRobot)**: ``examples/embodiment/config/libero_spatial_dagger_openpi_lerobot.yaml``
- **Pi0 + RoboTwin**: ``examples/embodiment/config/robotwin_adjust_bottle_dagger_openpi.yaml``

**2. Key DAgger Parameters**

.. code:: yaml

   algorithm:
     dagger:
       only_save_expert: False   # Expert acts with probability beta and also relabels student steps
       init_beta: 1.0
       beta_schedule: "exponential"
       beta_decay: 0.99
       beta_min: 0.05            # Optional; defaults to 0.05 in code

     replay_buffer:
       enable_cache: True
       cache_size: 2000
       min_buffer_size: 16
       sample_window_size: 2000

For the MLP ManiSkill example, the config uses a larger replay buffer and
``beta_decay: 0.98`` by default. Check the YAML file you launch for the exact
values.

**Online LeRobot parameters**

The online LeRobot config does not use ``algorithm.replay_buffer``. Tune the
``algorithm.dagger.online_lerobot`` fields instead:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Key
     - Meaning
   * - ``online_lerobot.enabled``
     - Turn on in-memory LeRobot episode collection during DAgger training.
   * - ``online_lerobot.only_success``
     - Keep only successful episodes; failed rollouts are discarded.
   * - ``online_lerobot.data_path``
     - Root directory for LeRobot shards written by the rolling dataset.
   * - ``online_lerobot.finalize_interval``
     - Finalize LeRobot metadata every N completed episodes.
   * - ``rolling_lerobot_window_size``
     - Maximum number of frames kept in the rolling training window.
   * - ``enable_decoded_cache``
     - Cache decoded frames to speed up actor DataLoader sampling.

**3. Launch Commands**

Use the same config name with either launcher:

**Sync Mode**

::

   bash examples/embodiment/run_embodiment.sh maniskill_dagger_mlp
   bash examples/embodiment/run_embodiment.sh libero_spatial_dagger_openpi
   bash examples/embodiment/run_embodiment.sh libero_spatial_dagger_openpi_lerobot
   bash examples/embodiment/run_embodiment.sh robotwin_adjust_bottle_dagger_openpi
   # For RoboTwin, add the following two commands before running the .sh file:
   # export ROBOT_PLATFORM=ALOHA export ROBOTWIN_PATH=/path/to/RoboTwin

**Async Mode**

::

   bash examples/embodiment/run_async.sh maniskill_dagger_mlp
   bash examples/embodiment/run_async.sh libero_spatial_dagger_openpi
   bash examples/embodiment/run_async.sh libero_spatial_dagger_openpi_lerobot
   bash examples/embodiment/run_async.sh robotwin_adjust_bottle_dagger_openpi
   # For RoboTwin, add the following two commands before running the .sh file:
   # export ROBOT_PLATFORM=ALOHA export ROBOTWIN_PATH=/path/to/RoboTwin

Visualization and Visualization and Results
-------------------------------------------

**1. TensorBoard Logs**

.. code-block:: bash

   tensorboard --logdir ./logs

**2. Useful Monitoring Metrics**

For metric definitions, see :doc:`Training metrics <../../reference/metrics>`. DAgger-specific metrics:

- ``env/success_once``: Recommended success metric for embodied DAgger runs.
- ``train/dagger/actor_loss``: Supervised DAgger loss on replayed expert-labeled samples.
- ``train/actor/lr``: Learning rate.
- ``train/actor/grad_norm``: Gradient norm.
- ``train/replay_buffer/num_trajectories``: Number of trajectories stored in the replay buffer.
- ``train/replay_buffer/total_samples``: Number of replay-buffer samples available for training.
- ``train/replay_buffer/cache_size``: Number of cached flattened trajectories.

Visualization and Results
-------------------------

.. csv-table::
   :header: "Configuration", "Student init SR", "Expert SR", "Training Time", "Student final SR"

   "MLP + ManiSkill", "0%", "100%", "20min", "100%"
   "Pi0 + LIBERO", "60%", "95%", "17h", "93%"
