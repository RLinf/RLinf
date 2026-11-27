DAgger
======

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

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: LIBERO (Pi0)
      :sync: libero

      Install the environment and download a pretrained Pi0 student checkpoint (the expert checkpoint can come from a PPO run in :doc:`pi0`):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate
         hf download RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT --local-dir ./RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

      Set the student and expert paths in ``examples/embodiment/config/libero_spatial_dagger_openpi.yaml`` (or ``libero_spatial_dagger_openpi_lerobot.yaml`` for online LeRobot; rollout inherits ``actor.model.model_path`` automatically):

      .. code-block:: yaml

         actor:
           model:
             model_path: ./RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

         rollout:
           expert_model:
             model_path: /path/to/expert_model

      Launch sync or async training (append ``_lerobot`` for the online LeRobot variant):

      .. code-block:: bash

         # Sync mode
         bash examples/embodiment/run_embodiment.sh libero_spatial_dagger_openpi
         bash examples/embodiment/run_embodiment.sh libero_spatial_dagger_openpi_lerobot

         # Async mode
         bash examples/embodiment/run_async.sh libero_spatial_dagger_openpi
         bash examples/embodiment/run_async.sh libero_spatial_dagger_openpi_lerobot

   .. tab-item:: RoboTwin (Pi0)
      :sync: robotwin

      Install the environment and download a pretrained Pi0 student checkpoint (set up RoboTwin code and assets as in :doc:`robotwin`, and obtain an expert checkpoint from :doc:`pi0`):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env robotwin && source .venv/bin/activate
         hf download RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle --local-dir ./RLinf-Pi0-RoboTwin-SFT-adjust_bottle

      Set the student, expert, and RoboTwin asset paths in ``examples/embodiment/config/robotwin_adjust_bottle_dagger_openpi.yaml``:

      .. code-block:: yaml

         actor:
           model:
             model_path: ./RLinf-Pi0-RoboTwin-SFT-adjust_bottle

         rollout:
           expert_model:
             model_path: /path/to/expert_model

         env:
           train:
             assets_path: ../RoboTwin
           eval:
             assets_path: ../RoboTwin

      Export ``ROBOTWIN_PATH``, then launch sync or async training:

      .. code-block:: bash

         export ROBOTWIN_PATH="$(cd ../RoboTwin && pwd)"
         # Sync mode (or use run_async.sh for async mode)
         bash examples/embodiment/run_embodiment.sh robotwin_adjust_bottle_dagger_openpi

   .. tab-item:: ManiSkill (MLP)
      :sync: maniskill

      Install the environment (reuses the ManiSkill / LIBERO environment):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate

      Set the expert checkpoint (from a PPO run in :doc:`mlp`) and optional student warm start in ``examples/embodiment/config/maniskill_dagger_mlp.yaml``:

      .. code-block:: yaml

         runner:
           ckpt_path: null                       # Optional student warm start
           expert_ckpt_path: /path/to/expert_ckpt

      Launch sync or async training:

      .. code-block:: bash

         # Sync mode (or use run_async.sh for async mode)
         bash examples/embodiment/run_embodiment.sh maniskill_dagger_mlp

.. dropdown:: Online LeRobot DAgger

   The classic configs store expert-labeled trajectories in an in-memory **replay buffer**. ``libero_spatial_dagger_openpi_lerobot.yaml`` adds an **online LeRobot** path that collects completed episodes in LeRobot format, streams them to the actor in memory, and trains from a frame-level rolling window.

   **Classic replay buffer vs online LeRobot**

   Both paths optimize the same DAgger objective and differ only in data storage and sampling:

   - **Classic replay buffer** — stores full trajectories in an in-memory buffer and samples with a trajectory-level sliding window.
   - **Online LeRobot** — collects complete episodes in LeRobot format and samples with a frame-level rolling window, supporting native Pi0 action-chunk supervision and optional success-only filtering.

   Prefer the online LeRobot path for Pi0 or other chunk-based models, when you need success-only filtering, or when bridging online DAgger with offline SFT or real-robot datasets.

   **Online LeRobot pipeline**

   1. **Mixed rollout and expert relabeling** — same ``beta`` scheduling and expert relabeling as classic DAgger.
   2. **Episode collection** — when ``algorithm.dagger.online_lerobot.enabled`` is ``True``, the trajectory collector accumulates per-env frames and exports completed episodes.
   3. **Actor ingestion** — completed episodes stream to the actor and append to the rolling LeRobot dataset.
   4. **Rolling-window training** — the actor samples from the rolling dataset (optionally with a decoded frame cache) and optimizes ``embodied_dagger`` loss.

   **Offline vs online collection**

   - **Offline disk export** — ``env.train.data_collection.enabled`` writes LeRobot shards to disk for later SFT or HG-DAgger. See :doc:`../../guides/data_collection` and :doc:`hg-dagger`.
   - **Online in-memory collection** — ``algorithm.dagger.online_lerobot.enabled`` streams episodes directly to the actor during DAgger training. Do **not** enable ``env.train.data_collection`` on the same train env when online LeRobot is on.

.. dropdown:: Advanced configuration (optional)

   The shipped configs already include working defaults for beta scheduling, replay buffer sizing, and online LeRobot settings. Override them only when tuning a run.

   **Classic DAgger and replay-buffer parameters**

   .. code-block:: yaml

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

   For the MLP ManiSkill example, the config uses a larger replay buffer and ``beta_decay: 0.98`` by default. Check the YAML file you launch for the exact values.

   **Online LeRobot parameters**

   The online LeRobot config does not use ``algorithm.replay_buffer``. Tune ``algorithm.dagger.online_lerobot`` instead:

   .. code-block:: yaml

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

Visualization and Results
-------------------------

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

**3. Experimental Results**

.. csv-table::
   :header: "Configuration", "Student init SR", "Expert SR", "Training Time", "Student final SR"

   "MLP + ManiSkill", "0%", "100%", "20min", "100%"
   "Pi0 + LIBERO", "60%", "95%", "17h", "93%"
