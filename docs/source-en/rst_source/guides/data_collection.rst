Data Collection
===============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/data_formats.svg
   :align: center
   :width: 100%
   :alt: RLinf data collection sources, stored formats, and downstream consumers

   Data collection sources, stored formats, and downstream consumers.

Record real-robot demonstrations or simulator rollouts into a shared output directory (``save_dir``, which defaults to ``${runner.logger.log_path}``). RLinf provides two collection entry points:

- **Real-robot teleoperation**: Run ``examples/embodiment/collect_data.sh`` to record demonstrations. Set ``export_format`` to one format, or pair ``replay_buffer`` with one episode format (omitting it writes ``replay_buffer`` and ``lerobot_dataset``).

- **Training or evaluation**: Enable ``data_collection`` under ``env`` in a training or evaluation YAML. Each env worker writes one episode format at a time: ``lerobot_dataset`` or ``pickle_episode``.

Both entry points share the same three format names. You cannot combine ``pickle_episode`` and ``lerobot_dataset`` in one run, and env workers during training or evaluation do not write ``replay_buffer``.

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Data format
     - Save directory
     - Downstream consumer
   * - ``replay_buffer``
     - ``{save_dir}/demo_buffer/``
     - RLPD training via ``algorithm.demo_buffer.load_path`` (such as Franka peg insertion and DOS-W1).
   * - ``lerobot_dataset``
     - ``{save_dir}/lerobot_dataset/``
     - OpenPI SFT, HG-DAgger, and imitation pipelines that read LeRobot shards.
   * - ``pickle_episode``
     - ``{save_dir}/pickle_episode/``
     - Reward-model preprocessing: ``examples/reward/preprocess_reward_dataset.py`` builds ResNet binary datasets, and ``examples/reward/vlm_trend/`` builds VLM trend labels.

Collect Real-Robot Data
-----------------------

Configure the robot IP, teleoperation device, target pose, export format, and episode count in ``examples/embodiment/config/realworld_collect_data.yaml``:

.. code-block:: yaml

   cluster:
     node_groups:
       - label: franka
         hardware:
           configs:
             - robot_ip: "192.168.1.100"
   env:
     eval:
       teleop: spacemouse          # spacemouse, gello, pico, or none
       override_cfg:
         target_ee_pose: [0.5, 0.0, 0.3, 0.0, 3.14, 0.0]
         success_hold_steps: 1     # consecutive steps at the target before success
       data_collection:
         enabled: True
         save_dir: ${runner.logger.log_path}
         export_format: [replay_buffer, lerobot_dataset]
   runner:
     num_data_episodes: 20         # stop after this many successes

Start collection from the repository root (pass another config name as the first argument when needed):

.. code-block:: bash

   bash examples/embodiment/collect_data.sh
   bash examples/embodiment/collect_data.sh realworld_collect_data_gello

The script exits after ``runner.num_data_episodes`` successful episodes. Outputs land under ``logs/{timestamp}/demo_buffer/`` (successful ``.pt`` trajectories marked as expert data) and ``logs/{timestamp}/lerobot_dataset/rank_0/id_0/`` (or ``pickle_episode/*.pkl`` when exporting pickle episodes).

Point a later run at the same ``demo_buffer`` directory to append trajectories. For LeRobot datasets, set ``resume: true`` to continue shard numbering without overwriting finished data.

Collect Data During Training or Evaluation
------------------------------------------

Enable ``data_collection`` under ``env.eval`` (or ``env.train``) in a training or evaluation YAML. For example, ``maniskill_ppo_mlp_collect`` records successful pickle episodes for reward-model training:

.. code-block:: yaml

   env:
     eval:
       data_collection:
         enabled: True
         save_dir: ${runner.logger.log_path}
         export_format: pickle_episode
         only_success: True

Pickle episodes are written as ``.pkl`` files under ``{save_dir}/pickle_episode/``. Setting ``export_format: lerobot_dataset`` writes shards under ``{save_dir}/lerobot_dataset/rank_*/id_*/``.

Merge Streaming Shards
----------------------

For long episodes such as dual-arm YAM, set ``export_format: lerobot_dataset`` and ``streaming: true``. Frames are written to disk as they arrive, so memory stays constant regardless of episode length.

Each kept take becomes its own shard, with both successful and failed takes stored and labeled with ``is_success`` (``only_success`` does not filter during streaming). Operator discards and recording-queue overflows are excluded from the published dataset.

Before training, merge the shards into one dataset and drop failed takes:

.. code-block:: bash

   python -m toolkits.lerobot.merge_lerobot_datasets \
       --source-dir logs/{timestamp}/lerobot_dataset \
       --output-dir logs/{timestamp}/lerobot_merged \
       --only-success

Inspect Collected Data
----------------------

Visualize recorded replay buffers or LeRobot shards to check camera frames, states, and actions before training:

.. code-block:: bash

   python -m toolkits.replay_buffer.visualize \
       --replay_dir logs/{timestamp}/demo_buffer

   python -m toolkits.lerobot.visualize_lerobot_dataset \
       --dataset-path logs/{timestamp}/lerobot_dataset/rank_0/id_0 \
       --output-dir logs/{timestamp}/lerobot_visualized

.. dropdown:: On-disk format details and custom wrapper usage

   The episode collector writes on a background thread and flushes remaining data when the environment closes, so disk I/O does not block training steps. Each env worker uses its own ``rank``, which appears in filenames and shard directories.

   **replay_buffer**

   Only successful episodes are saved. On real robots, ``manual_episode_control_only`` counts ``manual_done`` alone; otherwise ``reward >= 0.5`` or ``manual_done`` counts as success.

   Invalid recordings and discarded takes (``recording_invalid`` and ``episode_discarded``) are neither saved nor counted. Each trajectory is stored as ``trajectory_*.pt`` beside ``metadata.json`` and ``trajectory_index.json``.

   Every transition is flagged as expert intervention data so RLPD separates demonstrations from online policy rollouts. Tensor shapes follow the robot and cameras, and images stay in memory until the episode ends.

   .. code-block:: python

      {
          "transitions": {
              "obs": {"states", "main_images"},
              "next_obs": {"states", "main_images"},
              "action": "float32[T, action_dim]",
              "rewards": "float32[T, 1]",
              "dones": "bool[T, 1]",
              "terminations": "bool[T, 1]",
              "truncations": "bool[T, 1]",
          },
          "intervene_flags": "all ones",
      }

   **pickle_episode**

   The collector writes one ``.pkl`` file when each episode ends. With ``only_success: true`` and streaming off, failed episodes are dropped.

   .. code-block:: text

      rank_{rank}_env_{env_idx}_episode_{episode_id}_step_{global_step}_{success|fail}.pkl

   File contents:

   .. code-block:: python

      {
          "rank": int,
          "env_idx": int,
          "episode_id": int,
          "step": int,
          "success": bool,
          "observations": list,  # length = num_steps + 1; item 0 comes from reset()
          "actions": list,
          "rewards": list,
          "terminated": list,
          "truncated": list,
          "infos": list,         # reward-model preprocessing reads the main image and per-step success
      }

   **lerobot_dataset**

   One or more episodes are written into a shard at ``lerobot_dataset/rank_{rank}/id_{N}/`` (streaming mode writes one kept take per shard). Setting ``resume: true`` continues numbering from existing shards without modifying finished data.

   .. code-block:: text

      id_0/
      ├── meta/info.json          # fps, robot_type, dimensions
      ├── meta/episodes.jsonl
      ├── meta/tasks.jsonl
      ├── meta/stats.json
      └── data/chunk-000/episode_*.parquet

   .. list-table::
      :header-rows: 1
      :widths: 28 72

      * - Column
        - Contents
      * - ``image``
        - Main camera image, uint8
      * - ``extra_view_image`` / ``extra_view_image-N``
        - Other views; empty when there is no extra view
      * - ``state`` / ``actions``
        - State and action vectors, ``float32``
      * - ``timestamp`` / ``frame_index``
        - Frame time in seconds, and index within the episode
      * - ``episode_index`` / ``index`` / ``task_index``
        - Global episode number, global frame number, and task number
      * - ``done`` / ``is_success``
        - Whether this step is the last frame; whether the episode succeeded

   Observation key fallback order:

   - **Main image**: reads ``main_images``, then ``image``, then ``full_image``. Float images in ``[0, 1]`` are scaled by 255 to uint8.

   - **Extra views**: reads ``extra_view_images``, then ``extra_view_image``. A stacked ``[N, H, W, C]`` tensor splits into ``extra_view_image-0``, ``extra_view_image-1``, and so on.

   - **State**: reads ``states``, then ``state``.

   Streaming storage and success detection:

   - With ``streaming: true``, each frame's camera views are written immediately as uncompressed PNGs, and the parquet file is written when the episode ends. Because parquet keeps those raw PNG bytes, a 640x480 view takes about 0.9 MB per frame on disk.

   - Both successful and failed takes are stored with ``is_success`` stamped at the end, alongside ``stream_frames.jsonl`` for per-frame states and actions. Operator discards delete the unpublished episode, and recording-queue overflows move the segment into ``invalid_episodes/`` inside that shard without publishing it.

   - Per-step success checks nested ``final_info`` and ``episode`` dicts before the top-level info dict.

   - An episode counts as successful if ``success_once``, ``success_at_end``, or ``success`` is true; if none of those keys appear, the collector uses its running success flag.

   **Wrapping an environment in a custom script**

   Pass the parent directory as ``save_dir`` (the wrapper creates the format subdirectory automatically) and close the environment at the end to flush remaining episodes:

   .. code-block:: python

      env = CollectEpisode(
          env=base_env,
          save_dir="./logs/run",
          export_format="lerobot_dataset",   # or "pickle_episode"
          robot_type="panda",
          fps=10,
          only_success=True,
      )
      env.close()   # flush the remaining episodes
