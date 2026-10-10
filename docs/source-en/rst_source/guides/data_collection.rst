Data Collection
===============

There are two entry points, and both write the same directories. ``save_dir``
defaults to ``${runner.logger.log_path}``.

- **Real-robot teleoperation** starts ``collect_real_data.py`` through
  ``examples/embodiment/collect_data.sh``. ``export_format`` selects one store,
  or ``replay_buffer`` together with one episode format.
- **Training or evaluation** enables ``data_collection`` under ``env``. The env
  worker writes one episode format at a time.

The real-robot script accepts only ``replay_buffer``, ``pickle_episode``, and
``lerobot_dataset``. ``pickle_episode`` and ``lerobot_dataset`` cannot be
selected together. Omitting ``export_format`` writes ``replay_buffer`` and
``lerobot_dataset``. The training env worker also accepts ``pickle`` and
``lerobot`` as the latter two names.

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Data format
     - Save directory
     - Reader
   * - ``replay_buffer``
     - ``{save_dir}/demo_buffer/``
     - RLPD. Set ``algorithm.demo_buffer.load_path`` to this directory, for example Franka peg insertion and DoSW1.
   * - ``lerobot_dataset``
     - ``{save_dir}/lerobot_dataset/``
     - OpenPI SFT, HG-DAgger, and other imitation pipelines that read LeRobot shards.
   * - ``pickle_episode``
     - ``{save_dir}/pickle_episode/``
     - Reward-model preprocessing. ``examples/reward/preprocess_reward_dataset.py`` builds the ResNet binary dataset; ``examples/reward/vlm_trend/`` builds VLM trend labels.

To save a replay buffer and a LeRobot dataset from the real robot:

.. code-block:: yaml

   env:
     eval:
       data_collection:
         enabled: True
         save_dir: ${runner.logger.log_path}
         export_format: [replay_buffer, lerobot_dataset]

Streaming LeRobot collection, such as dual-arm YAM, sets
``export_format: lerobot_dataset`` and ``streaming: true``. Frames are written
as they are recorded, so memory does not grow with episode length. Successful
and failed episodes are both stored and stamped with ``is_success``;
``only_success`` does not apply while streaming. Operator discards and
recording-queue overflows are not published into the dataset.

To collect pickle episodes during training for a reward model:

.. code-block:: yaml

   env:
     eval:
       data_collection:
         enabled: True
         save_dir: ${runner.logger.log_path}
         export_format: pickle
         only_success: True

``maniskill_ppo_mlp_collect`` uses this setup. ``.pkl`` files land in
``{save_dir}/pickle_episode/``. With ``export_format: lerobot``, shards land in
``{save_dir}/lerobot_dataset/rank_*/id_*/``.

Real-robot teleoperation collection steps
------------------------------------------

Set the robot, teleoperation device, and target pose in
``examples/embodiment/config/realworld_collect_data.yaml``:

.. code-block:: yaml

   cluster:
     node_groups:
       hardware:
         configs:
           robot_ip: "192.168.1.100"
   env:
     eval:
       teleop: spacemouse          # spacemouse, gello, pico, or none
       override_cfg:
         target_ee_pose: [0.5, 0.0, 0.3, 0.0, 3.14, 0.0]
         success_hold_steps: 1     # consecutive steps at the target before success
   runner:
     num_data_episodes: 20         # stop after this many successes

Then run:

.. code-block:: bash

   bash examples/embodiment/collect_data.sh
   bash examples/embodiment/collect_data.sh realworld_collect_data_gello

The script exits after ``runner.num_data_episodes`` successes. When ``save_dir``
is the log directory, the tree is:

.. code-block:: text

   logs/{timestamp}/
   ├── demo_buffer/
   ├── lerobot_dataset/rank_0/id_0/
   └── pickle_episode/*.pkl

Each successful trajectory in ``demo_buffer`` is a ``.pt`` file whose
``intervene_flags`` are all ones, marking expert data. To append data, point a
later run at the same ``demo_buffer`` directory.

Inspect the outputs:

.. code-block:: bash

   python toolkits/replay_buffer/visualize.py \
       --replay_dir logs/{timestamp}/demo_buffer

   python toolkits/lerobot/visualize_lerobot_dataset.py \
       --dataset-path logs/{timestamp}/lerobot_dataset/rank_0/id_0 \
       --output-dir logs/{timestamp}/lerobot_visualized

Data format details
-------------------

``CollectEpisode`` writes on a background thread and flushes the remainder in
``env.close()``, so collection does not block a training step. Each env worker
uses its own ``rank``, which appears in filenames and shard directories.

**replay_buffer**

Only successful episodes are stored. On the real robot,
``manual_episode_control_only`` counts ``manual_done`` alone; otherwise
``reward >= 0.5`` or ``manual_done`` counts as success. ``recording_invalid``
and ``episode_discarded`` are neither stored nor counted. Each trajectory is a
``trajectory_*.pt`` file, next to ``metadata.json`` and
``trajectory_index.json``. ``intervene_flags`` are all ones so RLPD can tell
expert data from online policy data. A later run pointed at the same directory
appends trajectories. Tensor shapes follow the robot and cameras; they are not
part of the format.

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

Images stay in memory until the episode ends, so usage grows with length.

**pickle_episode**

``CollectEpisode`` writes one ``.pkl`` when the episode ends. With
``only_success: true`` and streaming off, failed episodes are dropped.
The filename is:

.. code-block:: text

   rank_{rank}_env_{env_idx}_episode_{episode_id}_step_{global_step}_{success|fail}.pkl

Contents:

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

``CollectEpisode`` writes one or more episodes into a shard at
``lerobot_dataset/rank_{rank}/id_{N}/``. ``resume: true`` continues numbering
from existing shards and does not rewrite finished data.

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
     - Other views. Empty when there is no extra view
   * - ``state`` / ``actions``
     - State and action vectors, ``float32``
   * - ``timestamp`` / ``frame_index``
     - Frame time in seconds, and the index within the episode
   * - ``episode_index`` / ``index`` / ``task_index``
     - Global episode number, global frame number, and task number
   * - ``done`` / ``is_success``
     - Whether this step is the last frame; whether the episode succeeded

The main image is taken from ``main_images``, then ``image``, then
``full_image``. Extra views come from ``extra_view_images``, then
``extra_view_image``; a stacked ``[N, H, W, C]`` tensor is split into
``extra_view_image-0``, ``extra_view_image-1``, and so on. State prefers
``states``, then ``state``. Float images in ``[0, 1]`` are multiplied by 255
and stored as uint8.

To wrap an environment in your own script, pass the parent directory as
``save_dir``. The wrapper adds the format subdirectory:

.. code-block:: python

   env = CollectEpisode(
       env=base_env,
       save_dir="./logs/run",
       export_format="lerobot",   # or "pickle"
       robot_type="panda",
       fps=10,
       only_success=True,
   )
   env.close()   # flush the remaining episodes

With ``streaming: true``, each frame is written as a PNG immediately and the
parquet is written when the episode ends. Successful and failed episodes are
both kept, and ``is_success`` is stamped at the end. An operator discard
deletes the current unpublished episode. A recording-queue overflow is moved
to ``invalid_episodes/`` inside that shard and is not published. A streaming
shard also has ``stream_frames.jsonl``, which records each frame's ``state``
and ``actions``.

Each info dict checks ``success_once``, ``success_at_end``, and ``success``
inside ``final_info`` and ``episode`` before the info root. The episode is
successful if any of those values is true. If none of the keys appear, the
flag maintained during collection is used.
