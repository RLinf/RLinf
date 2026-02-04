Replay Buffer Tutorial
==============================

This tutorial focuses on **practical usage** and **configuration tips** for
`TrajectoryReplayBuffer`. For a fuller design overview and data-flow details,
see the API doc: :doc:`../../apis/replay_buffer`.

Quick Start
-----------

.. code-block:: python

   from rlinf.data.replay_buffer import TrajectoryReplayBuffer

   buffer = TrajectoryReplayBuffer(
       seed=1234,
       storage_dir="/path/to/buffer",
       enable_cache=True,
       cache_size=5,
       storage_format="pt",
       sample_window_size=100,
       save_trajectories=True,
   )

Common Parameters
-----------------

- `storage_dir`: trajectory storage directory; defaults to the log directory if not specified.
- `storage_format`: `pt` (default) or `pkl`.
- `enable_cache` / `cache_size`: enable cache and set its size for throughput.
- `sample_window_size`: sample from the most recent N trajectories; 0 means all.
- `save_trajectories`: whether to persist to disk; `False` requires cache and
  disables checkpoints.

Add Trajectories
----------------

.. code-block:: python

   # trajectories is List[Trajectory]
   buffer.add_trajectories(trajectories)

Key behavior during writes:

- generate `uuid` and `trajectory_id` for each trajectory
- update `_trajectory_index` and counters
- async save by background thread (when `save_trajectories=True`)

Sampling for Training
---------------------

.. code-block:: python

   batch = buffer.sample(num_chunks=256)
   # batch shape: [num_chunks, ...]

Sampling draws transitions within the window and returns a rollout-aligned batch dict.

Save and Load
-------------

.. code-block:: python

   buffer.save_checkpoint("/path/to/ckpt")

   buffer.load_checkpoint(
       "/path/to/ckpt",
       is_distributed=True,
       local_rank=0,
       world_size=4,
   )

Checkpoint saving is unavailable when `save_trajectories=False`.
On load, `storage_dir` is restored from metadata in the checkpoint.
The trajectory data is saved in format of `trajectory_{trajectory_id}_{model_weights_uuid}_{model_update_count}.{storage_format}`.

CLI Test
--------

.. code-block:: bash

   python rlinf/data/replay_buffer.py \
     --load-path /path/to/buffer \
     --num-chunks 1024 \
     --cache-size 10 \
     --enable-cache

This command loads a buffer checkpoint and samples once, printing batch keys and shapes.

Merge / Split Tool
------------------

Script path: `toolkits/replay_buffer/merge_or_split_replay_buffer.py`

.. code-block:: bash

   # Merge multiple ranks (interleaved by original trajectory_id)
   python toolkits/replay_buffer/merge_or_split_replay_buffer.py \
     --source-path /path/to/buffer \
     --save-path /path/to/merged \
     --copy

.. code-block:: bash

   # Split a single buffer by first N trajectories
   python toolkits/replay_buffer/merge_or_split_replay_buffer.py \
     --source-path /path/to/buffer \
     --save-path /path/to/split \
     --split-count 30 \
     --copy

Cleanup and Reset
-----------------

.. code-block:: python

   buffer.close()        # close async save thread
   buffer.clear()        # clear index and counters
   buffer.clear_cache()  # clear cache and close thread

Tips
----

- **Throughput**: enable cache and set `cache_size` to recent trajectories.
- **Data freshness**: use `sample_window_size` to limit the sampling window.
