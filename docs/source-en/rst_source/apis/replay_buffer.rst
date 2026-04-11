Replay Buffer
======================

This section introduces the design, core data structures, and sampling flow of
`TrajectoryReplayBuffer` in RLinf. It is a **trajectory-based** replay buffer
that stores batched trajectories with shape `[T, B, ...]`. It does not split at
the episode level; each trajectory can contain multiple episodes. Sampling is
performed at the chunk-step granularity.

Design Goals
------------

- **Low memory usage**: the buffer maintains only trajectory indices in memory.
- **Low I/O overhead**: trajectories are stored on disk; metadata and indices are
  separated; async writes reduce training stalls.
- **High-throughput sampling**: windowed sampling with vectorized index mapping.
- **Extensibility**: supports caching, storage formats, and distributed loading.
- **Training-friendly**: outputs align with rollout batch format (dict keys and
  tensor structures stay consistent).

Core Data Structures
--------------------

Trajectory Index
^^^^^^^^^^^^^^^^

Each trajectory entry includes:

- `uuid`: unique identifier
- `trajectory_id`: increasing integer ID
- `num_samples`: number of samples in the trajectory (`T * B`)
- `shape`: trajectory tensor shape
- `max_episode_length`: maximum episode length

Index and metadata files:

- `trajectory_index.json`: trajectory index
- `metadata.json`: buffer metadata (total samples, format, seed, etc.)

Trajectory Cache
^^^^^^^^^^^^^^^^

`TrajectoryCache` is a FIFO cache that stores **flattened** trajectories:

- trajectories are stored as `[T, B, ...]`
- flattened to `[T*B, ...]` for transition-level sampling
- only keeps the most recent trajectories to avoid frequent disk reads

Sampling Flow
-------------

Sampling is performed via `sample(num_chunks)` and returns a batch with shape
`[num_chunks, ...]`.

1. **Sliding window**: sample from the most recent `sample_window_size`
   trajectories (0 means all).
2. **Uniform sampling**: sample indices over the window's total samples.
3. **Index mapping**: map global indices to per-trajectory indices (`bucketize`).
4. **Batch sampling**: load each trajectory once and sample in batch to build
   `[num_chunks, ...]`.

This design reduces repeated loads while keeping the sampling distribution smooth.

Data Flow Overview
------------------

Multi-round interactions between the model and environment are collected by the
Rollout Worker, converted into `Trajectory`, and sent through the Channel to the
Actor Worker. The Actor Worker stores the trajectories in `TrajectoryReplayBuffer`
and samples batches for training.

Rollout Worker: Trajectory Collection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The rollout worker accumulates step results over time. Here, `[obs_0, obs_1]`
represents a transition, and `[act_0, r_0, dones, ...]` represents a
**ChunkStepResult**.

.. code-block:: text

   time ->
   [ obs_0, obs_1, act_0, r_0, dones_0, ... ] --┐
   [ obs_1, obs_2, act_1, r_1, dones_1, ... ] --┼-- EmbodiedRolloutResult -- Trajectory
   [ obs_2, obs_3, act_2, r_2, dones_2, ... ] --┤                                |
   [ obs_3, obs_4, act_3, r_3, dones_3, ... ] --┘                           Channel(put)

Actor Worker: Trajectory Storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The actor worker receives `Trajectory` from the Channel, stores them in
`TrajectoryReplayBuffer`, and samples batches for training.

.. code-block:: text

   Channel(get) -- Trajectory -- ReplayBuffer
                                    |
                                    |- add_trajectories(trajectories)
                                    |    |- generate uuid + trajectory_id
                                    |    |- update _trajectory_index / counters
                                    |    `- async save by thread pool
                                    |
                                    `- sample(num_chunks) --> training batch

Storage and Async Writing
-------------------------

Supported formats:

- `pt`: default `torch.save`
- `pkl`: `pickle`

Trajectories are written asynchronously via a single-thread `ThreadPoolExecutor`.
Metadata and indices are updated after writes complete to reduce I/O stalls.

Checkpoint and Distributed Loading
----------------------------------

The buffer supports checkpointing and distributed loading by rank:

- `load_path`: path to the checkpoint directory that contains both metadata and trajectory files
- `is_distributed`: enable sharded loading
- `local_rank`: load the shard for the current rank (0-based)
- `world_size`: total number of ranks (shard count)
- maintain consistency for `size`, `total_samples`, and `trajectory_counter`

Usage Tips
----------

- **Long trajectories**: prefer windowed sampling to reduce stale data.
- **High concurrency**: enable cache to improve sampling throughput.
- **No persistence needed**: set `auto_save=False`; cached trajectories and metadata
  are saved into the checkpoint path when `save_checkpoint` is called.
