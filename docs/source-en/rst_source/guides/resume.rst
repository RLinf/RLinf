Checkpoint Resume
=================

Unexpected events—network errors, power loss, node pre-emptions—can
interrupt a long-running distributed job.  
To tackle this challenge, RLinf saves a full checkpoint every ``runner.save_interval`` steps and lets
you resume from the most recent snapshot with minimal loss of work.


Checkpoint layout
-----------------

Assume the following YAML fragment:

.. code-block:: yaml

   runner:
     task_type: math
     logger:
       log_path: ${runner.output_dir}/${runner.experiment_name}
       project_name: rlinf
       experiment_name: ${runner.experiment_name}

     save_interval: 50          
     experiment_name: grpo-1.5b
     output_dir: ./logs

If Megatron is used as the training backend, its checkpoints will appear under `output_dir/experiment_name/checkpoints/`,
while if FSDP/FSDP2 is used as the training backend, its checkpoints will appear under `log_path/experiment_name/checkpoints/`.

Megatron Checkpoints
~~~~~~~~~~~~~~~~~~~~~~

Megatron Checkpoint's file structure looks like this:

.. code-block:: text

   logs/grpo-1.5b/checkpoints/
   ├── global_step_50/
   │   ├── actor/
   │   │   ├── iter_0000050/
   │   │   │   ├── mp_rank_00/
   │   │   │   │   ├── distrib_optim.pt
   │   │   │   │   └── model_optim_rng.pt
   │   │   │   └── mp_rank_01/                 
   │   │   │       ├── distrib_optim.pt
   │   │   │       └── model_optim_rng.pt
   │   │   └── latest_checkpointed_iteration.txt
   │   └── data/
   │       └── data.pt                         
   └── global_step_100/
       └── …


Key points
^^^^^^^^^^^^^^^

* **Sharded weights** – files inside ``mp_rank_*`` follow the Megatron
  tensor-parallel layout; each GPU only reloads its own slice.
* **Optimizer / RNG state** – *both* the Adam parameters
  (``distrib_optim.pt``) *and* random-number generators are captured,
  guaranteeing bit-for-bit reproducibility after resume.
* **Data sampler** – ``data.pt`` stores dataloader, so no
  samples are skipped or repeated.



FSDP/FSDP2 Checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~

FSDP/FSDP2 uses either DCP (``torch.distributed.checkpoint``) or a per-rank
``local_shard`` checkpoint, depending on the actor worker. The default DCP
layout is:

.. code-block:: text

   experiment_name/checkpoints/
   ├── global_step_10/
   │   └── actor/
   │       ├── dcp_checkpoint/
   │       │   ├── __0_0.distcp
   │       │   ├── __1_0.distcp
   │       │   ├── __2_0.distcp
   │       │   └── __3_0.distcp
   │       └── model_state_dict/
   │           └── full_weights.pt
   └── global_step_20/
       └── …

DCP stores the training state across distributed checkpoint files (``.distcp``).
The optional ``model_state_dict/full_weights.pt`` export contains model weights;
resume training from the complete checkpoint directory to restore optimizer,
scheduler, and RNG state as well.

Some workers save one file per rank instead. SAC and DAgger select
``local_shard`` when ``actor.fsdp_config.use_orig_params`` is true, while IQL
uses it for its policy, critic, and value models. For example, the SAC actor
directory for two ranks contains:

.. code-block:: text

   global_step_10/actor/
   └── local_shard_checkpoint/
       ├── checkpoint_rank_0.pt
       └── checkpoint_rank_1.pt

Each file contains that rank's model shards, optimizer state, scheduler state,
and RNG state. FSDP2 restores the saved local tensors into the current model's
distributed parameters, including parameters split unevenly across ranks.
Keep the same model, FSDP version, world size, sharding configuration, and
rank-to-shard mapping when resuming from this format. ``local_shard`` does not
redistribute shards for a different topology. Preserve every rank file and
use the same launcher and configuration with ``runner.resume_dir`` as below;
the worker selects the corresponding checkpoint format automatically.


Resuming training
-----------------

1. **Choose the latest checkpoint**

   If ``global_step_10/`` is the highest numbered directory it is the
   newest snapshot.

2. **Edit the YAML**

   .. code-block:: yaml

      runner:
        resume_dir: ${runner.output_dir}/${runner.experiment_name}/checkpoints/global_step_10


3. **Relaunch exactly as before**

   Start Ray, then the same ``run_main_*.sh`` launcher. 
   RLinf will automatically detect the ``resume_dir`` and:

   * Restores model shards, optimizer, RNG and dataloader state on every
     node/rank.
   * Continues step counting from ``global_step_10`` — your next saved
     checkpoint will be ``global_step_20`` (because ``save_interval`` is
     10).

.. tip::

   To verify resumption, look for the log line.  
   If the next training step starts at 30, then the resume is working well!


Select and resume the best SFT checkpoint
-----------------------------------------

Use ``runner.early_stop`` to select ``best_model/`` by a validation metric and
optionally stop after repeated checks without improvement:

.. code-block:: yaml

   runner:
     early_stop:
       enabled: true
       monitor: value_spearman
       mode: max
       patience: 5
       min_delta: 0.001

Set ``mode: min`` for a loss or ``mode: max`` for a score. The metric name must
match a key returned by the worker's validation method. A missing key raises an
error listing the available metrics. An improvement must exceed ``min_delta``;
non-finite values count toward patience and never select a checkpoint.

If you omit ``mode``, it defaults to ``max`` for ``val_accuracy`` and ``min``
for every other metric. Set ``enabled: false`` to keep selecting the best
checkpoint while training to the configured step limit.

SFT checkpoints include ``runner_state.json`` alongside ``actor/``. It records
the step, best metric value, and patience counter after validation. Set
``runner.resume_dir`` to either ``global_step_<N>/`` or ``best_model/`` to
restore them together with the worker checkpoint. Keep ``monitor``, ``mode``,
and ``min_delta`` unchanged when resuming. A checkpoint that has already
exhausted patience performs no further training steps. When periodic saving is
enabled, early stopping also writes a checkpoint at the stopping step.

Older ``global_step_<N>/`` checkpoints still load; a warning explains that
best-model tracking and patience start fresh because those checkpoints contain
no runner state. The first finite validation value then replaces ``best_model/``.
A save interrupted while ``runner_state.incomplete`` exists is rejected on resume.
Worker-specific data-loader and RNG restoration follows the
worker's checkpoint support.
