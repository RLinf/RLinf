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


Control FSDP SFT Save Costs
----------------------------------------------------------------

When frequent policy exports are useful but saving optimizer state each time
is too costly, set ``actor.fsdp_config.training_state_save_interval`` to a
positive integer. ``runner.save_interval`` still schedules exports; training
state is included at export steps divisible by the new interval. Leaving the
new field unset preserves full training-state saves on every checkpoint.
Choose aligned intervals, for example:

.. code-block:: yaml

   runner:
     save_interval: 100
   actor:
     fsdp_config:
       save_full_model_weights: true
       training_state_save_interval: 1000
       checkpoint_format: dcp
       checkpoint_communication_backend: gloo

This exports weights every 100 steps and saves recoverable training state at
every 1,000-step export. The SFT runner forces training state at a final save
or an early-stop save when checkpointing is enabled. Between those saves,
``save_full_model_weights: true`` is required because only inference weights
are written. For this opt-in SFT path, ``actor/checkpoint_metadata.json`` is
published with ``complete: false`` before saving and ``complete: true`` after
the worker finishes. A checkpoint with ``save_training_state: false`` cannot
resume training; select an earlier complete checkpoint containing state.

The optional ``checkpoint_communication_backend: gloo`` creates a temporary
CPU group for DCP and RNG/data object collectives. It requires
``checkpoint_format: dcp``; leaving it unset uses the existing process group.
It does not replace the training communication backend. Model-state extraction
and the surrounding training barriers may still use the training group.

For OpenPI, enable ``data.stateful_loader`` from the start to restore the data
stream as well as model, optimizer and RNG state. Its loader manifest is an
additional completion requirement; see
:doc:`OpenPI SFT <../examples/embodied/sft_openpi>`.
