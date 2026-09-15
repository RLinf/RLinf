Checkpoint Resume
=================

Resume an interrupted training job by setting ``runner.resume_dir`` to a saved
checkpoint. RLinf saves checkpoints every ``runner.save_interval`` steps. Use
the backend-specific layouts below to locate your checkpoint, then relaunch
training with the same configuration.


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

FSDP/FSDP2 Checkpoint's file structure looks like this:

.. code-block:: text

   experiment_name/checkpoints/
   ├── global_step_10/
   │   └── actor/
   │       ├── dcp_checkpoint/
   │       │   ├── .metadata
   │       │   ├── __0_0.distcp
   │       │   ├── __1_0.distcp
   │       │   ├── __2_0.distcp
   │       │   └── __3_0.distcp
   │       └── model_state_dict/
   │           └── full_weights.pt
   └── global_step_20/
       └── …

FSDP/FSDP2 saves model parameters, optimizer state, learning-rate schedulers,
and random-number generator (RNG) states through DCP
(``torch.distributed.checkpoint``). Keep all ``.distcp`` files and the
``.metadata`` file when copying a checkpoint.

DCP checkpoints preserve a separate RNG state for each actor rank: Python,
NumPy, PyTorch CPU, and the worker's current accelerator device when available.
Resume with the same actor world size to restore each rank's next random draws.
When the world size changes, existing ranks restore their saved streams and new
ranks keep their initialized RNG states. Loading continues with a warning;
the changed topology does not reproduce the original training sequence.
Environment state and separately created generators are outside this RNG
snapshot, so restoring it does not by itself guarantee identical training curves.

Older DCP checkpoints with a single RNG dictionary remain loadable. They may
have deduplicated different ranks' RNG states; loading them cannot recover the
discarded streams. The ``local_shard`` checkpoint format continues to store RNG
state in each rank's own file.


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
