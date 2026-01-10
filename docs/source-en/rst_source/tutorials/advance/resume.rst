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

FSDP/FSDP2 Checkpoint's file structure looks like this:

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
   │           └── full_weigths.pt
   └── global_step_20/
       └── …

FSDP/FSDP2 saves and loads checkpoints via DCP (torch.distributed.checkpoint), resulting in a set of distributed checkpoint files (.distcp).
Each file contains a slice of model parameters, optimizer state, and RNG state.


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


