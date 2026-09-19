OpenWAM Supervised Fine-Tuning
===============================

Use this recipe to fine-tune OpenWAM on the native LIBERO dataset through RLinf's Ray-managed FSDP runner. The recipe loads OpenWAM's checkpoint directory, reuses its native dataloader and joint video/action loss, and trains the unfrozen modules with full-shard FSDP.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      OpenWAM Wan2.2

   .. grid-item-card:: Method
      :text-align: center

      Full-parameter SFT

   .. grid-item-card:: Data
      :text-align: center

      Native LIBERO reader

   .. grid-item-card:: Hardware
      :text-align: center

      4 GPUs recommended with FSDP2

The OpenWAM checkpoint supplies the model and dataloader settings. Set ``data.train_data_paths`` to the dataset root (or a list of roots: each is read with the same dataloader settings and the windows are concatenated, so the mixture is sampled in proportion to size); the loader reads ``config.yaml`` from ``actor.model.model_path`` and keeps the native frame, action, and normalization conventions.

Installation
------------

Install the OpenWAM environment and RLinf:

.. code:: bash

   bash requirements/install.sh embodied --model openwam --env libero
   source .venv/bin/activate

Run It
------

Set the checkpoint and dataset paths in ``examples/sft/config/model/openwam.yaml`` and ``examples/sft/config/libero_sft_openwam.yaml``. The checked-in recipe maps the actor to GPUs ``0-3``. Four ranks are recommended because OpenWAM keeps its trainable DiT/action parameters in the root FSDP2 unit; a two-card run can exceed the memory budget of 80-GiB GPUs.

The dataset reader comes from the checkpoint's ``config.yaml``, so a recipe pairs a checkpoint with data of the same type. One recipe per OpenWAM reader ships under ``examples/sft/config/``; each inherits ``libero_sft_openwam.yaml`` and only sets the paths and the experiment name:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Recipe
     - Data
   * - ``libero_sft_openwam``
     - LIBERO (LeRobot v3, EEF10 actions)
   * - ``agibotworld_sft_openwam``
     - AgibotWorld
   * - ``interndata_a1_sft_openwam``
     - InterData A1
   * - ``mixture_sft_openwam``
     - Mixed OpenWAM reader data
   * - ``muka_franka_sft_openwam``
     - Muka Franka
   * - ``oxe_droid_sft_openwam``
     - OXE DROID
   * - ``robocoin_sft_openwam``
     - RoboCoin
   * - ``robotwin_sft_openwam``
     - RoboTwin 2.0 (aloha-agilex, 20-D dual-arm EEF)
   * - ``robodojo_sft_openwam``
     - RoboDojo real-robot data
   * - ``ebench_sft_openwam``
     - EBench
   * - ``robocasa365_sft_openwam``
     - RoboCasa365
   * - ``robocasa_gr1_sft_openwam``
     - RoboCasa GR1 humanoid
   * - ``vlabench_sft_openwam``
     - VLABench

The base recipe enables ``fsdp_config.gradient_checkpointing: true`` and forwards it to OpenWAM's own block checkpointing (``use_gradient_checkpointing``). It also uses ``global_batch_size: 8`` with ``micro_batch_size: 1`` so the effective batch grows without increasing per-rank activation memory.

Start the Ray-managed FSDP runner:

.. code:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_openwam

Override ``cluster.component_placement.actor`` and keep ``actor.global_batch_size`` divisible by the actor world size when you change the GPU count.

The preset loads the weights in fp32 (``precision: fp32``) so the optimizer keeps fp32 master weights while FSDP computes in bf16 (``mixed_precision.param_dtype``); bf16 master weights would round away almost every update at ``lr: 1e-6``. The policy is a single root FSDP2 unit because OpenWAM's joint denoising driver reads block weights outside their forward. ``reshard_after_forward`` only applies to the frozen ``ResidualBlock`` subunits named by the wrap policy; the trainable root parameters remain resident for the joint forward and backward. The checked-in four-rank placement and gradient checkpointing are therefore part of the memory budget, rather than a guarantee that a two-card 80-GiB run will fit. The model preset also keeps ``load_to_device: false``: every rank builds the policy on the CPU and FSDP moves its shard to the GPU while wrapping. The evaluation recipes load straight onto the GPU with ``load_to_device: true``.

Validation and resuming
-----------------------

Set ``data.val_data_paths`` (one dataset root or a list, read with the same dataloader settings) and ``runner.val_check_interval`` to report ``eval/loss``, ``eval/loss_video`` and ``eval/loss_action`` averaged over the validation loader. ``actor.eval_batch_size`` sets the per-rank validation batch and ``actor.eval_max_batches`` caps the number of validation batches per rank for large datasets. LeRobot-style readers select episodes by split: validation reads the ``val`` split by default, so set ``data.openwam_val_split`` to ``train`` when the validation root is a separate held-out dataset that only ships a train split (an empty validation set is rejected at start-up).

Checkpoints store the data loader, sampler (including the shuffle epoch) and RNG states next to the model weights, so ``runner.resume_dir=<log_path>/<experiment_name>/checkpoints/global_step_<N>`` continues with the next unseen batch instead of restarting the epoch. The OpenWAM recipe sets ``runner.strict_resume: true`` and fails if an older checkpoint has no ``data.pt`` or ``rng.pt``; unset it only when restarting the data stream is intentional.

Visualization and Results
-------------------------

Monitor ``train/loss``, ``train/loss_video``, and ``train/loss_action`` in TensorBoard. RLinf writes FSDP model and optimizer shards under ``runner.logger.log_path/<experiment_name>/checkpoints/global_step_<N>/actor``.

Export for deployment
---------------------

The FSDP worker stores the whole ``OpenWAMPolicy`` state dict at ``<log_path>/<experiment_name>/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt`` (a sharded ``dcp_checkpoint`` directory is consolidated automatically). Rebuild a self-contained OpenWAM checkpoint directory from it:

.. code-block:: bash

   python toolkits/openwam/export_checkpoint.py \
       --rlinf-checkpoint ../results/libero_sft_openwam/checkpoints/global_step_1000 \
       --source-checkpoint /path/to/openwam-libero-sft-30000 \
       --output /path/to/openwam-libero-sft-rlinf-step1000 --link-assets --verify cuda

The exporter strips the ``architecture.`` prefix, drops ``vlm_backbone.*`` (OpenWAM stores the VLM as a directory), checks that the key set equals the source checkpoint and writes ``checkpoint_step_<N>.safetensors`` next to the config, tokenizer and normalization files copied (or symlinked with ``--link-assets``) from ``--source-checkpoint``. ``--verify`` reloads the result through ``openwam.deploy.load_from_checkpoint_dir``. The export directory works as a checkpoint for OpenWAM's own tooling and as ``rollout.model.model_path`` in the evaluation recipes below.

Evaluation
----------

The LIBERO evaluation recipes use the same checkpoint contract as this recipe. ``evaluations/libero/`` ships ``libero_{spatial,object,goal,10}_openwam_eval.yaml`` (see :doc:`../../evaluations/guides/libero`); each episode is logged as ``[libero eval] task_id=.., trial_id=.., success=..`` so success can be split per task. RoboTwin checkpoints use ``evaluations/robotwin/robotwin_<task>_openwam_eval.yaml`` for all 50 tasks (see :doc:`../../evaluations/guides/robotwin`).

Run the short smoke recipe (one environment, 30 steps, i.e. three 10-step generations) after setting ``MUJOCO_GL=egl`` and ``PYOPENGL_PLATFORM=egl``. The recipes place the env worker and rollout worker on separate GPUs so EGL rendering does not share a GPU with OpenWAM inference:

.. code-block:: bash

   python evaluations/eval_embodied_agent.py \
     --config-path ../tests/e2e_tests/evaluations --config-name libero_spatial_openwam_eval

Keep ``env.eval.max_steps_per_rollout_epoch`` divisible by ``rollout.model.num_action_chunks`` (10 with the default ``openwam.inference_horizon``) when you shorten a recipe.

The checked-in recipes target checkpoints trained with native delta EEF10 actions (``env.eval.openwam_action_representation: native_delta_eef10``). For a checkpoint trained on the canonical LIBERO bucket, which emits absolute EEF10 goals, set it to ``absolute_eef10``: RLinf then converts each goal against the current achieved pose before sending the 7-D OSC command. Keep this value aligned with the dataset metadata of the checkpoint under evaluation.


Alternate video encoders
------------------------

Some older OpenWAM checkpoints use legacy encoder names (``vjepa2_1``, ``flux_vae``, or ``wan_vae``); RLinf normalizes these names at deployment. If the encoder weights are stored outside the checkpoint, set ``rollout.model.encoder_model_path`` to the local encoder directory. RLinf stages a temporary config and leaves the checkpoint unchanged.
