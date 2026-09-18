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

      2+ GPUs with FSDP

The OpenWAM checkpoint supplies the model and dataloader settings. Set ``data.train_data_paths`` to the dataset root (or a list of roots: each is read with the same dataloader settings and the windows are concatenated, so the mixture is sampled in proportion to size); the loader reads ``config.yaml`` from ``actor.model.model_path`` and keeps the native frame, action, and normalization conventions.

Installation
------------

Install the OpenWAM environment and RLinf:

.. code:: bash

   bash requirements/install.sh embodied --model openwam --env libero
   source .venv/bin/activate

Run It
------

Set the checkpoint and dataset paths in ``examples/sft/config/model/openwam.yaml`` and ``examples/sft/config/libero_sft_openwam.yaml``. The checked-in recipe maps the actor to GPUs ``0-1`` and uses ``use_orig_params: true`` because OpenWAM freezes part of its backbone while training the action modules.

Start the Ray-managed FSDP runner:

.. code:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_openwam

Override ``cluster.component_placement.actor`` and keep ``actor.global_batch_size`` divisible by the actor world size when you change the GPU count.

The preset loads the weights in fp32 (``precision: fp32``) so the optimizer keeps fp32 master weights while FSDP computes in bf16 (``mixed_precision.param_dtype``); bf16 master weights would round away almost every update at ``lr: 1e-6``. The policy is a single FSDP unit (OpenWAM's joint denoising driver reads block weights outside their forward), so the recipe uses FSDP2, which does not keep FSDP1's full-precision unsharded flat parameter. The model preset also keeps ``load_to_device: false``: every rank builds the policy on the CPU and FSDP moves its shard to the GPU while wrapping, so no rank ever holds the whole model on its device. The evaluation recipes load straight onto the GPU with ``load_to_device: true``.

Validation and resuming
-----------------------

Set ``data.val_data_paths`` (one dataset root or a list, read with the same dataloader settings) and ``runner.val_check_interval`` to report ``eval/loss``, ``eval/loss_video`` and ``eval/loss_action`` averaged over the validation loader. ``actor.eval_batch_size`` sets the per-rank validation batch and ``actor.eval_max_batches`` caps the number of validation batches per rank for large datasets. LeRobot-style readers select episodes by split: validation reads the ``val`` split by default, so set ``data.openwam_val_split`` to ``train`` when the validation root is a separate held-out dataset that only ships a train split (an empty validation set is rejected at start-up).

Checkpoints store the data loader, sampler (including the shuffle epoch) and RNG states next to the model weights, so ``runner.resume_dir=<log_path>/<experiment_name>/checkpoints/global_step_<N>`` continues with the next unseen batch instead of restarting the epoch.

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
