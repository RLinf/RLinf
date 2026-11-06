Evaluate and Train OpenPI with Short Memory
===========================================

Use this guide to evaluate a short-memory π₀.₅ checkpoint on LIBERO-Mem and
fine-tune one with FSDP or FSDP2. Start with the history settings, supply your
checkpoint and normalization statistics, then launch evaluation or SFT.
These recipes use RLinf's native ``openpi`` model with ``pi05_libero`` transforms.
Short memory supports ``openpi.task: eval`` and ``sft``; RL replay does not yet
carry camera history.

Choose the Visual History
-------------------------

Both recipes sample camera frames from the current episode, ordered from oldest
to current. ``num_frames`` includes the current frame and is configurable; eight
is the recipe default. ``stride`` counts environment control steps for evaluation
and dataset frames for SFT. At 20 Hz, eight frames with stride 70 span 24.5 seconds.

Evaluation configures history under ``env.eval.short_memory``. SFT uses
``data.short_memory`` and reads the sampling rate from LeRobot metadata:

.. code-block:: yaml

   # Evaluation: examples/embodiment/config/env/libero_mem.yaml
   short_memory:
     num_frames: 8
     stride: 70
     fps: 20.0

   # SFT: examples/sft/config/libero_mem_sft_openpi_pi05.yaml
   data:
     short_memory:
       num_frames: 8
       stride: 70

Match evaluation's ``fps`` to the environment control rate. Evaluation derives
relative times from control-step differences; SFT uses the selected frames'
actual dataset timestamps. Missing frames at the episode start are zero-padded
and masked out, and histories are cleared when their environment resets.

Only camera inputs carry history. State and prompt describe the current frame;
training action targets start at that frame. The vision encoder mixes history
at zero-based layers ``[3, 7, 11, 15]`` and retains only current-frame tokens after
layer 15. These settings are configurable through
``openpi.short_memory_temporal_layers`` and
``openpi.short_memory_drop_history_layer``. They reuse the vision weights without
adding learned parameters or increasing the language model's visual token count.
With ``short_memory: false`` (the default), OpenPI uses its ordinary image path.
For evaluation without history, set both
``rollout.model.openpi.short_memory=False`` and ``env.eval.short_memory=null``.

Prepare the Environment and Weights
-----------------------------------

Use the OpenPI environment and LIBERO-Mem fork described in
:doc:`LIBERO </rst_source/examples/embodied/libero>` for simulation. SFT reads
LeRobot demonstrations and does not launch a simulator. Keep the dataset's
LeRobot version and video decoder dependencies installed in the training
environment. TorchCodec decoding requires a compatible PyTorch/TorchCodec pair
and FFmpeg shared libraries; image-backed and video-backed datasets use the same
short-memory loader.

The commands below use the dedicated ``.venv-libero-mem`` environment from that
installation.

Supply your own compatible π₀.₅ checkpoint and the OpenPI ``norm_stats.json``
used with its dataset. All ``/path/to/...`` values in the recipes are placeholders.
``model_path`` accepts a safetensors checkpoint directory or an RLinf checkpoint
with ``actor/model_state_dict/full_weights.pt`` (also its ``actor`` or
``model_state_dict`` directory). Direct safetensors and ``.pt`` weight paths are
accepted. Model shape and transforms come from the selected YAML configuration,
so keep ``pi05_libero``, the action horizon, and normalization consistent with
the weights. Loading weights alone starts a new training run; complete training
state is restored through ``runner.resume_dir``.

Evaluate on LIBERO-Mem
----------------------

Once the simulator and weights are ready, run the dedicated evaluation recipe
from the repository root in the activated environment:

.. code-block:: bash

   source .venv-libero-mem/bin/activate
   bash evaluations/run_eval.sh libero libero_mem_openpi_pi05_eval \
     rollout.model.model_path=/path/to/short_memory_checkpoint \
     rollout.model.openpi_data.norm_stats_path=/path/to/libero-mem/norm_stats.json

This selects ``env_type: libero`` with ``task_suite_name: libero_mem``, 20 parallel
environments and a 700-step horizon. The model predicts 10 actions, executes five
per call, and uses five denoising steps. History is recorded at every control
step, including steps within an action chunk. Keep
``env.eval.skip_intermediate_renders=False`` when using history.

To change history length or spacing, append, for example,
``env.eval.short_memory.num_frames=3 env.eval.short_memory.stride=35`` to the
command. Use the checkpoint's training history settings for comparable results.
See :doc:`LIBERO Evaluation </rst_source/evaluations/guides/libero>` for evaluation
outputs and the LIBERO page for LIBERO-Mem success semantics.

Fine-Tune with FSDP
-------------------

For training, prepare a map-style LeRobot LIBERO-Mem dataset with task prompts,
state, actions and both cameras. Set ``data.train_data_paths`` to its local root
or LeRobot repository ID. Historical camera queries stay within each episode;
the current state, prompt and forward action window follow the usual OpenPI
transforms. Prompt tokenization runs once per sample.

For a current-frame training control, keep ``data.short_memory`` and set
``actor.model.openpi.short_memory=False``. This selects the same resumable loader
and sampler while returning only the current camera frames. Without
``data.short_memory``, a model with short memory disabled uses its existing loader.

Launch the SFT entry point directly so that Hydra overrides reach the recipe:

.. code-block:: bash

   source .venv-libero-mem/bin/activate
   export EMBODIED_PATH="$PWD/examples/sft"
   export PYTHONPATH="$PWD:${PYTHONPATH:-}"
   python examples/sft/train_vla_sft.py \
     --config-name libero_mem_sft_openpi_pi05 \
     data.train_data_paths=/path/to/libero-mem-lerobot \
     actor.model.model_path=/path/to/libero10_pi05_sft_checkpoint \
     actor.model.openpi_data.norm_stats_path=/path/to/libero-mem-lerobot/norm_stats.json \
     runner.logger.log_path="$PWD/logs/libero_mem_short_memory"

The recipe trains the complete model with the existing flow-matching loss,
12,750 updates, microbatch size four per rank and global batch size 128. Adjust
the batch sizes and update budget to your available GPUs. The history controls
are ``data.short_memory.num_frames`` and ``data.short_memory.stride``; optional
``+data.episodes='[0,1]'`` selects dataset episode IDs for a bounded run.

The model keeps FP32 parameters for optimizer updates and uses BF16 backbone
activations. The ``openpi_fsdp`` preset retains FP32 compute for action/time
projections, RMSNorm and vision embedding modules. Gradient checkpointing uses
the non-reentrant implementation. Short memory adds no separate optimizer or
trainable memory parameters.

To use FSDP2, append ``actor.fsdp_config.strategy=fsdp2`` to the same command.
Keep ``actor.training_backend=fsdp``: that field selects the shared worker, while
``strategy`` selects its FSDP implementation. FSDP2 applies the same named
precision policies. Keep the original parameter dtype uniform within each
sharding boundary; configure compute dtypes through those policies rather than
casting individual parameters. FSDP2 requires
``keep_low_precision_grads: false``, the preset default.

Resume Training
---------------

The short-memory loader uses ``StatefulDataLoader`` with a distributed sampler.
SFT checkpoints save each rank's loader state and sampler epoch in ``actor/data.pt``
and RNG state in ``actor/rng.pt`` alongside the training checkpoint. Keep these files when
continuing a run.

Append ``+runner.resume_dir=/path/to/checkpoints/global_step_2125`` to the SFT
command to restore a complete checkpoint. Keep the dataset, history settings,
worker count, world size and training configuration consistent with the saved
run. See :doc:`Checkpoint Resume <../resume>` for checkpoint layout and general
resume instructions.
