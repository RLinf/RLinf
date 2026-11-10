Evaluate and Train OpenPI with Short Memory
===========================================

Use this guide to evaluate a short-memory π₀.₅ checkpoint on LIBERO-Mem and
fine-tune one with FSDP1. Start with the history settings, supply your
checkpoint and normalization statistics, then launch evaluation or SFT.
These configurations use RLinf's native ``openpi`` model with LIBERO transforms.
Short memory supports ``openpi.task: eval`` and ``sft``; RL replay does not yet
carry camera history.

Choose the Visual History
-------------------------

Both configurations sample camera frames from the current episode, ordered from oldest
to current. ``num_frames`` includes the current frame and is configurable; eight
is the configuration default. ``stride`` counts environment control steps for evaluation
and dataset frames for SFT. In evaluation at 20 Hz, eight frames with stride 70
span 24.5 seconds.

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
dataset timestamps. The two strides describe the same physical interval only
when the dataset retains every control step at that rate. Conversion that drops
idle frames and renumbers timestamps compresses the training timeline; reading
those timestamps does not recover the removed time. Check the conversion before
equating training and evaluation history spans. The public dataset below declares
20 Hz, but its metadata alone does not establish that no control steps were
dropped. Missing frames at the episode start are zero-padded
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

For SFT, download the public
`RLinf-Pi05-LIBERO-SFT weights <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT>`__:

.. code-block:: bash

   hf download RLinf/RLinf-Pi05-LIBERO-SFT \
     --local-dir checkpoints/RLinf-Pi05-LIBERO-SFT

This supplies the initial weights for the configuration below. Compute LIBERO-Mem
normalization statistics from the training dataset as described below. The
reported paired experiment used a separate LIBERO-10 ``global_step_8000``
checkpoint; these public weights are a starting point for training, not an
assertion of identical experiment lineage.

For evaluation, supply your fine-tuned checkpoint and its OpenPI
``norm_stats.json``. All ``/path/to/...`` values are placeholders.
``model_path`` accepts a safetensors checkpoint directory or an RLinf checkpoint
with ``actor/model_state_dict/full_weights.pt`` (also its ``actor`` or
``model_state_dict`` directory). Direct safetensors and ``.pt`` weight paths are
accepted. Model shape and transforms come from the selected YAML configuration,
so keep the LIBERO transforms, action horizon, and normalization consistent with
the weights. Loading weights alone starts a new training run; complete training
state is restored through ``runner.resume_dir``.

Evaluate on LIBERO-Mem
----------------------

Once the simulator and weights are ready, run the dedicated evaluation configuration
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

History increases vision computation even though the language model receives
the same number of visual tokens. In a reviewer measurement on one RTX PRO 5000,
``predict_action_batch`` with eight frames took 174.0 ms versus 141.6 ms for the
current-frame model at batch size one, and 1641.7 ms versus 859.2 ms at batch size
20. Measure with your own batch size and hardware when choosing history length.

Prepare the Training Data
-------------------------

Download `CollisionCode/libero_mem_lerobot_v2.1
<https://huggingface.co/datasets/CollisionCode/libero_mem_lerobot_v2.1>`__ into
LeRobot's cache. Its metadata lists 961 episodes, 321,405 frames and 10 tasks
at 20 Hz, with all episodes in the training split:

.. code-block:: bash

   export HF_LEROBOT_HOME="$PWD/datasets"
   hf download CollisionCode/libero_mem_lerobot_v2.1 --repo-type dataset \
     --local-dir "$HF_LEROBOT_HOME/CollisionCode/libero_mem_lerobot_v2.1"

Keep ``HF_LEROBOT_HOME`` set for the following commands. The SFT configuration selects
``pi05_libero_mem``, which reads the published columns directly and maps them to
the same LIBERO policy inputs used at evaluation:

.. list-table::
   :header-rows: 1
   :widths: 45 25 30

   * - Dataset column
     - Policy input
     - Contents
   * - ``observation.images.image``
     - ``observation/image``
     - Third-person RGB video.
   * - ``observation.images.wrist_image``
     - ``observation/wrist_image``
     - Wrist RGB video.
   * - ``observation.state``
     - ``observation/state``
     - Eight-dimensional end-effector and gripper state.
   * - ``action``
     - ``actions``
     - Seven-dimensional delta pose and gripper action.

Task prompts come from LeRobot task metadata. The loader uses episode boundaries
and timestamps for history sampling. If you already converted the dataset to
``image``, ``wrist_image``, ``state`` and ``actions``, use ``pi05_libero`` instead
of ``pi05_libero_mem`` for both normalization and SFT. Keep that existing data
and configuration when reproducing an earlier run.

Compute normalization statistics with the same data configuration used for SFT:

.. code-block:: bash

   export PYTHONPATH="$PWD:${PYTHONPATH:-}"
   python toolkits/lerobot/calculate_norm_stats.py \
     --config-name pi05_libero_mem \
     --repo-id CollisionCode/libero_mem_lerobot_v2.1

This writes ``norm_stats.json`` under
``assets/pi05_libero_mem/CollisionCode/libero_mem_lerobot_v2.1/``. Pass this file
explicitly to training and evaluation. See
:doc:`OpenPI SFT </rst_source/examples/embodied/sft_openpi>` for the normalization
workflow with other datasets.

Fine-Tune with FSDP
-------------------

With the dataset and statistics ready, set ``data.train_data_paths`` to its local
root or LeRobot repository ID. Historical camera queries stay within each episode;
the current state, prompt and forward action window follow the usual OpenPI
transforms. Prompt tokenization runs once per sample.

For a current-frame training control, keep ``data.short_memory`` and set
``actor.model.openpi.short_memory=False``. This selects the same resumable loader
and sampler while returning only the current camera frames. Without
``data.short_memory``, a model with short memory disabled uses its existing loader.

Launch the SFT entry point directly so that Hydra overrides reach the configuration:

.. code-block:: bash

   source .venv-libero-mem/bin/activate
   export EMBODIED_PATH="$PWD/examples/sft"
   export PYTHONPATH="$PWD:${PYTHONPATH:-}"
   python examples/sft/train_vla_sft.py \
     --config-name libero_mem_sft_openpi_pi05 \
     data.train_data_paths=CollisionCode/libero_mem_lerobot_v2.1 \
     actor.model.model_path="$PWD/checkpoints/RLinf-Pi05-LIBERO-SFT" \
     actor.model.openpi_data.norm_stats_path="$PWD/assets/pi05_libero_mem/CollisionCode/libero_mem_lerobot_v2.1/norm_stats.json" \
     runner.logger.log_path="$PWD/logs/libero_mem_short_memory"

The configuration trains the complete model with the existing flow-matching loss,
12,750 updates, microbatch size four per rank and global batch size 128. Adjust
the batch sizes and update budget to your available GPUs. The history controls
are ``data.short_memory.num_frames`` and ``data.short_memory.stride``; optional
``+data.episodes='[0,1]'`` selects dataset episode IDs for a bounded run.

The model keeps FP32 parameters for optimizer updates and uses BF16 backbone
activations. The ``openpi_fsdp`` preset retains FP32 compute for action/time
projections, RMSNorm and vision embedding modules. Gradient checkpointing uses
the non-reentrant implementation. Short memory adds no separate optimizer or
trainable memory parameters.

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
