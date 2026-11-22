OpenPI Visual Short Memory
==========================

Give OpenPI π₀.₅ access to recent camera frames during SFT and evaluation.
Short Memory mixes those frames inside the vision encoder, then passes only
the current frame's enriched features to the language model.

How History Enters the Policy
----------------------------------------

A single image may show a bowl on a plate without revealing whether the robot
has already lifted it. Short Memory lets the policy read the preceding views
when choosing its next action. Each camera supplies frames from the same
episode, ordered from oldest to current; state, prompt and action targets still
belong to the current step.

The frames enter SigLIP together, each tagged with a fixed relative sinusoidal time encoding (zero for the current frame, offset in seconds for past frames) before LayerNorm and the existing SigLIP Q/K/V projections. At selected layers, each attention head runs two stages reusing the same SigLIP weights and adding no new learnable parameters:

1. **Temporal mixing:** At each patch position, causal masked attention mixes the current frame with earlier valid frames (zeroing padded frames) to produce history-aware values.
2. **Spatial mixing:** Within each frame, full patch-to-patch attention uses the original queries and keys with those history-aware values, then passes through the standard output projection, residual connection, and MLP.

By default, temporal mixing runs at zero-based layers ``[3, 7, 11, 15]``. After layer 15, the encoder discards historical frames and keeps only the enriched current frame, so the language model still sees the usual number of visual tokens. SFT trains the existing network weights with the standard flow-matching loss.

Choose the History Window
-------------------------

The supplied LIBERO-Mem configurations use eight frames, including current,
spaced 70 steps apart. At 20 Hz this spans 24.5 seconds. Evaluation records every
control step, including steps inside an action chunk; SFT samples within each
demonstration episode. Missing frames at the start are zero-padded and masked,
and an environment reset clears that environment's history.

Set ``num_frames`` and ``stride`` under ``data.short_memory`` for SFT or
``env.eval.short_memory`` for evaluation. Evaluation also uses ``fps`` to convert
step offsets to seconds; SFT reads dataset timestamps. Match physical time spans
when comparing them, especially if dataset conversion removed frames.
The model switch is ``actor.model.openpi.short_memory`` for SFT and
``rollout.model.openpi.short_memory`` for evaluation; both supplied configurations
enable it. Temporal layers and the point at which history is discarded are set
by ``openpi.short_memory_temporal_layers`` and
``openpi.short_memory_drop_history_layer``.

Train and Evaluate on LIBERO-Mem
----------------------------------------

Prepare the OpenPI environment and weights using :doc:`OpenPI SFT <../sft_openpi>`
and the simulator using :doc:`LIBERO-Mem <../libero>`. The commands below use the
installer's default ``.venv``. Download the
`LIBERO-Mem demonstrations <https://huggingface.co/datasets/CollisionCode/libero_mem_lerobot_v2.1>`__
and `initial weights <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT>`__
as described in the SFT guide, then compute task-specific normalization:

.. code-block:: bash

   python toolkits/lerobot/calculate_norm_stats.py --config-name pi05_libero_mem --repo-id CollisionCode/libero_mem_lerobot_v2.1

This writes ``assets/pi05_libero_mem/CollisionCode/libero_mem_lerobot_v2.1/norm_stats.json``.
Set the checkpoint and that statistics file in the SFT launch:

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh libero_mem_sft_openpi_pi05 \
     actor.model.model_path=./RLinf-Pi05-LIBERO-SFT \
     actor.model.openpi_data.norm_stats_path="$PWD/assets/pi05_libero_mem/CollisionCode/libero_mem_lerobot_v2.1/norm_stats.json"

This runs full-parameter SFT with eight-frame history. Set the batch sizes and
update budget in ``examples/sft/config/libero_mem_sft_openpi_pi05.yaml`` for your
GPUs. For a current-frame control, retain ``data.short_memory`` and set
``actor.model.openpi.short_memory=False`` to use the same resumable loader.
See :doc:`Checkpoint Resume </rst_source/guides/resume>` to continue a saved run.

Evaluate the fine-tuned weights with the same statistics:

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_mem_openpi_pi05_eval \
     rollout.model.model_path=./logs/checkpoints/global_step_1000/actor \
     rollout.model.openpi_data.norm_stats_path="$PWD/assets/pi05_libero_mem/CollisionCode/libero_mem_lerobot_v2.1/norm_stats.json"

The configuration evaluates 20 environments for 700 control steps. Keep
``env.eval.skip_intermediate_renders=False`` to record history within each
action chunk. For current-frame evaluation, set
``rollout.model.openpi.short_memory=False`` and ``env.eval.short_memory=null``.
Results and success semantics are described in
:doc:`LIBERO Evaluation </rst_source/evaluations/guides/libero>`.

.. note::

   Short Memory supports ``openpi.task=eval`` and ``sft``. RL training is not
   supported because its replay inputs do not carry camera history.
