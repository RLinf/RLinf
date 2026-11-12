Short Memory
============

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

The frames enter SigLIP together. At selected layers, attention first mixes
history at each patch position, then mixes patches within each frame. The
equations below describe one attention head, omitting the batch dimension.

Let :math:`T` be the number of frames, :math:`P` the patches per frame, and
:math:`d` the feature width of one head. Frames :math:`t=1,\ldots,T` run from
oldest to current, so :math:`t=T` is current; :math:`p=1,\ldots,P` indexes patch
positions. Each frame receives fixed relative-time encoding
:math:`e(\tau_t)=\mathrm{PE}(\tau_t)-\mathrm{PE}(0)`, where :math:`\tau_t` is its
time offset from current in seconds. PE is a fixed sinusoidal encoding;
subtracting PE(0) gives the current frame a zero time encoding. The layer's
LayerNorm and existing SigLIP projections then produce queries, keys
and values :math:`Q,K,V\in\mathbb{R}^{T\times P\times d}`.

Attention uses query/key similarity to form a weighted sum of values:

.. math::

   \operatorname{Attn}(Q,K,V;M)
   = \operatorname{softmax}_{\mathrm{row}}\!\left(
       \frac{QK^\top}{\sqrt{d}}+M
     \right)V.

Softmax normalizes the key scores in each query row. The factor
:math:`1/\sqrt{d}` scales similarity scores. The additive mask :math:`M` is zero
for allowed keys and :math:`-\infty` for excluded keys, giving them zero weight.

**Temporal mixing.** Hold patch position :math:`p` fixed and attend over frames:

.. math::

   \widetilde V_{:,p,:}
   = \operatorname{Attn}(Q_{:,p,:},K_{:,p,:},V_{:,p,:};M_{\mathrm{time}}).

A colon retains all entries along an axis, so these inputs are
:math:`T\times d` matrices. For a valid query frame :math:`t`,
:math:`M_{\mathrm{time}}[t,s]` is zero only when :math:`s\leq t` and frame
:math:`s` is valid; otherwise it is :math:`-\infty`. Each grid position thus
reads its current and past valid frames to produce history-aware values
:math:`\widetilde V`. Padded frame outputs are zeroed after each layer.

**Spatial mixing.** Hold frame :math:`t` fixed and attend over its patches:

.. math::

   O_{t,:,:}
   = \operatorname{Attn}(Q_{t,:,:},K_{t,:,:},\widetilde V_{t,:,:};0).

These inputs are :math:`P\times d` matrices, and every patch in the frame can
attend to every other patch. Q and K retain their original projections; only
V is replaced by :math:`\widetilde V`. The model does not derive a second set
of Q/K/V projections from the temporally mixed values. Head outputs O are
concatenated and passed through the existing output projection, residual
connections and MLP. Both attention stages reuse SigLIP parameters, and the
fixed time encoding adds no learned weights.

By default, temporal mixing runs at zero-based layers ``[3, 7, 11, 15]``.
After layer 15, the encoder discards historical frames and keeps the current
frame, which now contains information from earlier views. The remaining vision
layers and language model therefore process the usual number of visual tokens.
More history increases vision computation, rather than language-model context
length. SFT trains the existing network weights with the usual flow-matching loss.

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

   source .venv/bin/activate
   export PYTHONPATH="$PWD:${PYTHONPATH:-}"
   python toolkits/lerobot/calculate_norm_stats.py --config-name pi05_libero_mem --repo-id CollisionCode/libero_mem_lerobot_v2.1

This writes ``assets/pi05_libero_mem/CollisionCode/libero_mem_lerobot_v2.1/norm_stats.json``.
Set the checkpoint and that statistics file in the SFT launch:

.. code-block:: bash

   EMBODIED_PATH="$PWD/examples/sft" python examples/sft/train_vla_sft.py \
     --config-name libero_mem_sft_openpi_pi05 \
     actor.model.model_path=/path/to/RLinf-Pi05-LIBERO-SFT \
     actor.model.openpi_data.norm_stats_path="$PWD/assets/pi05_libero_mem/CollisionCode/libero_mem_lerobot_v2.1/norm_stats.json"

This runs full-parameter SFT with eight-frame history. Set the batch sizes and
update budget in ``examples/sft/config/libero_mem_sft_openpi_pi05.yaml`` for your
GPUs. For a current-frame control, retain ``data.short_memory`` and set
``actor.model.openpi.short_memory=False`` to use the same resumable loader.
See :doc:`Checkpoint Resume </rst_source/guides/resume>` to continue a saved run.

Evaluate the fine-tuned weights with the same statistics:

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_mem_openpi_pi05_eval \
     rollout.model.model_path=/path/to/short_memory_checkpoint \
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
