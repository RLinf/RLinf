LLM On-Policy Distillation
==========================

Train a Student language model on its own math-reasoning responses using scores
from a frozen Teacher. This recipe starts with the synchronous FSDP example,
then shows how to choose compatible checkpoints, reserve Teacher GPUs, and
select either a distribution loss or a sampled-token objective.

Overview
--------

The Student generates each response. The Teacher scores those same contexts,
and the Actor updates only the Student. You can train from Teacher scores alone
or combine distillation with the task's reinforcement-learning objective.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Qwen2.5 Student and a tokenizer-compatible Teacher

   .. grid-item-card:: Algorithm
      :text-align: center

      On-policy distillation, optionally combined with task rewards

   .. grid-item-card:: Data
      :text-align: center

      AReaL-boba math prompts

   .. grid-item-card:: Hardware
      :text-align: center

      Six GPUs in the example: four Student GPUs and two Teacher GPUs

.. list-table:: Tasks
   :header-rows: 1
   :widths: 25 40 35

   * - Task
     - Config
     - Training
   * - Math reasoning
     - ``qwen2.5-1.5b-opd-fsdp``
     - FSDP Actor, vLLM rollout and Teacher; synchronous updates

How Distillation Works
----------------------

With ``runner.execution_mode: sync``, each round generates responses with the
current Student, obtains Teacher scores, and finishes the Student update
before the next rollout. Teachers keep their checkpoints throughout training.
When several Teachers are configured, the sample's routing key selects one.

For distribution objectives, ``topk_source: teacher`` selects the Teacher's most
likely tokens. With ``topk_source: student``, rollout records the Student's
top-k IDs at each generated position and asks the Teacher to score those tokens.
The Actor reuses its normal training forward to compute current probabilities
on that fixed support. Both paths preserve the original context at each response
position.

Teacher and Student probabilities are normalized over each model's complete
output vocabulary, using its configured vocabulary size even when the tokenizer
has fewer IDs. Top-k selects entries from that distribution and preserves their
probability mass. Tensor-parallel alignment padding beyond the model vocabulary
is excluded.

Installation
------------

Install the agentic environment for the inference engine you will use. The
example uses vLLM for both rollout and Teacher scoring:

.. code-block:: bash

   bash requirements/install.sh agentic --engine vllm --venv .venv-opd
   source .venv-opd/bin/activate

This creates and activates the training environment. To use SGLang, install
with ``--engine sglang``, set ``rollout.rollout_backend: sglang``, and set
``distillation.teacher_models.default.inference.backend: sglang``. See
:doc:`Installation <../../../start/installation>` for platform prerequisites.

For Student top-k or MC-K with a SGLang Teacher, use SGLang ``0.5.12.post1`` with the
per-position scoring patch. ``requirements/install.sh`` applies it automatically
for that version. To update an existing environment, activate it and run:

.. code-block:: bash

   bash requirements/patches/apply_sglang.sh

The patch lets the Teacher score the complete sequence in a prefill-only
request and gather different candidate IDs at each position, including across
prefill and logits chunks. It preserves full model-vocabulary probabilities.
Other SGLang versions require a matching patch before Student top-k scoring.
Teacher top-k uses the engine's native prompt scoring in one request. The vLLM
Teacher retains its per-prefix compatibility path for Student-selected support;
both rollout backends record Student top-k IDs during generation.

Download the Model
------------------

Prepare local Student and Teacher checkpoints and the AReaL-boba JSONL data
described in :doc:`Math Reasoning <reasoning>`. The Student checkpoint must be
supported by the Actor and rollout backends; the Teacher must be supported by
its chosen inference engine. Use the Student tokenizer for both Actor and
rollout.

Teacher and Student must have identical token-to-ID mappings, added tokens,
and special-token settings. A shared model family or vocabulary size alone
does not establish compatibility. Initialization checks tokenizer fingerprints
and rejects a mismatch. Each model must cover all real tokenizer IDs, and the
Teacher context limit must exceed the longest prompt plus response by at least
one token.

Allocate Teacher Devices
------------------------

Reserve explicit devices for the frozen Teacher alongside the Student
allocation. The supplied config uses GPU ranks 0–3 for Actor, rollout, and
reward, and ranks 4–5 for one tensor-parallel Teacher replica:

.. code-block:: yaml

   cluster:
     component_placement:
       actor,rollout,reward: 0-3
       teacher_default: 4-5
   distillation:
     enabled: true
     teacher_models:
       default:
         model_path: /path/to/teacher
         num_replicas: 1
         inference:
           backend: vllm
           tensor_parallel_size: 2
           pipeline_parallel_size: 1
           max_model_len: 32768

The placement key is ``teacher_<model key>``. Allocate
``num_replicas × tensor_parallel_size × pipeline_parallel_size`` GPUs, with
each replica on one node. Teacher devices must be disjoint from every Student
component; use explicit ranges in place of ``all``. Keep
``cluster.auto_scheduler: false`` if you configure that option. See
:doc:`Component placement <../../../concepts/placement>` for placement
syntax.

Choose the Objective
--------------------

Start with the example's ``forward_kl_topk`` objective and ``topk: 64``. The
following modes are selected under ``distillation.distillation_loss``:

.. list-table:: Supported Objectives
   :header-rows: 1
   :widths: 35 65

   * - ``loss_mode``
     - Requirements
   * - ``forward_kl_topk``, ``forward_kl_topk_tail``
     - Positive ``topk`` with Teacher or Student support; ``topk: -1`` uses the full vocabulary.
   * - ``reverse_kl_topk``, ``reverse_kl_topk_tail``
     - Positive ``topk`` requires ``topk_source: student``; ``topk: -1`` uses the full vocabulary.
   * - ``k1`` (``kl``)
     - Sampled-token objective; requires ``use_policy_gradient: true``.
   * - ``k2`` (``mse``), ``k3`` (``low_var_kl``), ``abs``
     - Sampled-token objectives; support direct or policy-gradient training.

Tail modes account for probability mass outside the selected support. Full
vocabulary targets can require substantial host memory and transfer bandwidth;
``inference.max_payload_bytes`` bounds each Teacher request. Reverse, tail, and
full-vocabulary modes require ``log_prob_min_clamp: null``. All modes currently
use ``policy_loss_mode: vanilla`` and text-only inputs.

Set ``use_policy_gradient: true`` to use the distillation signal as a
policy-gradient objective. This requires sampling with ``do_sample: true``,
``temperature: 1.0``, ``top_p: 1.0``, ``top_k: -1`` (or ``0``), and
``repetition_penalty: 1.0``. For SGLang rollout, also check any overrides in
``rollout.sampling_params``. The Teacher scores raw model probabilities.

Policy-gradient training also requires old Student logprobs. For synchronous single-sample training, set either
``rollout.return_logprobs: true`` to collect them during generation or
``algorithm.recompute_logprobs: true`` to recompute them before the Actor update.
The supplied config disables both for direct distillation.

For joint training, set ``use_task_rewards: true`` and configure the task's
reward, advantage, and PPO settings as in :doc:`reasoning`. The objective is
the task loss plus ``distillation_loss_coef`` times the distillation loss;
pure distillation uses coefficient one. Task and Teacher masks have independent
normalization. Entropy and reference-KL regularizers retain task token weighting
when ``algorithm.use_valid_token_scale: true``. If rejection sampling removes
every sequence on any Actor rank, all Actor ranks skip that round's update and
report ``rollout/skipped_empty_batch``.

Sample Multiple Tokens per Position
-----------------------------------

For sampled-token policy gradients, set ``distillation.distillation_loss.mc_k``
to the number of draws per generated position. Its default is 1. With
``mc_k: 4``, SGLang records the emitted token and three additional draws with
replacement from the same Student distribution. Only the emitted token extends
the trajectory. Keep repeated IDs, since each occurrence counts as a draw.

Rollout records a behavior logprob for each draw. At the original prefix the
Teacher scores all four IDs, and the Actor computes the clipped policy-gradient
loss for each ID using its current probability divided by its recorded behavior
probability. For ``loss_mode: k1``, the detached advantage is the Teacher
logprob minus the current Student logprob. The Actor averages the four losses
before reducing over trainable tokens.
Tool output and padding do not contribute. The sync example is
``qwen3.5-2b-opd-mc4-pg-fsdp2-sglang``.

Run with Separate Actor and Rollout GPUs
----------------------------------------

Use ``qwen3.5-2b-async-opd-mc4-pg-fsdp2-sglang`` for continuous generation.
It sets ``runner.execution_mode: separate_async`` and places the FSDP Actor
on GPU 0, the SGLang rollout server on GPU 1, and the frozen Teacher on GPUs
2 and 3. Use either MC-K or the single sampled token for policy gradients.
Finite Student top-k requires synchronous execution.

A prompt group enters the training queue after its retained responses have
Teacher scores. The Actor takes complete groups in completion order. While it
updates one batch, rollout and scoring can work on subsequent groups. Set
``rollout.return_logprobs: true`` and ``algorithm.recompute_logprobs: false``.
A long response may span multiple weight updates. Each draw keeps the logprob
from the Student that generated it, including tokens from earlier versions.

``algorithm.staleness_threshold`` sets the maximum number of outstanding groups
to ``(staleness_threshold + 1) * data.rollout_batch_size``. This includes groups
generating, scoring, waiting in the queue, or training until the updated
weights are published. A threshold of 1 allows two batches in total. To drop
groups by version age as well, set ``agentloop.max_actor_accept_staleness``.
By default, older completed groups remain eligible for training.

Run It
------

Run from the repository root after replacing the model and dataset paths below.
The YAML already selects pure distillation and the six-GPU placement above:

.. code-block:: bash

   PYTHONPATH="$PWD:$PYTHONPATH" python examples/reasoning/main_grpo.py \
     --config-path config/math --config-name qwen2.5-1.5b-opd-fsdp \
     actor.model.model_path=/path/to/student \
     actor.tokenizer.tokenizer_model=/path/to/student \
     rollout.model.model_path=/path/to/student \
     distillation.teacher_models.default.model_path=/path/to/teacher \
     'data.train_data_paths=[/path/to/AReaL-boba-106k.jsonl]' \
     'data.val_data_paths=[/path/to/AReaL-boba-106k.jsonl]'

Use either MC-K config name above with the same entry point and checkpoint
overrides.

This launches the frozen Teacher service, generates Student responses, and
updates the FSDP Actor. Change the YAML to adjust sequence lengths, batch sizes,
or placement before launching. For a Megatron Actor, start from the corresponding
GRPO config and carry over the ``distillation`` section and disjoint Teacher
placement.

Visualization and Results
-------------------------

Watch ``actor/training/distillation/loss`` and
``actor/training/distillation/token_count`` in the configured TensorBoard log
directory for synchronous runs. Separate async runs use
``train/distillation/loss`` and ``train/distillation/token_count``. The first
measures the objective and the second counts contributing response tokens. Compare validation task rewards against
the Student checkpoint under the same evaluation settings; this example does
not prescribe an expected improvement. See :doc:`Training metrics
<../../../reference/metrics>` for the shared rollout and training metrics.
