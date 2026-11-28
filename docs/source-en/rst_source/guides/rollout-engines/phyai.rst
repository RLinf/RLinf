PhyAI Rollout Backend
=====================

`PhyAI <https://github.com/mingti-org/phyai>`_ is an inference framework for
embodied models such as VLAs. RLinf integrates PhyAI in two ways:

- **PPO training**: runs a PhyAI engine inside each rollout worker for action
  sampling, while RLinf manages the LIBERO environments, FSDP actor,
  observation transforms, and weight synchronization.
- **Standalone HTTP evaluation**: serves the PI0.5 model as a separate process
  behind the PhyAI Gateway, while RLinf runs LIBERO and queries it over HTTP
  (see :ref:`phyai-http-evaluation-en`).

Prerequisites
-------------

You need a Linux machine with NVIDIA GPUs and Python 3.12. The backend
supports PI0.5 on LIBERO with synchronous PPO, and each rollout
worker uses exactly one GPU. The training path is verified on two
NVIDIA H800 80 GB GPUs with the RLinf PI0.5 LIBERO SFT checkpoint.

.. _phyai-installation-en:

Installation
------------

From the RLinf repository root, install RLinf, OpenPI, LIBERO, and PhyAI into
one virtual environment, then activate it:

.. code-block:: bash

   bash requirements/install.sh \
     --python 3.12 \
     --install-rlinf \
     embodied \
     --model openpi \
     --env libero \
     --engine phyai
   source .venv/bin/activate

What this does:

1. Installs RLinf, OpenPI, and LIBERO exactly as for the Hugging Face rollout.
2. Installs the PhyAI packages at the version RLinf is tested with, for the
   CUDA version of the installed Torch. PhyAI's own dependency list would
   replace the Torch and Transformers versions OpenPI needs, so the installer
   keeps those versions and stops if PhyAI's CUDA build does not match Torch.

Prepare the Checkpoint
----------------------

Download the PI0.5 LIBERO SFT checkpoint:

.. code-block:: bash

   hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir RLinf-Pi05-LIBERO-SFT

Then set ``actor.model.model_path`` in
``examples/embodiment/config/libero_spatial_ppo_openpi_pi05_phyai.yaml`` to the
downloaded directory. The rollout reads the same path, so PhyAI and the actor
load the same weights and normalize observations with the same statistics.
:doc:`RL on π₀ and π₀.₅ Models <../../examples/embodied/pi0>` lists the other
PI0.5 checkpoints and download mirrors.

To start from an RL training checkpoint instead, also set ``runner.ckpt_path``
to its ``.pt`` file. Each PhyAI engine loads that state dict on top of the
downloaded checkpoint, so the first rollout uses the same weights as the actor,
frozen parameters included.

Configure the Backend
---------------------

The example config already selects PhyAI. These are the lines that matter:

.. code-block:: yaml

   defaults:
     - hybrid_engines/phyai@rollout.phyai

   algorithm:
     recompute_logprobs: True
     importance_sampling_fix: True
     importance_sampling_clip: 1.0

   rollout:
     rollout_backend: phyai
     phyai:
       max_batch_size: null

``rollout.rollout_backend: phyai`` makes the training script start PhyAI
rollout workers instead of the Hugging Face ones. The ``hybrid_engines/phyai``
default supplies the kernel policy, which fixes the attention and activation
kernels PhyAI uses for PI0.5 so the numbers stay as close to the actor as the
two implementations allow. ``max_batch_size`` caps how many observations one
engine runs at once; ``null`` sizes it to the environments assigned to that
worker.

The three ``algorithm`` keys deal with the remaining difference between PhyAI
and the actor, which the next section explains.

Correct Rollout and Actor Differences
-------------------------------------

PhyAI and the FSDP actor hold the same weights, but they compute with different
kernels: paged attention, fused activations, and a different reduction order.
So when the actor scores an action that PhyAI sampled, it gets a slightly
different log-probability than PhyAI reported.

PPO measures each update against the policy that produced the data. If that
reference came from PhyAI, the kernel gap would show up in every PPO ratio and
could trigger clipping on its own. With
``recompute_logprobs``, the actor rescores every sampled action before training
and uses its own score as the reference instead. ``importance_sampling_fix``
then reweights each sample by how much more likely the actor finds the action
than PhyAI did, capped at ``importance_sampling_clip``, so the update stays
correct for data the actor did not sample itself. Keep the three keys together:
startup stops if ``importance_sampling_fix`` is on without
``recompute_logprobs``.

:doc:`PPO <../../reference/algorithms/ppo>` describes the correction and its
metrics in detail.

Run Training
------------

With the checkpoint path set, launch the example from the repository root:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh \
     libero_spatial_ppo_openpi_pi05_phyai

What this does: starts the embodied training entry point with an FSDP actor,
PhyAI rollout workers, and LIBERO environment workers. Each step collects
rollouts with PhyAI, has the actor rescore them, updates the actor, and sends
the new weights to every PhyAI engine before the next rollout.

Check the Run
-------------

The launcher writes logs under ``logs/<timestamp>-<config-name>/``. Watch
``env/success_once`` for task success. ``actor/importance_sampling_weight``
near 1 with a small ``actor/importance_sampling_clip_fraction`` means PhyAI and
the actor still agree closely; a weight drifting away from 1 means their scores
are diverging. :doc:`Training Metrics <../../reference/metrics>` defines the
remaining metrics.

.. _phyai-http-evaluation-en:

Evaluate Through the PhyAI HTTP Service
---------------------------------------

In this evaluation-only mode, the PI0.5 model runs as a separate inference
service decoupled from RLinf's LIBERO environments. RLinf runs the environments
and sends batched observations over HTTP to the PhyAI Gateway, while the model
server handles weight loading, observation preprocessing, and action
postprocessing.

.. code-block:: text

   RLinf evaluation -> HTTP MessagePack -> PhyAI Gateway -> model server

By default, RLinf starts a local model server and Gateway and stops them when
evaluation ends. To connect to an already deployed external service instead,
set ``rollout.phyai.gateway_url``.

Requirements and Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You need at least one NVIDIA GPU and the virtual environment from
:ref:`phyai-installation-en`, which already includes the model server and
Gateway.

By default, evaluation uses all available GPUs and starts one model server and
Gateway on the first GPU. To dedicate one or more GPUs to serving, assign
``phyai_server`` under ``cluster.component_placement`` (one service pair per
assigned GPU, called round-robin by rollout workers); see
:doc:`Placement <../../concepts/placement>`.

Download the Checkpoint and Tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HTTP evaluation requires the `LeRobot PI0.5 LIBERO checkpoint
<https://huggingface.co/lerobot/pi05_libero_finetuned_v044>`_ and the
PaliGemma tokenizer. This format differs from the RLinf/OpenPI checkpoint used
for PPO training above and cannot be mixed.

.. warning::

   The `PaliGemma repository
   <https://huggingface.co/google/paligemma-3b-pt-224>`_ is gated. Accept its
   license on Hugging Face and authenticate with ``hf auth login`` or
   ``HF_TOKEN`` before downloading. Mirrors do not bypass authentication.

Download both snapshots at the verified revisions below:

.. code-block:: bash

   hf download lerobot/pi05_libero_finetuned_v044 \
     --revision 7152f13e34c6c115d808aea0e5d08aca74eac14b \
     --local-dir ./pi05_libero_finetuned_v044
   hf download google/paligemma-3b-pt-224 \
     --revision 35e4f46485b4d07967e7e9935bc3786aad50687c \
     --local-dir ./paligemma-3b-pt-224

Run Automatic Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~

Activate the PhyAI environment and launch LIBERO-Spatial evaluation from the
RLinf repository root, passing your checkpoint and tokenizer directories:

.. code-block:: bash

   source .venv/bin/activate
   bash evaluations/run_eval.sh libero libero_spatial_pi05_eval_phyai \
     rollout.phyai.checkpoint_dir=./pi05_libero_finetuned_v044 \
     rollout.phyai.tokenizer_dir=./paligemma-3b-pt-224

The script starts a local model server and Gateway, evaluates 128 parallel
LIBERO-Spatial environments for 1 epoch (16-step action chunks), and stops both
services on exit. Requests are split by ``rollout.phyai.max_batch_size: 8`` so
1, 2, 4, or 8 GPUs work out of the box.

To avoid passing the paths on every run, edit ``checkpoint_dir`` and
``tokenizer_dir`` in
``evaluations/libero/libero_spatial_pi05_eval_phyai.yaml``.

Connect to an External Service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the model server and Gateway already run in another environment or on
another host, pass ``rollout.phyai.gateway_url``:

.. code-block:: bash

   source .venv/bin/activate
   bash evaluations/run_eval.sh libero libero_spatial_pi05_eval_phyai \
     rollout.phyai.gateway_url=http://127.0.0.1:30000

Configure the checkpoint and tokenizer on the external model server directly.
Keep ``server.model_name``, the Gateway's ``--backend`` flag, and RLinf's
``rollout.phyai.model_name`` identical (all default to ``pi05``). See the
`PhyAI server and Gateway documentation
<https://phyai.mintlify.app/deployment/server-gateway>`_ for standalone
deployment steps.

Check the Results
~~~~~~~~~~~~~~~~~

When evaluation finishes, the script prints the final metrics table—including
``eval/success_once``, ``eval/success_at_end``, and ``eval/num_trajectories``—
and writes logs and TensorBoard files under
``logs/<timestamp>-libero_spatial_pi05_eval_phyai/``. Auto-launched model
server and Gateway logs go to its ``phyai/`` subdirectory; see
:doc:`LIBERO Evaluation <../../evaluations/guides/libero>` for the benchmark
protocol.
