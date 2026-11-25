PhyAI Rollout Backend
=====================

Use `PhyAI <https://github.com/mingti-org/phyai>`_ to train a PI0.5 policy
in process or evaluate it through the PhyAI Gateway. PhyAI is an inference
framework for physical-AI models such as VLAs. The in-process path puts one
engine in each RLinf rollout worker and keeps the LIBERO environments, FSDP
actor, observation transforms, and weight synchronization in RLinf.

The sections below install and run the in-process PPO backend. To evaluate
through a Gateway, follow :ref:`phyai-http-evaluation-en`. That path runs RLinf
environment workers and an HTTP rollout worker, while the Gateway and model
server run as separate processes.

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

Evaluate Through the PhyAI Gateway
----------------------------------

Use this evaluation-only path to send batched observations through the Gateway.
Without ``rollout.phyai.gateway_url``, RLinf starts the model server and Gateway
on a placement-selected worker and manages their lifetime. Set a Gateway URL to
use externally managed services instead.

The model server loads the checkpoint and handles preprocessing and action
postprocessing; RLinf runs LIBERO and sends observations over HTTP.

.. code-block:: text

   RLinf evaluation -> HTTP MessagePack -> PhyAI Gateway -> model server

Use the Evaluation Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the environment from :ref:`phyai-installation-en`; it includes the PhyAI
model server and Gateway. The model server can run in another environment or on
another host with compatible PhyAI, Gateway, checkpoint, and tokenizer versions.

Download the Checkpoint and Tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Gateway path uses the `LeRobot PI0.5 LIBERO checkpoint
<https://huggingface.co/lerobot/pi05_libero_finetuned_v044>`_ and the PaliGemma
tokenizer. Keep the
checkpoint's preprocessing, normalization, and action postprocessing files.
The RLinf/OpenPI checkpoint used for in-process training has a different layout
and cannot be substituted directly.

.. warning::

   The `PaliGemma repository
   <https://huggingface.co/google/paligemma-3b-pt-224>`_ is gated. Accept its
   license on Hugging Face, then authenticate with ``hf auth login`` or provide
   ``HF_TOKEN`` before downloading. A mirror does not bypass these requirements.

Download both snapshots at the pinned revisions below. Replace ``/path/to``
with your asset directory; existing local copies of these snapshots can also be
used:

.. code-block:: bash

   hf auth login
   # Optional mirror: export HF_ENDPOINT=https://hf-mirror.com
   hf download lerobot/pi05_libero_finetuned_v044 \
     --revision 7152f13e34c6c115d808aea0e5d08aca74eac14b \
     --local-dir /path/to/pi05_libero_finetuned_v044
   hf download google/paligemma-3b-pt-224 \
     --revision 35e4f46485b4d07967e7e9935bc3786aad50687c \
     --local-dir /path/to/paligemma-3b-pt-224

Configure the Gateway Services
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For automatic launch, set these paths in
``evaluations/libero/libero_spatial_pi05_eval_phyai.yaml``. The asset directories
must be accessible on the worker selected by placement:

.. code-block:: yaml

   rollout:
     phyai:
       server_config: tests/e2e_tests/evaluations/phyai_gateway_server.yaml
       checkpoint_dir: /path/to/pi05_libero_finetuned_v044
       tokenizer_dir: /path/to/paligemma-3b-pt-224

RLinf writes a temporary PhyAI server configuration, starts both services, and
waits for a healthy backend. With ``gateway_url`` omitted, let RLinf manage the
services instead of starting duplicate processes manually.

For an externally managed Gateway, set ``rollout.phyai.gateway_url`` to its HTTP
URL. Configure the asset paths on the external model server; RLinf does not use
``server_config``, ``checkpoint_dir``, or ``tokenizer_dir`` in this mode. See the
`PhyAI server and Gateway documentation
<https://phyai.mintlify.app/deployment/server-gateway>`_ for service installation
and startup.

Use the same model name, for example ``pi05``, for:

* the model server's ``server.model_name``;
* the model name passed to the Gateway's ``--backend`` option; and
* RLinf's ``rollout.phyai.model_name``.

Run the Evaluation
~~~~~~~~~~~~~~~~~~

With the paths or external Gateway URL configured, run the standard evaluation
entry point from the RLinf repository root:

.. code-block:: bash

   source .venv/bin/activate
   bash evaluations/run_eval.sh libero libero_spatial_pi05_eval_phyai

The configuration selects ``serving_mode: gateway`` and requests batched
16-step action chunks. ``runner.only_eval`` must remain true. Leave
``CUDA_VISIBLE_DEVICES`` unset so RLinf placement can select the requested GPU;
see :doc:`the placement guide <../../evaluations/reference/configuration>` for
multi-GPU placement.

Inspect ``success_once``, ``success_at_end``, and ``num_trajectories`` in the
final evaluation metrics. Automatically launched services are stopped by the
evaluation driver; stop externally managed services after the run.
