PhyAI Embodied Rollout Backend
==============================

This guide trains a PI0.5 policy on LIBERO with PPO while
`PhyAI <https://github.com/mingti-org/phyai>`_ generates the rollouts. PhyAI is
an inference framework for physical-AI models such as VLAs. Each RLinf rollout
worker runs one PhyAI engine on its GPU, and the engine samples actions with
CUDA graphs and FlashInfer kernels instead of the Hugging Face model code.
Everything else stays with RLinf: the LIBERO environments, observation and
action transforms, the FSDP actor, and weight synchronization.

You'll install the backend, point the example at a checkpoint, and run
training. Before the run, one section explains why PhyAI training keeps
importance-sampling correction turned on, because that setting decides how the
actor treats actions it did not sample itself.

Prerequisites
-------------

You need a Linux machine with NVIDIA GPUs and Python 3.12. The backend
currently supports PI0.5 on LIBERO with synchronous PPO, and each rollout
worker uses exactly one GPU. The training path has been smoke-tested on two
NVIDIA H800 80 GB GPUs with the RLinf PI0.5 LIBERO SFT checkpoint.

Installation
------------

From the RLinf repository root, install RLinf, OpenPI, LIBERO, and PhyAI into
one virtual environment, then activate it:

.. code-block:: bash

   bash requirements/install.sh \
     --python 3.12 \
     --torch 2.11.0 \
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

   # For mainland China users: export HF_ENDPOINT=https://hf-mirror.com
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
