PhyAI Embodied Rollout Backend
==============================

Use PhyAI to generate PI0.5 rollouts for LIBERO while RLinf's FSDP actor trains
the policy. This guide covers installation, checkpoint configuration, and the
synchronous PPO example. RLinf handles environments, observation transforms,
action execution, and weight synchronization; each rollout worker runs an
in-process PhyAI engine on one GPU.

Prerequisites
-------------

Prepare a Linux machine with NVIDIA GPUs and a PI0.5 LIBERO checkpoint. The
installer uses Python 3.12 and Torch 2.11.0. The training path has been smoke-tested on
two NVIDIA A800 80 GB GPUs with the RLinf PI0.5 LIBERO SFT checkpoint.

Installation
------------

From the RLinf repository root, install RLinf, OpenPI, LIBERO, and PhyAI into
the same virtual environment, then activate it:

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

The installer detects the NVIDIA driver, selects a compatible Torch CUDA build,
and retains the Torch and Transformers versions installed for OpenPI. It then
selects ``requirements/agentic/phyai_0.1.1_cu12.txt`` or
``requirements/agentic/phyai_0.1.1_cu13.txt`` for the same CUDA line and installs
the three PhyAI packages with ``--no-deps`` from the official PhyAI wheel index.
It stops before installing PhyAI when the selected CUDA line and
``torch.version.cuda`` have different major versions.

Configure the Checkpoint and Backend
------------------------------------

Set ``actor.model.model_path`` in
``examples/embodiment/config/libero_spatial_ppo_openpi_pi05_phyai.yaml`` to your
PI0.5 LIBERO checkpoint directory. The checkpoint must contain
``model.safetensors`` and
``physical-intelligence/libero/norm_stats.json``. The rollout model inherits
the actor's checkpoint path and normalization statistics so both use the same
observation and action transforms.

To initialize from an RL training checkpoint, set ``runner.ckpt_path`` to its
``.pt`` file. Each PhyAI rollout engine loads the complete state dict after
loading ``model.safetensors`` so evaluation and the initial training rollout use
the same weights as the actor, including frozen parameters.

The example selects PhyAI through ``rollout.rollout_backend`` and loads the
kernel policy through a Hydra default:

.. code-block:: yaml

   defaults:
     - hybrid_engines/phyai@rollout.phyai

   rollout:
     rollout_backend: phyai

``examples/embodiment/config/hybrid_engines/phyai.yaml`` supplies
``rollout.phyai.kernel_config`` as a YAML mapping. It selects the attention and
activation kernels used by the PI0.5 adapter. Adjust
``rollout.phyai.max_batch_size`` to set each engine's batch capacity; a
``null`` value derives it from the number of environments assigned to the worker.

Run Training
------------

After setting the checkpoint path, launch the example from the repository root:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh \
     libero_spatial_ppo_openpi_pi05_phyai

This starts the standard embodied training entry point with an FSDP actor,
PhyAI rollout workers, and LIBERO environment workers. RLinf samples action
chunks, updates the actor, and sends updated weights back to the PhyAI engines.

The example uses bucket weight synchronization and enables actor log-probability
recomputation with importance-sampling correction. Keep
``algorithm.recompute_logprobs`` and ``algorithm.importance_sampling_fix``
enabled together. See :doc:`PPO <../../reference/algorithms/ppo>` for the
correction and its diagnostic metrics.

The training integration uses PI0.5 with a value head after the VLM,
``value_vlm_mode: mean_token``, and ``joint_logprob: False``. Use synchronous,
non-pipelined training; the config validator checks these requirements before
workers launch.

Monitor the Run
---------------

The launcher writes logs under ``logs/<timestamp>-<config-name>/``. Monitor
``env/success_once`` for task success and the PPO importance-sampling metrics
for rollout-to-actor log-probability differences. See
:doc:`Training Metrics <../../reference/metrics>` for metric definitions.
