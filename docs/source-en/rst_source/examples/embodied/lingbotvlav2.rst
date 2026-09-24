PPO on LingBot-VLA 2.0
======================

Fine-tune the LingBot-VLA 2.0 RoboTwin checkpoint on ``click_bell`` with PPO and
GAE, then evaluate the saved policy. RLinf trains the action expert and value
head with FSDP2 while freezing Qwen3-VL. The default training placement requires
four GPUs with 48 GB each. The model is installed separately, following
:doc:`../../extending/model_adaptation`.

Installation
------------

Prepare a RoboTwin checkout with RLinf's ``robotwin.envs.vector_env`` interface.
The installer pins the official V2 source and applies the compatibility patch
in ``requirements/embodied/models/lingbotvlav2/``. It adds checkpoint metadata,
lazy dataset imports and MoE hooks for likelihood replay and checkpointing.
Use a separate venv and source checkout from :doc:`LingBot-VLA 1.0 <lingbotvla>`:
both packages import as ``lingbotvla``.

.. code-block:: bash

   export ROBOTWIN_PATH=/path/to/RoboTwin
   bash requirements/install.sh embodied --model lingbotvlav2 --env robotwin \
     --torch 2.9.0 --transformers 4.57.6
   source .venv/bin/activate
   export LINGBOT_VLA_V2_PATH="${LINGBOT_VLA_V2_PATH:-$VIRTUAL_ENV/lingbot-vla-v2}"

To reuse a checkout, set ``LINGBOT_VLA_V2_PATH`` before installation. The installer
checks the revision and original/patched hashes in ``source.json`` and refuses
conflicting edits. The verified runtime uses PyTorch 2.9.0+cu126, Transformers
4.57.6 and FlashAttention 2.8.3.post1.

The Docker target uses the same installer:

.. code-block:: bash

   docker buildx build -f docker/Dockerfile \
     --build-arg BUILD_TARGET=embodied-robotwin-lingbotvlav2 \
     -t rlinf:robotwin-lingbotvlav2 .

Model and Assets
----------------

Download the official `V2 RoboTwin release
<https://huggingface.co/robbyant/lingbot-vla-v2-6b-robotwin>`_ and
`Qwen3-VL-4B-Instruct <https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct>`_.
Preserve the V2 release directory structure: its checkpoint config contains only
a family marker, so strict weight loading also needs the enclosing
``lingbotvla_cli.yaml`` and Qwen metadata.

.. code-block:: bash

   export LINGBOT_VLA_V2_CKPT=/path/to/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt
   export LINGBOT_VLA_V2_TRAIN_CONFIG=/path/to/lingbot-vla-v2-6b-robotwin/lingbotvla_cli.yaml
   export QWEN3_VL_PATH=/path/to/Qwen3-VL-4B-Instruct
   # RoboTwin expects the root containing assets/.
   export ROBOTWIN_ASSETS_PATH="$ROBOTWIN_PATH"
   export REPO_PATH=$(pwd)
   export PYTHONPATH="$REPO_PATH:$LINGBOT_VLA_V2_PATH:$ROBOTWIN_PATH"

The source and both model cards declare Apache-2.0. The compatibility patch
retains source notices and includes the upstream license. Both public model
repositories were non-gated when checked on 2026-09-22.

Training
--------

Start with the two-update smoke test: it runs two environments for 100 simulator
steps per episode and checks rollout/actor likelihood agreement before each
update. Actor ranks use GPUs 0–1, rollout uses GPU 2 and environments use GPU 3;
change ``cluster.component_placement`` to select resources.

.. code-block:: bash

   bash tests/e2e_tests/embodied/run.sh robotwin_ppo_lingbotvlav2 \
     runner.logger.log_path=/path/to/results/vla2_smoke

The model consumes head and both wrist RGB images in HWC order, 14D state and
an instruction. Upstream ``FeatureTransform`` owns preprocessing and robot
layout. Actions are ``[B, 50, 14]``; model latents have 55 coordinates, including
grippers at indices 28 and 29. V2 retains its depth/video query tokens in the
prefix. PPO log-probs are ``[B, 2750]`` and values are ``[B, 1]``.

PPO scores the complete flow-SDE latent trajectory. Keep
``algorithm.logprob_type=chunk_level``, ``algorithm.reward_type=chunk_level`` and
``logprob_dim=55``. Recorded latents and density calculations remain FP32 with
BF16 weights, so ``actor.fsdp_config.mixed_precision.cast_forward_inputs=false``
is required. ``forward_micro_batch_size=1`` aligns MoE batch shapes; a joint
log-prob gap above ``logprob_replay_atol`` stops the update.

The smoke saves a checkpoint after two updates. Resume with
``runner.resume_dir`` while retaining the original ``actor.model.model_path``
for architecture construction. Once the smoke passes, launch the full recipe:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh robotwin_click_bell_ppo_lingbotvlav2 ALOHA

Check ``train/actor/total_loss``, ``train/actor/grad_norm``,
``train/actor/rollout_train_joint_logprob_gap_max`` and environment success in
TensorBoard. The optional ``LINGBOT_VLA_V2_CI=true`` repository variable enables
the CI jobs, which need provisioned weights and RoboTwin assets under
``/workspace/dataset`` plus four GPUs. Fresh installation and Docker builds
require separate validation from tests in an existing runtime.

Evaluation
----------

Use the standalone rollout configuration to evaluate with flow integration.
Set ``runner.ckpt_path`` to an RLinf actor's ``model_state_dict/full_weights.pt``;
omit it to evaluate the original checkpoint. Keep the original model paths
available to construct the architecture.

.. code-block:: bash

   bash evaluations/run_eval.sh robotwin_click_bell_lingbotvlav2_eval \
     runner.ckpt_path=/path/to/actor/model_state_dict/full_weights.pt

The upstream benchmark uses FP32 and warns that BF16 can change task success.
A short BF16 PPO test demonstrates execution, not convergence or benchmark
parity. Export to the upstream deployment format has not been validated.
