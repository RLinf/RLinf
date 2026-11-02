Fine-Tune SD3 with ImageReward
==============================

.. figure:: https://raw.githubusercontent.com/zai-org/ImageReward/78f8051059b34fef8e0032ccb40561c23a2ddbbb/figures/ImageReward.jpg
   :align: center
   :width: 80%

   ImageReward overview. Source: the ImageReward authors' official repository.

Fine-tune Stable Diffusion 3.5 with a differentiable image-preference reward
using Reward Feedback Learning (ReFL). Start with reward-only training, then
optionally add supervised image-text updates to study their effect on image
quality. This example reuses RLinf's diffusion environment, rollout workers,
FSDP actor, weight synchronization, and training runner.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      Diffusion generation

   .. grid-item-card:: Algorithms
      :text-align: center

      ReFL

   .. grid-item-card:: Tasks
      :text-align: center

      Text-to-image preference tuning

   .. grid-item-card:: Hardware
      :text-align: center

      NVIDIA CUDA; four-GPU recipe below

The config is ``examples/diffusion/config/sd3_refl_imagereward.yaml``.
See :doc:`../video_models` for the shared diffusion environment and NFT examples.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Model
     - Reward
     - Training data
   * - SD3.5 Medium with LoRA
     - ImageReward human-preference score
     - Text prompts; optional captioned images for supervised updates

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Description
   * - Observation
     - ``task_descriptions``: a batch of text prompts.
   * - Action
     - Generated RGB images; each image completes one environment episode.
   * - Reward
     - ``avg`` and ``image_reward``: the same scalar score; higher is better.
   * - Update inputs
     - Detached late denoising states and prompt embeddings from rollout.

Installation
------------

From the repository root, install the diffusion environment following
:doc:`/rst_source/start/installation`:

.. code-block:: bash

   bash requirements/install.sh embodied --model diffusion
   source .venv/bin/activate

The diffusion requirements pin ``image-reward==1.5`` and ``timm==0.6.13``.
When reusing an environment, check these versions before training; a newer
``timm`` can conflict with ImageReward's dependency requirements.

Prepare Models and Prompts
--------------------------

Accept the SD3.5 model license on Hugging Face and authenticate with ``hf auth
login`` before downloading gated weights. Download the generator, the reward
checkpoint, the BERT tokenizer used by ImageReward, and the official prompts:

.. code-block:: bash

   export SD3_MODEL="$PWD/models/stable-diffusion-3.5-medium"
   export IR_MODEL="$PWD/models/ImageReward"
   export IR_DATA="$PWD/data/ImageReward"
   mkdir -p "$SD3_MODEL" "$IR_MODEL" "$IR_DATA"

   hf download stabilityai/stable-diffusion-3.5-medium --local-dir "$SD3_MODEL"
   hf download zai-org/ImageReward ImageReward.pt med_config.json --local-dir "$IR_MODEL"
   python -c 'from transformers import BertTokenizer; BertTokenizer.from_pretrained("bert-base-uncased")'

   for file in refl_data.json test.json; do
     curl --fail --location \
       "https://raw.githubusercontent.com/zai-org/ImageReward/78f8051059b34fef8e0032ccb40561c23a2ddbbb/data/$file" \
       --output "$IR_DATA/$file" || break
   done

Keep these environment variables in the shell used to launch training.
Every node must be able to load ``bert-base-uncased`` from its Hugging Face
cache. For an offline run, prepare that cache on every node first.

Each prompt JSON is a list of objects containing ``text`` or ``prompt``.
The loader trims whitespace, removes duplicates while preserving order, and
excludes the prompts in ``env.train.dataset.exclude_path`` from training.
For the pinned official files, this leaves 9,756 training and 463 evaluation
prompts.

Run Reward-Only Training
------------------------

Start with the reward objective alone so that later comparisons isolate the
supervised updates. The following four-GPU recipe processes 512 training
samples per runner step in global batches of 64:

.. code-block:: bash

   REFL_ARGS=(
     actor.model.model_path="$SD3_MODEL"
     rollout.model.model_path="$SD3_MODEL"
     actor.refl_reward_model.model_path="$IR_MODEL/ImageReward.pt"
     actor.refl_reward_model.med_config="$IR_MODEL/med_config.json"
     env.train.dataset.path="$IR_DATA/refl_data.json"
     env.train.dataset.exclude_path="$IR_DATA/test.json"
     env.eval.dataset.path="$IR_DATA/test.json"
     env.train.total_num_envs=4
     env.train.rollout_epoch=128
     env.eval.total_num_envs=4
     env.eval.rollout_epoch=128
     actor.global_batch_size=64
     actor.micro_batch_size=1
     actor.refl_pre_data.num_workers=0
     rollout.pipeline_stage_num=1
     env.train.reward.batch_size=1
     env.eval.reward.batch_size=1
     algorithm.refl_grad_scale=1e-3
     runner.max_epochs=200
     runner.max_steps=200
     runner.val_check_interval=10
     runner.save_interval=10
   )

   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash -o pipefail examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
     "${REFL_ARGS[@]}" \
     algorithm.refl_pre_loss_weight=0.0 \
     runner.logger.experiment_name=sd3_refl_reward_only

This starts a fresh 200-step run, evaluates every 10 steps, and saves every
10 steps. Choose GPU placement for your machine using
:doc:`/rst_source/start/installation`; micro-batch size 1 is a starting point,
not a memory guarantee. The actor holds a differentiable ImageReward model,
and environment reward scoring also consumes memory.

Add Supervised Image-Text Updates
---------------------------------

To continue the generator's denoising training alongside the reward objective,
set ``algorithm.refl_pre_loss_weight`` above zero. This enables the optional
pre-training loss, called Lpre. Prepare local image-caption pairs in the
img2dataset ``files`` layout:

.. code-block:: text

   image_text_data/
     00000.jpg
     00000.txt
     shard_01/
       00001.png
       00001.txt

The loader searches subdirectories for ``.jpg``, ``.jpeg``, ``.png``, and
``.webp`` images. Each image needs a UTF-8 caption with the same stem and a
``.txt`` suffix. Images are resized, center-cropped to 512 pixels, randomly
flipped, and normalized to [-1, 1]. The default ``caption_prefix`` removes
``The image shows`` and its following space only at the start of a caption.

Point the config at the image directory and start a separate run:

.. code-block:: bash

   export PRE_DATA=/path/to/image_text_data

   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash -o pipefail examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
     "${REFL_ARGS[@]}" \
     actor.refl_pre_data.path="$PRE_DATA" \
     algorithm.refl_pre_loss_weight=0.1 \
     runner.logger.experiment_name=sd3_refl_pre01

This run uses the same reward coefficient and rollout settings as the
reward-only run. Repeat with ``algorithm.refl_pre_loss_weight=0.01`` and a
new experiment name for a lower-weight comparison. Keep
``actor.refl_pre_data.num_workers=0`` to load images in the actor process;
this avoids creating DataLoader child processes after accelerator setup.

How the Two Updates Interact
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The actor uses a late rollout state to predict a clean latent, decodes it,
and differentiates the reward through the frozen VAE and ImageReward model.
The reward loss is

.. math::

   L_{\mathrm{reward}} = \lambda\,\mathbb{E}
   [\max(0, m - r(\hat{x}_0, c))],

where ``refl_grad_scale`` sets :math:`\lambda` and ``refl_margin`` sets
:math:`m`. ``refl_xt_source=rollout`` uses stored late states;
``resample`` re-noises the final rollout latent at a sampled late time.

The supervised objective for SD3 is flow matching:

.. math::

   z_t = (1-t)z_0 + t\epsilon, \qquad
   L_{\mathrm{pre}} = \mathbb{E}\left[
   \lVert v_\theta(z_t,t,c) - (\epsilon-z_0)\rVert^2\right].

For each global batch, the actor first takes an optimizer step on
``refl_pre_loss_weight * Lpre`` when enabled, then another step on Lreward.
Both steps share one AdamW optimizer and its moment estimates; reducing the
Lpre coefficient does not proportionally reduce its parameter update.
A zero coefficient skips the image loader and the supervised step entirely.

Lpre trains on your captioned images rather than constraining outputs to the
original SD3 model. Dataset mismatch or competing updates can lower the
ImageReward score. Compare with the reward-only run and inspect generated
images; a higher reward alone does not establish better visual quality.
See the `ImageReward paper <https://arxiv.org/abs/2304.05977>`_ for ReFL's
original SD1.4 formulation; this recipe adapts the denoising loss to SD3 flow
matching.

Verify Training
---------------

Before a long run with Lpre, exercise training, validation, and checkpoint
saving with a three-step run:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash -o pipefail examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
     "${REFL_ARGS[@]}" \
     actor.refl_pre_data.path="$PRE_DATA" \
     algorithm.refl_pre_loss_weight=0.1 \
     env.train.rollout_epoch=16 \
     env.eval.rollout_epoch=1 \
     runner.max_epochs=3 runner.max_steps=3 \
     runner.val_check_interval=3 runner.save_interval=3 \
     runner.logger.experiment_name=sd3_refl_smoke

Check that the process exits successfully and that the experiment's
``checkpoints/global_step_3/actor`` directory contains checkpoint files.
To exercise Lpre iterator rollover in this short run, use a separate
``PRE_DATA`` directory containing exactly 64 captioned images: each of the
four actor ranks consumes its 16-image partition once per runner step.
The second and third steps then recreate the iterator.

Run the CPU unit tests in the diffusion environment:

.. code-block:: bash

   pytest tests/unit_tests/test_diffusion.py

These tests use generated data and randomly initialized tiny models. A small
local module replaces the external ImageReward checkpoint while RLinf's
scorer runs normally. The tests cover prompt exclusion, image preprocessing, differentiable rewards, SD3
rollout/forward consistency, supervised gradients, and config composition.
They do not establish reward improvement with pretrained weights or validate
multi-GPU FSDP execution. Tests requiring optional dependencies are skipped when those dependencies
are unavailable.

Set ``env.eval.video_cfg.save_video=true`` to record evaluation samples under
``env.eval.video_cfg.video_base_dir``. The diffusion environment renders each
image as repeated video frames for the shared recording wrapper.

Monitor ``eval/image_reward`` and ``train/actor/refl_reward`` alongside generated
images. With Lpre enabled, also inspect ``train/actor/refl_pre_loss``,
``train/actor/refl_pre_grad_norm``, and ``train/actor/refl_reward_grad_norm``. Gradient
norm ratios are not AdamW parameter-update ratios. With 512 evaluation
samples and 463 prompts, 49 prompts repeat in each evaluation; compare runs
with the same evaluation settings. See
:doc:`/rst_source/guides/logger` for the shared logging workflow.
