Fine-Tune SD3 with ImageReward
==============================

.. figure:: https://raw.githubusercontent.com/zai-org/ImageReward/78f8051059b34fef8e0032ccb40561c23a2ddbbb/figures/ImageReward.jpg
   :align: center
   :width: 80%

   ImageReward overview. Source: the ImageReward authors' official repository.

Fine-tune Stable Diffusion 3.5 Medium with Reward Feedback Learning (ReFL)
by backpropagating an ImageReward preference loss through generated images.
Start with reward-only training, then add supervised image-text updates to
compare the two training setups.

Overview
--------

The recipe uses LoRA and the diffusion environment: each prompt produces an
image, and its ImageReward score supplies the reward.

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

| **You'll do:** install → prepare models and prompts → train → inspect images and rewards.
| **Prerequisites:** an SD3.5 model license and four NVIDIA GPUs for the command below.

Tasks
~~~~~

Use ``examples/diffusion/config/sd3_refl_imagereward.yaml`` for this recipe.
The supervised update takes captioned images separately from the prompt dataset.

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

Each image completes one environment episode. Rollout also stores late
denoising states for the differentiable actor update.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Description
   * - Observation
     - ``task_descriptions``: a batch of text prompts.
   * - Action
     - Generated RGB images.
   * - Reward
     - ``avg`` and ``image_reward``: the same scalar score; higher is better.
   * - Update inputs
     - Detached late denoising states and prompt embeddings from rollout.

Installation
------------

From the repository root, install and activate the diffusion environment:

.. code-block:: bash

   bash requirements/install.sh embodied --model diffusion
   source .venv/bin/activate

The installer selects ``image-reward==1.5`` and its required
``timm==0.6.13``. Keep these versions when reusing a training environment.
See :doc:`/rst_source/start/installation` for installation options.

Download the Model
------------------

Accept the SD3.5 license on Hugging Face and run ``hf auth login``. Download
the generator, ImageReward checkpoint, tokenizer, and official prompt files:

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

Keep these variables in the shell used for training. On multiple nodes,
prepare the ``bert-base-uncased`` tokenizer cache on each node.

Prompt files contain a JSON list of objects with a ``text`` or ``prompt``
field. The loader trims whitespace, removes duplicates, and excludes evaluation
prompts from training through ``env.train.dataset.exclude_path``. The pinned
files yield 9,756 training prompts and 463 evaluation prompts.

Run It
------

Start with reward-only training to establish a comparison for supervised
updates. The command below places actor, env, and rollout on GPUs 0–3.

Reward-Only Training
~~~~~~~~~~~~~~~~~~~~

Set the local paths and launch a 200-step run. Each runner step generates
512 training samples, processed in eight global batches of 64:

.. code-block:: bash

   REFL_ARGS=(
     'cluster.component_placement={actor\,env\,rollout:0-3}'
     actor.model.model_path="$SD3_MODEL"
     rollout.model.model_path="$SD3_MODEL"
     actor.model.sd3.lora_path=null
     actor.refl_reward_model.type=image_reward
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
     actor.seed=42
     env.train.seed=42
     env.eval.seed=0
     runner.max_epochs=200
     runner.max_steps=200
     runner.val_check_interval=10
     runner.save_interval=10
     runner.resume_dir=null
     runner.ckpt_path=null
   )

   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash -o pipefail examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
     "${REFL_ARGS[@]}" \
     algorithm.refl_pre_loss_weight=0.0 \
     runner.logger.experiment_name=sd3_refl_reward_only

The run evaluates and saves a checkpoint every 10 steps. For a short first
check, override ``runner.max_epochs=2``, ``runner.max_steps=2``,
``runner.val_check_interval=2``, and ``runner.save_interval=-1``.
Micro-batch size 1 is a starting point; both actor-side differentiable scoring
and environment scoring consume GPU memory. Adjust placement using
:doc:`/rst_source/concepts/placement`.

The actor scorer is selected by ``actor.refl_reward_model.type``; its default
is ``image_reward``. For environment scoring, keep ``env.*.reward.model``
set to ``visual.image_reward``.

Configure the Reward Update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The actor selects a late denoising state, predicts a clean latent in one step,
and decodes it to an image. The VAE and ImageReward parameters stay frozen,
while gradients pass through both to the generator. The loss is

.. math::

   L_{\mathrm{reward}} = \lambda\,\mathbb{E}
   [\max(0, m - r(\hat{x}_0, c))].

Here ``refl_grad_scale`` sets :math:`\lambda` and ``refl_margin`` sets
:math:`m`; samples scoring at least :math:`m` have zero reward gradient.
These settings live under ``algorithm``:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Setting
     - Default
     - Effect
   * - ``refl_margin``
     - ``2.0``
     - Reward threshold in the hinge loss.
   * - ``refl_grad_scale``
     - ``1e-3``
     - Multiplier on the reward loss.
   * - ``refl_xt_source``
     - ``rollout``
     - Select a stored late state. ``resample`` re-noises the final latent at a sampled late time.
   * - ``refl_pre_loss_weight``
     - ``0.0``
     - A positive value enables the supervised update below.

The `ImageReward paper <https://arxiv.org/abs/2304.05977>`_ introduces ReFL
with SD1.4. This recipe applies its reward feedback to SD3 flow matching.

Add Supervised Image-Text Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To continue denoising training alongside the reward update, prepare local
image-caption pairs in the img2dataset ``files`` layout. This optional
supervised objective is called Lpre:

.. code-block:: text

   image_text_data/
     00000.jpg
     00000.txt
     shard_01/
       00001.png
       00001.txt

The loader searches subdirectories for ``.jpg``, ``.jpeg``, ``.png``, and
``.webp`` images with same-stem UTF-8 ``.txt`` captions. It resizes and
center-crops images to 512 pixels, randomly flips them, and normalizes pixels
to ``[-1, 1]``. The default ``caption_prefix`` removes ``The image shows``
and its following space from the start of a caption. You can use your own pairs
or prepare a subset from
`PD3M <https://huggingface.co/datasets/Spawning/PD3M>`_ in this layout.

Set the image-caption directory and reuse the settings above:

.. code-block:: bash

   export PRE_DATA=/path/to/image_text_data

   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash -o pipefail examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
     "${REFL_ARGS[@]}" \
     actor.refl_pre_data.path="$PRE_DATA" \
     algorithm.refl_pre_loss_weight=0.1 \
     runner.logger.experiment_name=sd3_refl_pre01

For each global batch, the actor first takes an optimizer step on
``refl_pre_loss_weight * Lpre``, then another on the reward loss. Lpre uses
SD3's flow-matching denoising objective. A zero weight skips the image loader
and the supervised step.

The two steps share AdamW moment estimates, so a weight of ``0.01`` does not
mean a supervised parameter update of 1% of the update at weight ``1.0``.
The example value ``0.1`` is a comparison setting, not an established optimum.
When comparing weights, hold the dataset, initial model, learning rate, seeds,
batches, and run length fixed. Lpre adds eight optimizer steps per runner
step at the settings above, so also record the extra compute.

Visualization and Results
-------------------------

Inspect generated images alongside ``eval/image_reward`` and
``train/actor/refl_reward``. With Lpre enabled, also check
``train/actor/refl_pre_loss``, ``train/actor/refl_pre_grad_norm``, and
``train/actor/refl_reward_grad_norm``. Lpre logs its unweighted loss; the reward
loss includes ``refl_grad_scale``. A higher ImageReward score alone does not
establish better visual quality.

Evaluation generates 512 images from 463 prompts, so 49 prompts repeat. Keep
the evaluation settings fixed across comparisons. Set
``env.eval.video_cfg.save_video=true`` to save generated images as repeated
video frames under ``env.eval.video_cfg.video_base_dir``.

See :doc:`/rst_source/guides/logger` for log locations and
:doc:`/rst_source/reference/metrics` for shared metrics. For a different
scorer or model, follow :ref:`the ReFL interface guide <refl-model-interface>`.
