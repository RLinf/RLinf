Fine-Tune SD3 with ImageReward
==============================

.. figure:: https://raw.githubusercontent.com/zai-org/ImageReward/78f8051059b34fef8e0032ccb40561c23a2ddbbb/figures/ImageReward.jpg
   :align: center
   :width: 80%

   ImageReward overview, from the authors' official repository.

Fine-tune Stable Diffusion 3.5 Medium with Reward Feedback Learning (ReFL) to
improve ImageReward scores. Training uses text prompts, with optional
image-caption pairs for supervised updates.

Overview
--------

The ``sd3_refl_imagereward`` config trains SD3's LoRA weights on four NVIDIA GPUs.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      ReFL

   .. grid-item-card:: Models
      :text-align: center

      SD3.5 Medium

   .. grid-item-card:: Environments / Data
      :text-align: center

      Text prompts, image-caption pairs

   .. grid-item-card:: Training
      :text-align: center

      LoRA · FSDP2 · BF16

Tasks
~~~~~

Training optimizes text-to-image generation with ImageReward scores.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Model
     - Config
     - Reward
   * - SD3.5 Medium
     - ``sd3_refl_imagereward``
     - ImageReward human-preference score

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

Each prompt produces one image for ImageReward to score.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Value
     - Meaning
   * - Observation
     - A text prompt.
   * - Action
     - A generated RGB image.
   * - Reward
     - One ImageReward score per image; higher is better.

How ReFL Works
--------------

ReFL samples up to a late denoising step, then predicts a clean image in one
step. A frozen ImageReward model scores the image, and the reward gradient
updates the generator's LoRA weights. With the supervised loss Lpre enabled,
each reward update follows a supervised update on image-caption pairs.

This recipe applies the method from the
`ImageReward paper <https://arxiv.org/abs/2304.05977>`_ to SD3 flow matching.

Installation
------------

Install and activate the diffusion environment from the repository root:

.. code-block:: bash

   bash requirements/install.sh embodied --model diffusion
   source .venv/bin/activate

Download the Model and Data
---------------------------

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
       --output "$IR_DATA/$file"
   done

Training uses ``refl_data.json``; evaluation uses ``test.json``.
``actor.refl_data.exclude_path`` points to the evaluation file to exclude
those prompts from training.

Run It
------

Use ``examples/diffusion/config/sd3_refl_imagereward.yaml`` to train for 50 steps
on GPUs 0–3. Each step trains on 512 samples in global batches of 64,
for eight reward updates:

.. code-block:: bash

   REFL_ARGS=(
     'cluster.component_placement={actor\,env\,rollout:0-3}'
     actor.model.model_path="$SD3_MODEL"
     actor.refl_reward_model.model_path="$IR_MODEL/ImageReward.pt"
     actor.refl_reward_model.med_config="$IR_MODEL/med_config.json"
     actor.refl_data.path="$IR_DATA/refl_data.json"
     actor.refl_data.exclude_path="$IR_DATA/test.json"
     actor.refl_samples_per_step=512
     actor.global_batch_size=64
     actor.micro_batch_size=1
     actor.refl_pre_data.num_workers=0
     env.eval.dataset.path="$IR_DATA/test.json"
     env.eval.total_num_envs=4
     env.eval.rollout_epoch=128
     env.eval.reward.batch_size=1
     runner.max_epochs=50
     runner.max_steps=50
     runner.val_check_interval=10
     runner.save_interval=10
   )

   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
     "${REFL_ARGS[@]}" \
     runner.logger.experiment_name=sd3_refl_reward_only

The default ``algorithm.refl_pre_loss_weight=0.0`` trains with the reward loss
alone. The run evaluates and saves a checkpoint every 10 steps, and reports
``eval/image_reward`` as the evaluation score. Logs and checkpoints go under
``logs/<timestamp>-sd3_refl_imagereward/``.

Configure Reward Updates
~~~~~~~~~~~~~~~~~~~~~~~~

``actor.refl_samples_per_step`` sets the number of samples per step, in multiples
of the global batch size. The main reward settings are:

.. list-table::
   :header-rows: 1
   :widths: 32 15 53

   * - Setting
     - Default
     - Effect
   * - ``algorithm.refl_margin``
     - ``2.0``
     - Images at or above this score contribute zero reward gradient.
   * - ``algorithm.refl_grad_scale``
     - ``1e-3``
     - Multiplier on the reward loss.
   * - ``algorithm.refl_xt_source``
     - ``rollout``
     - Select a late trajectory state. ``resample`` re-noises the final latent.

Add Supervised Training
~~~~~~~~~~~~~~~~~~~~~~~

Pair each image with a same-stem UTF-8 ``.txt`` caption. The loader also searches
subdirectories:

.. code-block:: text

   image_text_data/
     00000.jpg
     00000.txt
     00001.png
     00001.txt

The experiments below use a subset of
`PD3M <https://huggingface.co/datasets/Spawning/PD3M>`_. Replace
``<image_text_dir>`` with your image-caption directory and run with Lpre weights
of ``0.1`` and ``0.01``:

.. code-block:: bash

   export PRE_DATA="<image_text_dir>"

   for WEIGHT in 0.1 0.01; do
     CUDA_VISIBLE_DEVICES=0,1,2,3 \
     bash examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
       "${REFL_ARGS[@]}" \
       actor.refl_pre_data.path="$PRE_DATA" \
       algorithm.refl_pre_loss_weight="$WEIGHT" \
       runner.logger.experiment_name="sd3_refl_pre${WEIGHT}"
   done

Each step has eight supervised updates and eight reward updates. Start with
Lpre weight ``0.01``, which gives higher evaluation scores in the results below.

Expected Results
----------------

The experiments use four NVIDIA A800 80 GB GPUs, SD3.5 Medium, LoRA (rank ``32``,
alpha ``64``), FSDP2, and BF16. Training runs for 50 steps with learning rate
``1e-4``, seed ``42``, and 512 samples per step.

Images are 512 × 512, with 10 training denoising steps and 40 evaluation steps.
Evaluation uses guidance scale ``4.5`` and generates 512 images from 463 prompts.
The table reports the mean ImageReward score.

.. list-table:: Mean Evaluation ImageReward
   :header-rows: 1
   :widths: 20 16 16 16 16 16

   * - Lpre Weight
     - Step 10
     - Step 20
     - Step 30
     - Step 40
     - Step 50
   * - ``0.1``
     - 0.1973
     - 0.6263
     - 0.6783
     - 0.7831
     - 0.9248
   * - ``0.01``
     - 0.5075
     - 0.7576
     - 0.9299
     - 1.1453
     - 1.1980

See :ref:`the ReFL interface guide <refl-model-interface>` to use another scorer
or generator, and :doc:`/rst_source/guides/logger` to view logs.
