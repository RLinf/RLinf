ReFL
====

.. figure:: https://raw.githubusercontent.com/zai-org/ImageReward/78f8051059b34fef8e0032ccb40561c23a2ddbbb/figures/ImageReward.jpg
   :align: center
   :width: 80%

   ImageReward overview, from the authors' official repository.

Fine-tune Stable Diffusion 3.5 Medium with Reward Feedback Learning (ReFL) to
improve ImageReward scores. Training uses text prompts, with optional
image-caption pairs for supervised updates. With Lpre weight ``0.01``, the
50-step run below reaches a mean evaluation ImageReward of ``1.1980``.

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

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: SD3.5 Medium

      **1. Install**

      Follow the shared :ref:`installation <diffusion-model-setup>` and
      :ref:`model download <diffusion-model-checkpoints>` instructions for SD3.5 Medium.

      **2. Download**

      Download the ImageReward checkpoint, tokenizer, and official prompt files:

      .. code-block:: bash

         export IR_MODEL="$PWD/models/ImageReward"
         export IR_DATA="$PWD/data/ImageReward"
         mkdir -p "$IR_MODEL" "$IR_DATA"

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

      **3. Launch**

      Edit ``examples/diffusion/config/sd3_refl_imagereward.yaml``. Set
      ``actor.model.model_path`` to the SD3.5 Medium directory. Set
      ``actor.refl_reward_model.model_path`` and ``actor.refl_reward_model.med_config``
      to the downloaded ``ImageReward.pt`` and ``med_config.json`` files.

      Set ``actor.refl_data.path`` to ``refl_data.json``. Point both
      ``actor.refl_data.exclude_path`` and ``env.eval.dataset.path`` to ``test.json``.
      To use GPUs 0–3, set the placement in the same YAML:

      .. code-block:: yaml

         cluster:
           component_placement:
             actor,env,rollout: 0-3

      The default config trains for 50 steps with 512 samples per step. A global
      batch size of 64 gives eight reward updates per step. Launch from the repository
      root:

      .. code-block:: bash

         bash examples/diffusion/run_diffusion.sh sd3_refl_imagereward

The default ``algorithm.refl_pre_loss_weight=0.0`` trains with the reward loss
alone. The run evaluates and saves a checkpoint every 10 steps, and reports
``eval/image_reward`` as the evaluation score. Logs and checkpoints go under
``logs/<timestamp>-sd3_refl_imagereward/``. Each actor checkpoint occupies about
9.4 GiB at
``<experiment_name>/checkpoints/global_step_<N>/actor/`` within that directory.
Set ``runner.resume_dir`` to the parent ``global_step_<N>`` directory to restore
the training state, including data positions and sampling RNGs.

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

The results below use 20,000 image-caption pairs from
`PD3M <https://huggingface.co/datasets/Spawning/PD3M>`_. Prepare them in a
separate environment with the steps below, or use your own image-caption
directory.

.. dropdown:: Optional: Prepare 20,000 PD3M image-caption pairs

   The script samples 20,000 image-caption pairs from the first two PD3M
   metadata shards using seed ``42``. It keeps images with license information,
   width and height of at least 512 pixels, and an aspect ratio of at most 2:1,
   while excluding iNaturalist images and captions that describe monochrome
   images or documents.

   .. code-block:: bash

      export PD3M_DIR="$PWD/data/PD3M"
      uv venv --python 3.10 "$PD3M_DIR/.venv"
      uv pip install --python "$PD3M_DIR/.venv/bin/python" \
        huggingface_hub pyarrow img2dataset==1.47.0

      "$PD3M_DIR/.venv/bin/hf" download Spawning/PD3M \
        metadata/pd3m.00.parquet metadata/pd3m.01.parquet \
        --repo-type dataset --local-dir "$PD3M_DIR"

      "$PD3M_DIR/.venv/bin/python" - "$PD3M_DIR" <<'PYTHON'
      import random
      import re
      import sys

      import pyarrow as pa
      import pyarrow.parquet as pq

      root = sys.argv[1]
      columns = ["id", "url", "caption", "width", "height", "license", "source"]
      rows = []
      for shard in ("00", "01"):
          rows += pq.read_table(
              f"{root}/metadata/pd3m.{shard}.parquet", columns=columns
          ).to_pylist()

      mono = re.compile(r"black[- ]and[- ]white|grayscale|sepia|monochrome", re.I)
      document = re.compile(
          r"\b(book|page|newspaper|document|letter|map|poster|stamp|coin|text (on|at|in))\b",
          re.I,
      )
      selected = [
          row for row in rows
          if row["license"]
          and row["source"] != "iNaturalist"
          and min(row["width"], row["height"]) >= 512
          and max(row["width"], row["height"]) <= 2 * min(row["width"], row["height"])
          and not mono.search(row["caption"])
          and not document.search(row["caption"])
      ]
      random.Random(42).shuffle(selected)
      pairs = [{key: row[key] for key in ("id", "url", "caption")}
               for row in selected[:20000]]
      pq.write_table(pa.Table.from_pylist(pairs), f"{root}/pd3m_refl.parquet")
      PYTHON

      export PRE_DATA="$PD3M_DIR/images_512"
      "$PD3M_DIR/.venv/bin/img2dataset" \
        --url_list="$PD3M_DIR/pd3m_refl.parquet" \
        --input_format=parquet --url_col=url --caption_col=caption \
        --output_format=files --output_folder="$PRE_DATA" \
        --image_size=512 --resize_mode=keep_ratio --resize_only_if_bigger=True \
        --number_sample_per_shard=1000 --processes_count=16 --thread_count=32 \
        --timeout=60 --retries=2 --enable_wandb=False

The downloader writes image-caption pairs under ``$PRE_DATA``. The loader
resizes and center-crops the images to 512 × 512. The config sets
``actor.refl_pre_data.caption_prefix`` to ``"The image shows "`` to remove that
opening from PD3M captions; set it to ``""`` to keep your own captions unchanged.

In the same YAML, set ``actor.refl_pre_data.path`` to the downloaded
``images_512`` directory and ``algorithm.refl_pre_loss_weight`` to ``0.01``.
Use ``0.1`` for the other comparison run. Launch each run with:

.. code-block:: bash

   bash examples/diffusion/run_diffusion.sh sd3_refl_imagereward

Each step has eight supervised updates and eight reward updates. Start with an
Lpre weight of ``0.01``. In the results below, this run scores higher than the
``0.1`` run.

Visualization and Results
-------------------------

The experiments use four NVIDIA A800 80 GB GPUs, SD3.5 Medium, LoRA (rank ``32``,
alpha ``64``), FSDP2, and BF16, with learning rate ``1e-4``, seed ``42``, and
512 samples per step. All three runs train for 50 steps.

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
   * - ``0`` (reward-only)
     - 0.9331
     - 1.0391
     - 1.1059
     - 1.1123
     - 1.1692
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
