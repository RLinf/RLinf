RL for Video Generation Models
==============================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/wan.png
   :align: center
   :width: 45%

   Video generation models trained with Diffusion-NFT in RLinf.

Use RLinf to fine-tune SD3 and Wan2.2 generation models with Diffusion-NFT.
The workflow reuses the embodied runner, registers image/video generation
models under ``actor.model``, and uses ``gen_reward`` to score generated media
with a one-step reward.

Overview
----------------------------------------

Train SD3 image generation models and Wan2.2 single-frame/video generation
models on the OCR prompt dataset.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      ``gen_reward``

   .. grid-item-card:: Algorithms
      :text-align: center

      Diffusion-NFT

   .. grid-item-card:: Tasks
      :text-align: center

      SD3 / Wan2.2

   .. grid-item-card:: Hardware
      :text-align: center

      1 node / GPUs

| **You'll do:** install dependencies → download models and dataset → choose a config → launch ``run_generation.sh`` → monitor ``avg`` reward.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · SD3 or Wan2.2 checkpoints · the OCR prompt dataset.

Tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose the config by generation model, output type, and training objective.

.. list-table::
   :header-rows: 1
   :widths: 24 28 24 24

   * - Model
     - Config
     - Output
     - Focus
   * - SD3
     - ``sd3_nft_ocr``
     - Image
     - Fine-tune SD3 image generation with OCR reward.
   * - Wan2.2 TI2V 5B
     - ``wan22_ti2v_5b_nft_ocr``
     - Image
     - Fine-tune Wan2.2 single-frame outputs with OCR reward.
   * - Wan2.2 TI2V 5B
     - ``wan22_ti2v_5b_nft_video_ocr``
     - Video
     - Fine-tune Wan2.2 multi-frame video outputs with OCR reward.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - Field
     - Description
   * - Observation
     - Text prompt sampled from the OCR prompt dataset.
   * - Action
     - Generated image or video output from SD3 or Wan2.2.
   * - Reward
     - Text-matching score computed by the OCR reward.
   * - Prompt
     - Natural-language text used to condition the generation model.

Installation
----------------------------------------

.. include:: _setup_common.rst

**Custom Environment**

.. code:: bash

   # Add --use-mirror for faster downloads in mainland China.
   bash requirements/install.sh embodied --model generation
   source .venv/bin/activate

To use a custom virtual environment directory, pass ``--venv <dir>``:

.. code:: bash

   bash requirements/install.sh embodied --model generation --venv /path/to/venv
   source /path/to/venv/bin/activate

This command creates a Python 3.10 environment and installs the SD3, Wan2.2,
and OCR reward dependencies, including Diffusers, PEFT, Transformers,
PaddleOCR, and PaddlePaddle.

.. warning::

   The checked-in generation configs use ``/path/to/...`` placeholders.
   Replace the model and dataset paths following the comments in the configs,
   or override ``actor.model.model_path`` and ``env.*.dataset.path`` at launch.

.. note::

   Wan2.2 requires a Diffusers version that supports
   ``Wan-AI/Wan2.2-TI2V-5B-Diffusers``. Use the installer above instead of an
   older embodied environment that already pins Diffusers for another model.

Download the Model
----------------------------------------

Before training, download the corresponding Diffusers checkpoint and set
``actor.model.model_path`` to the local model directory.

.. list-table::
   :header-rows: 1
   :widths: 22 36 42

   * - Model
     - Hugging Face Repo
     - Example Local Path
   * - Stable Diffusion 3.5 Medium
     - `stabilityai/stable-diffusion-3.5-medium <https://huggingface.co/stabilityai/stable-diffusion-3.5-medium>`__
     - ``/path/to/stable-diffusion-3.5-medium``
   * - Wan2.2 TI2V 5B
     - `Wan-AI/Wan2.2-TI2V-5B-Diffusers <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers>`__
     - ``/path/to/Wan2.2-TI2V-5B-Diffusers``

.. code:: bash

   huggingface-cli download stabilityai/stable-diffusion-3.5-medium \
      --local-dir /path/to/stable-diffusion-3.5-medium

   huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
      --local-dir /path/to/Wan2.2-TI2V-5B-Diffusers

After downloading, pass the same local path to ``actor.model.model_path`` when
launching. Stable Diffusion 3.5 Medium requires accepting the model access
terms on Hugging Face before downloading.

Download the Dataset
----------------------------------------

The SD3 and Wan2.2 examples use the OCR prompt dataset from
`NVlabs/DiffusionNFT <https://github.com/NVlabs/DiffusionNFT/tree/main/dataset/ocr>`__.
The dataset directory must contain ``train.txt`` and ``test.txt``; each line is
one prompt, and ``env.*.dataset.split`` selects the file to read.

.. code:: bash

   mkdir -p /path/to/dataset/ocr
   curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/train.txt \
      -o /path/to/dataset/ocr/train.txt
   curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/test.txt \
      -o /path/to/dataset/ocr/test.txt

Then set both dataset paths to the directory that contains these files.

Run It
----------------------------------------

Pass a generation config name to ``run_generation.sh`` and override the model
and dataset paths from the command line.

.. code:: bash

   bash examples/generation/run_generation.sh sd3_nft_ocr \
      actor.model.model_path=/path/to/stable-diffusion-3.5-medium \
      env.train.dataset.path=/path/to/dataset/ocr \
      env.eval.dataset.path=/path/to/dataset/ocr

.. code:: bash

   bash examples/generation/run_generation.sh wan22_ti2v_5b_nft_video_ocr \
      actor.model.model_path=/path/to/Wan2.2-TI2V-5B-Diffusers \
      env.train.dataset.path=/path/to/dataset/ocr \
      env.eval.dataset.path=/path/to/dataset/ocr

The training flow:

1. Samples prompt groups from the OCR prompt dataset.
2. Generates image or video candidates with SD3 or Wan2.2.
3. Scores the generated media with the OCR reward.
4. Applies the Diffusion-NFT update to the generation model.

Configure further in ``examples/generation/config/*.yaml``. Start with
``algorithm.group_size``, ``algorithm.nft_target_space``, ``algorithm.nft_tau``,
and ``actor.model.*.use_lora``.

Visualization and Results
----------------------------------------

Open TensorBoard on the generated log directory. Monitor ``avg`` reward and
``actor/*`` training metrics. If media saving is enabled in ``env.*.video_cfg``,
generated samples are written under ``video_base_dir``.

.. code:: bash

   tensorboard --host 0.0.0.0 --logdir logs/

For scalar definitions and logger layout, see :doc:`Training Metrics </rst_source/reference/metrics>`.
