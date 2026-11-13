Image and Video Generation
==========================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/wan.png
   :align: center
   :width: 45%

   SD3 and Wan2.2 image/video generation models.

Fine-tune SD3 and Wan2.2 with OCR rewards using Diffusion-NFT or DiffusionART.
Both methods use the same models, prompt dataset, and diffusion environment.
Prepare these shared dependencies below, then follow a training guide.

Choose a Training Method
------------------------

Choose the method before launching training; the guides contain its configs,
update settings, and expected results.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Diffusion-NFT
      :link: diffusion_nft
      :link-type: doc

      Train from relative rewards with positive/negative velocity candidates.
      Includes SD3 image OCR and Wan2.2 image/video OCR recipes.

   .. grid-item-card:: DiffusionART
      :link: diffusion_art
      :link-type: doc

      Build shared reward guidance over prompt groups and train against frozen
      velocity targets. Includes SD3 image OCR and Wan2.2 video OCR recipes.

Prompts, Outputs, and Rewards
-----------------------------

Each prompt describes an image or video and contains quoted target text, such as
a sign reading "OPEN". The diffusion environment scores the generated media and
ends the episode. ``algorithm.group_size`` controls how many candidates share a
prompt for reward comparison.

.. list-table:: Shared Training Data
   :header-rows: 1
   :widths: 24 76

   * - Value
     - Meaning
   * - Observation
     - A text prompt from ``env.*.dataset``.
   * - Action
     - The generated image or video.
   * - Reward
     - OCR text similarity in ``[0, 1]``; higher is better.

Image OCR compares recognized text with the quoted target using normalized
Levenshtein distance; an exact substring match also scores ``1``. Video OCR
applies the same scorer to frames. The NFT video recipe retains frame-aware
rewards, while the ART video recipe averages frame scores into one reward per
candidate. Inspect generated samples as well as reward: OCR does not measure
visual quality or temporal consistency.

.. _diffusion-model-setup:

Installation
------------

Start from the :doc:`RLinf installation prerequisites
</rst_source/start/installation>` and run the following commands from the
repository root on your GPU training host:

.. code:: bash

   bash requirements/install.sh embodied --model diffusion
   source .venv/bin/activate

The installer creates a Python 3.10 environment with Diffusers, PEFT,
Transformers, PaddleOCR, and CPU PaddlePaddle. Both ART recipes and the SD3 NFT
recipe use CPU OCR. Pass ``--venv /path/to/venv`` to install in another directory
and activate that environment instead.

The Wan NFT recipes set ``env.*.reward.use_gpu: True``. For GPU OCR, install GPU
PaddlePaddle with ``--paddle-gpu`` (CUDA 11.x or 12.x), then activate the resulting
environment:

.. code:: bash

   bash requirements/install.sh embodied --model diffusion --paddle-gpu
   source .venv/bin/activate

.. note::

   Use the diffusion installer for Wan2.2. Its dependencies support
   ``Wan-AI/Wan2.2-TI2V-5B-Diffusers``; an environment pinned for another model
   may contain an older Diffusers version.

.. _diffusion-model-checkpoints:

Download the Model
------------------

Download the checkpoint for the recipe you plan to run. Accept the SD3.5 Medium
access terms on Hugging Face before downloading that model.

.. code:: bash

   huggingface-cli download stabilityai/stable-diffusion-3.5-medium \
      --local-dir /path/to/stable-diffusion-3.5-medium

   huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
      --local-dir /path/to/Wan2.2-TI2V-5B-Diffusers

These commands save Diffusers checkpoints locally. Set
``actor.model.model_path`` in your selected training YAML to the corresponding
directory.

.. _diffusion-ocr-dataset:

Download the Dataset
--------------------

Both methods use the OCR prompts from
`NVlabs/DiffusionNFT <https://github.com/NVlabs/DiffusionNFT/tree/main/dataset/ocr>`__.
Download the training and test splits into one directory:

.. code:: bash

   mkdir -p /path/to/dataset/ocr
   curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/train.txt \
      -o /path/to/dataset/ocr/train.txt
   curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/test.txt \
      -o /path/to/dataset/ocr/test.txt

Each line is one prompt. ``env.*.dataset.split`` selects ``train.txt`` or
``test.txt``. In your selected training YAML, set ``env.train.dataset.path``
and ``env.eval.dataset.path`` to this directory. With the model and data in
place, continue with
:doc:`Diffusion-NFT <diffusion_nft>` or :doc:`DiffusionART <diffusion_art>`.

.. warning::

   Replace the ``/path/to/...`` placeholders in the download commands and your
   selected training YAML with local paths before launching training.

.. toctree::
   :hidden:
   :maxdepth: 1

   Diffusion-NFT <diffusion_nft>
   DiffusionART <diffusion_art>
