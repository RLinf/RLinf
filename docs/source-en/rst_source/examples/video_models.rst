Image & Video Generation
========================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/wan.png
   :align: center
   :width: 45%

   SD3 and Wan2.2 image/video generation models.

Fine-tune SD3 and Wan2.2 with OCR rewards using **Diffusion-NFT** or **DiffusionART**, or tune SD3.5 Medium with ImageReward using **ReFL**. Diffusion-NFT and DiffusionART share the models, OCR prompt dataset, and diffusion environment setup below; ReFL has its own self-contained setup on its recipe page.

Choose a Training Method
------------------------

Choose the method before launching training; the guides contain its configs,
update settings, and expected results.

.. grid:: 1 3 3 3
   :gutter: 3

   .. grid-item-card:: DiffusionART
      :link: diffusion_art
      :link-type: doc

      Build shared reward guidance over prompt groups and train against frozen
      velocity targets. Includes SD3 image OCR and Wan2.2 video OCR recipes.

   .. grid-item-card:: Diffusion-NFT
      :link: diffusion_nft
      :link-type: doc

      Train from relative rewards with positive/negative velocity candidates.
      Includes SD3 image OCR and Wan2.2 image/video OCR recipes.

   .. grid-item-card:: ReFL
      :link: embodied/refl
      :link-type: doc

      Fine-tune SD3.5 Medium with differentiable ImageReward feedback and
      optional supervised image-text updates.

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
</rst_source/start/installation>` (in mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``):

.. tab-set::

   .. tab-item:: CPU OCR (SD3 / ART)

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model diffusion && source .venv/bin/activate

   .. tab-item:: GPU OCR (Wan NFT)

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model diffusion --paddle-gpu && source .venv/bin/activate

The installer creates a Python 3.10 environment with Diffusers, PEFT,
Transformers, and PaddleOCR (``--paddle-gpu`` supports CUDA 11.x/12.x for Wan
NFT recipes). Pass ``--venv /path/to/venv`` to install into another directory.

.. note::

   Use the diffusion installer for Wan2.2. Its dependencies support
   ``Wan-AI/Wan2.2-TI2V-5B-Diffusers``; an environment pinned for another model
   may contain an older Diffusers version.

.. _diffusion-model-checkpoints:

Download the Model
------------------

Download the checkpoint for the recipe you plan to run (accept the SD3.5 Medium
access terms on Hugging Face first if using SD3.5):

.. tab-set::

   .. tab-item:: SD3.5 Medium

      .. code-block:: bash

         hf download stabilityai/stable-diffusion-3.5-medium --local-dir ./stable-diffusion-3.5-medium

   .. tab-item:: Wan2.2 TI2V 5B

      .. code-block:: bash

         hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir ./Wan2.2-TI2V-5B-Diffusers

Pass the downloaded checkpoint and dataset paths directly via Hydra CLI overrides when launching ``run_diffusion.sh`` (or update ``actor.model.model_path``, ``env.train.dataset.path``, and ``env.eval.dataset.path`` in the YAML):

.. code-block:: bash

   bash examples/diffusion/run_diffusion.sh sd3_nft_ocr \
       actor.model.model_path=./stable-diffusion-3.5-medium \
       env.train.dataset.path=./ocr_data \
       env.eval.dataset.path=./ocr_data

.. _diffusion-ocr-dataset:

Download the Dataset
--------------------

Both methods use the OCR prompts from
`NVlabs/DiffusionNFT <https://github.com/NVlabs/DiffusionNFT/tree/main/dataset/ocr>`__.
Download the training and test splits into one directory:

.. code-block:: bash

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

   DiffusionART <diffusion_art>
   Diffusion-NFT <diffusion_nft>
   ReFL <embodied/refl>
