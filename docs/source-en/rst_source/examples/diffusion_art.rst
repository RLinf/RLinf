DiffusionART
============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/wan.png
   :align: center
   :width: 45%

   SD3 and Wan2.2 generation models trained with DiffusionART.

Fine-tune SD3 or Wan2.2 with DiffusionART to improve text rendering in images
and videos using OCR rewards. Complete the :doc:`shared setup <video_models>`,
then launch the image or video config.

Overview
--------

Both configs use LoRA, FSDP2, and deterministic ODE sampling. OCR rewards measure
how closely the generated text matches the target text in the prompt.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      DiffusionART

   .. grid-item-card:: Models
      :text-align: center

      SD3 / Wan2.2

   .. grid-item-card:: Environments / Data
      :text-align: center

      OCR prompts

   .. grid-item-card:: Training
      :text-align: center

      LoRA · FSDP2 · BF16

Tasks
~~~~~

Choose the config for image or video generation.

.. list-table::
   :header-rows: 1
   :widths: 24 42 34

   * - Model
     - Config
     - Task
   * - SD3.5 Medium
     - ``sd3_art_ocr``
     - Image OCR
   * - Wan2.2 TI2V 5B
     - ``wan22_ti2v_5b_art_video_ocr``
     - Text-to-video OCR

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

Each generated sample receives one OCR score. For videos, the reward is the
mean score across sampled frames.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Value
     - Meaning
   * - Observation
     - A text prompt containing quoted target text.
   * - Action
     - A generated image or video.
   * - Reward
     - An OCR similarity score in ``[0, 1]``.

How ART Works
-------------

ART combines each prompt group's rewards and final latents into a shared
update direction. Training uses denoising states from the generated trajectories.
Both update windows share velocity targets frozen before the first update.

For the derivation and reference implementation, see the
`DiffusionART paper <https://arxiv.org/abs/2610.12572>`_,
`official code <https://github.com/vanzll/DiffusionART>`_, and
`project page <https://vanzll.github.io/diffusionart-project/>`_.

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run
``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt
images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__.
Both ART configs use CPU OCR; see :doc:`video_models` for full shared setup notes.

.. tab-set::

   .. tab-item:: SD3.5 Medium (Image OCR)

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model diffusion && source .venv/bin/activate

      **2. Download**

      Accept the SD3.5 Medium access terms on Hugging Face first, then download the checkpoint and OCR prompts:

      .. code-block:: bash

         hf download stabilityai/stable-diffusion-3.5-medium --local-dir ./stable-diffusion-3.5-medium
         mkdir -p ./ocr_data
         curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/train.txt -o ./ocr_data/train.txt
         curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/test.txt -o ./ocr_data/test.txt

      **3. Launch**

      Set ``actor.model.model_path``, ``env.train.dataset.path``, and ``env.eval.dataset.path`` in ``examples/diffusion/config/sd3_art_ocr.yaml`` (adjust GPU placement with ``cluster.component_placement``), then launch:

      .. code-block:: bash

         bash examples/diffusion/run_diffusion.sh sd3_art_ocr

   .. tab-item:: Wan2.2 TI2V 5B (Video OCR)

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model diffusion && source .venv/bin/activate

      **2. Download**

      .. code-block:: bash

         hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir ./Wan2.2-TI2V-5B-Diffusers
         mkdir -p ./ocr_data
         curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/train.txt -o ./ocr_data/train.txt
         curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/test.txt -o ./ocr_data/test.txt

      **3. Launch**

      Set ``actor.model.model_path``, ``env.train.dataset.path``, and ``env.eval.dataset.path`` in ``examples/diffusion/config/wan22_ti2v_5b_art_video_ocr.yaml`` (adjust GPU placement with ``cluster.component_placement``), then launch:

      .. code-block:: bash

         bash examples/diffusion/run_diffusion.sh wan22_ti2v_5b_art_video_ocr

Logs go under ``logs/<timestamp>-<config>/``. Training reports image OCR scores
as ``env/avg`` and video OCR scores as ``env/video_ocr``. For scalar definitions,
see :doc:`Training Metrics </rst_source/reference/metrics>`.

Configure ART Updates
~~~~~~~~~~~~~~~~~~~~~

Each round generates ``env.train.total_num_envs * env.train.rollout_epoch``
samples. The default ``algorithm.art_updates_per_rollout=2`` splits each round
into two updates, so ``actor.global_batch_size`` is half the samples per round.
For example, 96 environments and 12 rollout epochs give 1152 samples per round
and a global batch size of 576.

The main update settings are:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Setting
     - Effect
   * - ``algorithm.art_timestep_fraction``
     - Fraction of denoising states used for training: ``0.5`` for SD3, ``0.25`` for Wan.
   * - ``algorithm.art_guidance_gain``
     - Reward guidance scale; defaults to ``1.0``.
   * - ``algorithm.art_corrective_mask_enabled``
     - Enables the corrective movement mask; defaults to ``True``.
   * - ``algorithm.art_mask_threshold``
     - Movement threshold for the mask; defaults to ``1e-5`` with the ``sde_kl`` metric.

.. _diffusion-art-results:

Expected Results
----------------

The experiments use four NVIDIA A800 80 GB GPUs, Wan2.2 TI2V 5B, LoRA
(rank ``32``, alpha ``64``), FSDP2, BF16, and CPU OCR. Generation uses
512 × 512 resolution, five frames, ten denoising steps, and guidance scale ``1.0``.

ART uses the ``wan22_ti2v_5b_art_video_ocr`` config; NFT uses
``wan22_ti2v_5b_nft_video_ocr``. Training reward is the mean ``env/video_ocr``
over the last 5760 trajectories. For repeated runs, the table reports the mean
± sample standard deviation.

The first comparison uses ``video_level`` rewards for ART and ``frame_level``
rewards for NFT.

.. list-table:: Wan Video OCR at 40,320 Generated Trajectories
   :header-rows: 1
   :widths: 24 8 16 12 20 20

   * - Method
     - Runs
     - Trajectories per Round
     - Learning Rate
     - Training Reward
     - Training Time (h)
   * - ART
     - 1
     - 1152
     - 1e-4
     - 0.8584
     - 10.01
   * - NFT
     - 2
     - 576
     - 4e-4
     - 0.7565 ± 0.0544
     - 11.29 ± 0.52

With these configurations, ART's training reward at 40,320 trajectories is
**13.5% higher** and its training time **11.3% shorter** than the NFT averages. Training time comes
from the training-loop timer.

The second comparison matches trajectories per round and learning rate.
Both methods use ``video_level`` rewards, with one run each. The table compares
the first 29 rounds of each run.

.. list-table:: Wan Video OCR at 33,408 Generated Trajectories
   :header-rows: 1
   :widths: 18 22 18 22 20

   * - Method
     - Trajectories per Round
     - Learning Rate
     - Training Reward
     - Training Time (h)
   * - ART
     - 1152
     - 1e-4
     - 0.8446
     - 8.31
   * - NFT
     - 1152
     - 1e-4
     - 0.1076
     - 8.23

To reproduce the experiments, set the model and data paths in the chosen
config and apply these shared settings:

.. dropdown:: Advanced configuration (optional)

   .. code-block:: yaml

      cluster:
        component_placement:
          actor,env,rollout: "0-3"
      algorithm:
        group_size: 8
      actor:
        seed: 42
        model:
          wan22_ti2v_5b:
            max_generation_batch_size: 2
            compile_transformer_forward: True
            offload_auxiliary_modules: True
      env:
        train:
          total_num_envs: 96
          seed: 42
          reward:
            use_gpu: False
        eval:
          seed: 42
          reward:
            use_gpu: False
      runner:
        val_check_interval: -1

The batch and step settings for each configuration are below. Both NFT
configurations use ``algorithm.nft_tau=[1.0,0.01,0,70]``.

.. list-table:: Training Configurations
   :header-rows: 1
   :widths: 46 18 18 18

   * - Setting
     - ART, 1152/Round
     - NFT, 576/Round
     - NFT, 1152/Round
   * - ``env.train.rollout_epoch``
     - 12
     - 6
     - 12
   * - ``actor.global_batch_size``
     - 576
     - 288
     - 576
   * - ``actor.micro_batch_size``
     - 2
     - 1
     - 2
   * - ``actor.optim.lr``
     - 1e-4
     - 4e-4
     - 1e-4
   * - ``env.train.reward.reward_type``
     - ``video_level``
     - ``frame_level``
     - ``video_level``
   * - ``runner.max_epochs`` / ``runner.max_steps``
     - 70
     - 70
     - 200
   * - ``runner.save_interval``
     - 10
     - -1
     - 10
