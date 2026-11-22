Diffusion-NFT
=============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/wan.png
   :align: center
   :width: 45%

   SD3 and Wan2.2 generation models trained with Diffusion-NFT.

Fine-tune SD3 or Wan2.2 with Diffusion-NFT to improve text rendering in images
and videos. Candidates generated for the same prompt are scored with OCR;
relative rewards guide the model update. After the
:doc:`shared setup <video_models>`, select a recipe below, launch training, and
inspect rewards and generated samples.

Overview
--------

Select the output type and matching recipe for your model. For the objective
and derivation, see the
`DiffusionNFT paper <https://arxiv.org/abs/2509.16117>`_ and
`official code <https://github.com/NVlabs/DiffusionNFT>`_.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      SD3 / Wan2.2

   .. grid-item-card:: Algorithm
      :text-align: center

      Diffusion-NFT

   .. grid-item-card:: Output
      :text-align: center

      Image / Video

   .. grid-item-card:: Hardware
      :text-align: center

      1–8 GPUs (SD3.5 Medium); 4–8× 80 GB GPUs (Wan2.2 TI2V 5B)

| **You'll do:** select a recipe → launch training → inspect OCR reward and generated samples.
| **Prerequisites:** :doc:`Shared installation, model, and dataset setup <video_models>` · 1–8 GPUs for SD3.5 Medium (BF16 + LoRA), 4–8× 80 GB GPUs (A800/A100/H100) for Wan2.2 TI2V 5B.

Tasks
~~~~~

The configs cover image OCR and frame-aware video OCR.

.. list-table::
   :header-rows: 1
   :widths: 22 34 18 26

   * - Model
     - Config
     - Output
     - Reward granularity
   * - SD3
     - ``sd3_nft_ocr``
     - Image
     - One OCR score per image
   * - Wan2.2 TI2V 5B
     - ``wan22_ti2v_5b_nft_ocr``
     - Single frame
     - One OCR score per image
   * - Wan2.2 TI2V 5B
     - ``wan22_ti2v_5b_nft_video_ocr``
     - Video
     - OCR scores over video frames

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

Each episode scores one generated image or video for its prompt.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Value
     - Meaning
   * - Observation
     - A text prompt containing quoted target text.
   * - Action
     - The generated image or video.
   * - Reward
     - Image OCR similarity or frame-aware video OCR scores.

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run
``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt
images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__.
See :doc:`video_models` for full shared setup notes.

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

      Set ``actor.model.model_path``, ``env.train.dataset.path``, and ``env.eval.dataset.path`` in ``examples/diffusion/config/sd3_nft_ocr.yaml`` (check ``cluster.num_nodes``, ``cluster.component_placement``, and batch settings against your hardware), then launch:

      .. code-block:: bash

         bash examples/diffusion/run_diffusion.sh sd3_nft_ocr

   .. tab-item:: Wan2.2 TI2V 5B (Image / Video OCR)

      **1. Install**

      Wan NFT recipes use GPU OCR (``--paddle-gpu``):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model diffusion --paddle-gpu && source .venv/bin/activate

      **2. Download**

      .. code-block:: bash

         hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir ./Wan2.2-TI2V-5B-Diffusers
         mkdir -p ./ocr_data
         curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/train.txt -o ./ocr_data/train.txt
         curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/test.txt -o ./ocr_data/test.txt

      **3. Launch**

      Set ``actor.model.model_path``, ``env.train.dataset.path``, and ``env.eval.dataset.path`` in ``examples/diffusion/config/wan22_ti2v_5b_nft_ocr.yaml`` (single-frame) or ``examples/diffusion/config/wan22_ti2v_5b_nft_video_ocr.yaml`` (video), then launch:

      .. code-block:: bash

         # Single-frame image OCR
         bash examples/diffusion/run_diffusion.sh wan22_ti2v_5b_nft_ocr
         # Video OCR
         bash examples/diffusion/run_diffusion.sh wan22_ti2v_5b_nft_video_ocr

The launcher generates candidates, scores them with OCR, updates LoRA
parameters, and writes logs under ``logs/<timestamp>-<config>/``.

Set ``runner.resume_dir`` to a ``checkpoints/global_step_N`` directory to
resume NFT training, including off-policy EMA rollout weights. If an older
checkpoint lacks EMA weights, the actor logs a warning and initializes EMA from
the restored actor weights; this fallback is not an exact resume.

Configure the NFT Update
------------------------

The checked-in recipes combine grouped reward normalization with an NFT update
in diffusion velocity space:

1. For each prompt, generate ``group_size`` candidates and compute relative
   advantages. Image recipes use standard GRPO normalization within each prompt
   group. The video recipe normalizes frame/chunk rewards over the whole video
   group.
2. Sample a denoising step and reconstruct the current noisy state
   (``nft_xcur_source: resample``). The rollout/reference model supplies the old
   velocity prediction and the trainable model predicts the updated velocity.
3. Compute the velocity difference between the trainable and reference models,
   then form positive and negative perturbed candidates scaled by ``nft_beta``.
4. Convert both candidates to the configured target space. The provided configs
   use ``nft_target_space: x0`` and compare the predicted clean sample against
   the rollout target using weighted squared-error energies for the positive and
   negative candidates.
5. Map the relative advantage into ``[0, 1]`` and minimize the configured NFT
   objective. Better candidates emphasize the positive energy; worse candidates
   emphasize the negative energy. ``nft_clip_ratio`` can limit the update
   relative to the old velocity.

``nft_tau`` controls the rollout/reference weights. A scalar ``1.0`` is
on-policy. A schedule ``[start_tau, end_tau, start_step, end_step]`` linearly
anneals the EMA update; for example, ``[1.0, 0.01, 0, 70]`` transitions from the
current policy to a slowly updated reference over the first 70 updates.

.. list-table:: Important Diffusion-NFT options
   :header-rows: 1
   :widths: 28 24 48

   * - Option
     - Current recipes
     - Effect
   * - ``algorithm.group_size``
     - ``24`` for images, ``8`` for video
     - Number of candidates compared for each prompt.
   * - ``nft_beta``
     - ``0.1``
     - Scale of the positive/negative velocity perturbation.
   * - ``nft_target_space``
     - ``x0``
     - Space in which candidate prediction errors are measured.
   * - ``nft_weight_mode``
     - ``adaptive``
     - Normalizes energy using the detached mean absolute prediction error.
   * - ``nft_clip_ratio``
     - ``0.1`` for Wan2.2
     - Bounds the velocity update relative to the rollout/reference velocity.
   * - ``nft_tau``
     - Model-specific annealing schedule
     - Controls how quickly rollout/reference weights follow the actor.
   * - ``actor.model.*.use_lora``
     - ``True``
     - Trains parameter-efficient adapters instead of all model weights.

Expected Results
----------------

.. figure:: https://github.com/user-attachments/assets/6161b286-8df9-41c3-945e-cf30ffd9f185
   :align: center
   :width: 92%

   Reference ``env/avg`` reward curves for SD3 image OCR, Wan2.2 image OCR,
   and Wan2.2 video OCR training.

In the reference runs above, all three recipes improve from a low initial OCR
reward toward approximately ``0.8``. Wan2.2 image and video runs rise more
quickly, while the SD3 run reaches a similar range later. Treat this figure as a
training-shape reference rather than a guaranteed final score: seed, prompts,
model checkpoint, GPU count, dependency versions, and evaluation settings can
change both speed and final reward.

A healthy run should show improving ``env/avg`` together with bounded NFT update
statistics. Inspect generated samples as well as scalar reward—OCR can verify
text matching, but it does not by itself measure visual quality or temporal
consistency.

.. list-table:: Metrics to monitor
   :header-rows: 1
   :widths: 28 72

   * - Metric
     - Interpretation
   * - ``env/avg``
     - Reward selected by ``reward.key``; the primary task metric in these examples.
   * - ``actor/nft_loss``
     - Optimized NFT objective. Interpret it together with reward, not as a standalone quality score.
   * - ``actor/nft_tau``
     - Current rollout/reference EMA coefficient after schedule interpolation.
   * - ``actor/delta_v_norm``
     - Magnitude of the velocity update relative to the rollout/reference prediction.
   * - ``actor/clip_frac`` / ``actor/clip_loss_frac``
     - Fraction of updates affected by NFT clipping; sustained saturation suggests an overly aggressive update.
   * - ``actor/E_pos_mean`` / ``actor/E_neg_mean``
     - Positive and negative candidate energies used to build the NFT objective.

Open TensorBoard on the generated log directory. If media saving is enabled in
``env.*.video_cfg``, generated samples are written under ``video_base_dir``.

.. code:: bash

   tensorboard --host 0.0.0.0 --logdir logs/

For scalar definitions and logger layout, see
:doc:`Training Metrics </rst_source/reference/metrics>`.
