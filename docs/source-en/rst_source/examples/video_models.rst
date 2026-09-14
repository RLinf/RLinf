RL for Video Generation Models
==============================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/wan.png
   :align: center
   :width: 45%

   Image and video generation models trained with Diffusion-NFT in RLinf.

RLinf integrates Diffusion-NFT for reinforcement learning of SD3 and Wan2.2.
The pipeline treats a generated image or video as the action, evaluates it in
RLinf's ``diffusion`` environment, and updates the diffusion model from the
resulting reward. The current examples optimize text rendering with an OCR
reward and support both image-level and frame-aware video training.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environment
      :text-align: center

      ``diffusion``

   .. grid-item-card:: Algorithm
      :text-align: center

      Diffusion-NFT

   .. grid-item-card:: Models
      :text-align: center

      SD3 / Wan2.2

   .. grid-item-card:: Reward
      :text-align: center

      OCR / Video OCR

| **You'll do:** install dependencies → download models and prompts → select a config → launch ``run_diffusion.sh`` → monitor reward and NFT metrics.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · an SD3 or Wan2.2 checkpoint · the OCR prompt dataset.

Supported Tasks
~~~~~~~~~~~~~~~

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

How the Diffusion Environment Works
-----------------------------------

The diffusion environment adapts media generation to RLinf's environment
interface. It is intentionally a one-step environment: a prompt is sampled at
``reset()``, the rollout worker generates media, and ``step()`` scores that
media and terminates the episode.

1. **Build a prompt group.** The dataset samples one or more prompts and repeats
   each prompt according to ``algorithm.group_size``. Candidates from the same
   prompt form one comparison group.
2. **Generate candidates.** SD3 or Wan2.2 receives the prompt and produces an
   image or video. The rollout also records the denoising state required by the
   NFT update.
3. **Score the media.** The configured reward backend returns a dictionary of
   scores. ``reward.key`` selects the scalar or frame sequence used for RL; the
   supplied OCR configs use ``avg``.
4. **Close the episode.** The environment returns the score as the reward and
   marks the one-step episode as terminated. For video rewards, frame scores can
   be grouped into environment steps using ``reward.frame_interval``.

.. list-table:: Diffusion environment data contract
   :header-rows: 1
   :widths: 22 34 44

   * - Stage
     - Value
     - Meaning
   * - Observation
     - ``task_descriptions``
     - Text prompts sampled from ``env.*.dataset``.
   * - Action
     - Image or video tensor
     - Generated media returned by the diffusion policy.
   * - Reward
     - ``scores[reward.key]``
     - Image-level score or a sequence of frame/chunk scores.
   * - Episode metrics
     - ``return``, ``avg``, backend keys
     - Values logged by RLinf for training and evaluation.

The environment is not tied to OCR. Reward and dataset implementations are
loaded from ``rlinf.envs.sim.diffusion`` and ``rlinf.data.datasets.diffusion`` using
the ``dataset.type`` and ``reward.model`` fields, so another media scorer can be
integrated behind the same one-step interface.

Reward Scoring
--------------

Image OCR Reward
~~~~~~~~~~~~~~~~

The OCR backend extracts the quoted target text from each prompt, runs
PaddleOCR on the generated image, removes spaces, and compares the recognized
string with the target using Levenshtein distance. The normalized score is

.. math::

   r_{\mathrm{OCR}} = 1 -
   \frac{\min\left(d(\hat{y}, y), |y|\right)}{|y|},

where :math:`y` is the target text and :math:`\hat{y}` is the recognized text.
A score near ``1`` means the requested text is reproduced correctly; ``0``
means the edit distance is at least the target length. An exact substring match
also receives ``1``.

Video OCR Reward
~~~~~~~~~~~~~~~~

For Wan2.2 video training, the same OCR scorer is applied to every frame. The
backend can expose either frame-level scores or a video-level mean. The checked-
in ``wan22_ti2v_5b_nft_video_ocr`` config uses frame-aware rewards, groups frames
with ``frame_interval: 4``, and computes ``grpo_video`` advantages using
``advantage_mode: video``. This preserves frame/chunk credit while normalizing
scores across the candidates generated for the same prompt.

When adding another reward, keep its scale and direction explicit: higher values
must represent better samples, and candidates within a prompt group should have
enough score variation to produce a useful relative advantage.

Diffusion-NFT Update
--------------------

The checked-in recipes combine grouped reward normalization with an NFT update
in diffusion velocity space:

1. For each prompt, generate ``group_size`` candidates and compute relative
   advantages. Image recipes use standard GRPO normalization within each prompt
   group. The video recipe normalizes frame/chunk rewards over the whole video
   group.
2. Sample a denoising step and reconstruct the current noisy state
   (``nft_xcur_source: resample``). The rollout/reference model supplies the old
   velocity :math:`v_{old}` and the trainable model predicts :math:`v_\theta`.
3. Form :math:`\Delta v = v_\theta - v_{old}` and the positive/negative
   candidates :math:`v_{\pm} = v_{old} \pm \beta\Delta v`.
4. Convert both candidates to the configured target space. The provided configs
   use ``nft_target_space: x0`` and compare the predicted clean sample against
   the rollout target with weighted squared-error energies
   :math:`E_{+}` and :math:`E_{-}`.
5. Map the relative advantage into ``[0, 1]`` and minimize the configured NFT
   objective. Better candidates emphasize :math:`E_{+}`; worse candidates
   emphasize :math:`E_{-}`. ``nft_clip_ratio`` can limit the update relative to
   the old velocity.

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
     - Bounds the velocity update relative to :math:`v_{old}`.
   * - ``nft_tau``
     - Model-specific annealing schedule
     - Controls how quickly rollout/reference weights follow the actor.
   * - ``actor.model.*.use_lora``
     - ``True``
     - Trains parameter-efficient adapters instead of all model weights.

Installation
------------

.. include:: embodied/_setup_common.rst

**Custom Environment**

.. code:: bash

   # Add --use-mirror for faster downloads in mainland China.
   bash requirements/install.sh embodied --model diffusion
   source .venv/bin/activate

To use a custom virtual environment directory, pass ``--venv <dir>``:

.. code:: bash

   bash requirements/install.sh embodied --model diffusion --venv /path/to/venv
   source /path/to/venv/bin/activate

This command creates a Python 3.10 environment and installs the SD3, Wan2.2,
and OCR reward dependencies, including Diffusers, PEFT, Transformers,
PaddleOCR, and PaddlePaddle.

.. warning::

   The checked-in diffusion configs use ``/path/to/...`` placeholders.
   Replace the model and dataset paths following the comments in the configs,
   or override ``actor.model.model_path`` and ``env.*.dataset.path`` at launch.

.. note::

   Wan2.2 requires a Diffusers version that supports
   ``Wan-AI/Wan2.2-TI2V-5B-Diffusers``. Use the installer above instead of an
   older embodied environment that already pins Diffusers for another model.

Download the Model
------------------

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

Stable Diffusion 3.5 Medium requires accepting the model access terms on
Hugging Face before downloading.

Download the Dataset
--------------------

The examples use the OCR prompt dataset from
`NVlabs/DiffusionNFT <https://github.com/NVlabs/DiffusionNFT/tree/main/dataset/ocr>`__.
The dataset directory must contain ``train.txt`` and ``test.txt``; each line is
one prompt, and ``env.*.dataset.split`` selects the file to read.

.. code:: bash

   mkdir -p /path/to/dataset/ocr
   curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/train.txt \
      -o /path/to/dataset/ocr/train.txt
   curl -L https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/ocr/test.txt \
      -o /path/to/dataset/ocr/test.txt

Run Training
------------

Pass a Diffusion-NFT config name to ``run_diffusion.sh`` and override the local
model and dataset paths.

.. code:: bash

   bash examples/diffusion/run_diffusion.sh sd3_nft_ocr \
      actor.model.model_path=/path/to/stable-diffusion-3.5-medium \
      env.train.dataset.path=/path/to/dataset/ocr \
      env.eval.dataset.path=/path/to/dataset/ocr

.. code:: bash

   bash examples/diffusion/run_diffusion.sh wan22_ti2v_5b_nft_video_ocr \
      actor.model.model_path=/path/to/Wan2.2-TI2V-5B-Diffusers \
      env.train.dataset.path=/path/to/dataset/ocr \
      env.eval.dataset.path=/path/to/dataset/ocr

Configure further in ``examples/diffusion/config/*.yaml``.

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
