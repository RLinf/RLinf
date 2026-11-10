Diffusion Model Fine-Tuning with ART
====================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/wan.png
   :align: center
   :width: 45%

   SD3 and Wan2.2 generation models trained with DiffusionART.

Fine-tune SD3 or Wan2.2 with DiffusionART to improve text rendering using OCR
rewards. ART combines rewards and clean latent endpoints within each prompt
group into a shared direction, then trains against frozen velocity targets at
states visited during generation. Complete the :doc:`shared setup <video_models>`,
choose an image or video recipe, and inspect the training results below.

Overview
--------

Both recipes use LoRA with FSDP2 and deterministic ODE generation. They split
each prompt's candidates into two disjoint update windows that share targets
frozen before the first update. Corrective masking suppresses updates that would
move an already-drifted prediction farther outward. For the derivation and
reference implementation, see the
`DiffusionART paper <https://arxiv.org/abs/2610.12572>`_,
`official code <https://github.com/vanzll/DiffusionART>`_, and
`project page <https://vanzll.github.io/diffusionart-project/>`_.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      SD3 / Wan2.2

   .. grid-item-card:: Algorithm
      :text-align: center

      DiffusionART

   .. grid-item-card:: Output
      :text-align: center

      Image / Video

   .. grid-item-card:: Reward
      :text-align: center

      OCR / Video OCR

| **You'll do:** select a recipe → launch training → inspect OCR reward and generated samples.
| **Prerequisites:** :doc:`Shared installation, model, and dataset setup <video_models>`. The recorded Wan experiment used four NVIDIA A800 GPUs.

Tasks
~~~~~

Choose the config for the output you want to generate.

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

The environment scores each completed candidate; video rewards are averaged
before ART builds guidance for the prompt group.

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
     - One scalar OCR similarity score in ``[0, 1]`` per candidate.

Installation and Assets
-----------------------

Follow the shared :ref:`installation <diffusion-model-setup>`,
:ref:`model download <diffusion-model-checkpoints>`, and
:ref:`OCR dataset download <diffusion-ocr-dataset>` instructions. Both ART
recipes set ``env.*.reward.use_gpu: False`` and use the CPU OCR installation.

Run Training
------------

Open ``examples/diffusion/config/sd3_art_ocr.yaml`` or
``examples/diffusion/config/wan22_ti2v_5b_art_video_ocr.yaml`` before launching.
In the selected YAML, set ``actor.model.model_path``,
``env.train.dataset.path``, and ``env.eval.dataset.path`` to your downloaded
directories. Check ``cluster.num_nodes``, ``cluster.component_placement``, and
the batch settings against your hardware and the ART constraints below.

After checking the YAML, run its config name from the repository root:

.. code:: bash

   bash examples/diffusion/run_diffusion.sh sd3_art_ocr

   bash examples/diffusion/run_diffusion.sh wan22_ti2v_5b_art_video_ocr

The launcher generates and scores candidates, updates LoRA parameters, and
writes logs under ``logs/<timestamp>-<config>/``.

Configure the ART Update
------------------------

When changing rollout size or placement, each actor shard must receive
complete prompt groups.
``algorithm.art_updates_per_rollout`` must be ``1`` or ``2`` and divide
``algorithm.group_size``; ``env.train.group_size`` must match that group size.

.. warning::

   Set ``actor.global_batch_size`` to
   ``env.train.total_num_envs * env.train.rollout_epoch`` divided by
   ``algorithm.art_updates_per_rollout``. For 96 environments, 12 rollout
   epochs, and two updates, use ``actor.global_batch_size=576``. Each per-rank
   update window must be divisible by ``actor.micro_batch_size``. Environment
   counts must preserve complete groups across both env and actor ranks.

ART requires ``actor.training_backend=fsdp``, ``rollout.pipeline_stage_num=1``,
``algorithm.update_epoch=1``, and ``runner.weight_sync_interval=1``. Keep
``env.train.max_steps_per_rollout_epoch=actor.model.num_action_chunks`` so that
each rollout epoch produces one complete candidate. SD3 uses
``actor.model.sd3.train_sampler=ode`` with ``actor.model.sd3.noise_level=0.0``;
Wan records ODE trajectories.

.. list-table:: Main Update Options
   :header-rows: 1
   :widths: 40 60

   * - Option
     - Effect
   * - ``algorithm.art_timestep_fraction``
     - Fraction of denoising states used for training: ``0.5`` for SD3 and ``0.25`` for Wan in the supplied recipes.
   * - ``algorithm.art_guidance_gain``
     - Scales reward guidance; both recipes use ``1.0``.
   * - ``algorithm.art_corrective_mask_enabled``
     - Enables the corrective movement mask; both recipes use ``True``.
   * - ``algorithm.art_mask_threshold``
     - Movement threshold for the mask; both recipes use ``1e-5`` with ``art_mask_metric=sde_kl``.

.. _diffusion-art-results:

Expected Results
----------------

Watch ``env/video_ocr`` for Wan or ``env/avg`` for SD3 and inspect generated
samples. ``actor/active_group_frac`` shows whether prompt groups supply guidance;
``actor/skipped_zero_guidance`` marks rounds with no usable guidance on any rank.
Use :doc:`Training Metrics </rst_source/reference/metrics>` for logging and
TensorBoard setup. OCR reward alone does not measure visual quality or temporal
consistency, and ART and NFT losses are not directly comparable.

The recorded Wan comparison used four NVIDIA A800 GPUs, LoRA, FSDP2, and CPU
video OCR. The table reports the first crossing of a training reward of ``0.8``,
smoothed over 5760 generated trajectories, from one run per configuration.
The recorded sampling budgets below differ from the checked-in startup recipes.

.. list-table:: Recorded Wan Video OCR Runs
   :header-rows: 1
   :widths: 18 22 18 20 22

   * - Method
     - Trajectories per Round
     - Learning Rate
     - Recorded Rounds
     - First Smoothed Reward of 0.8
   * - ART
     - 1152
     - 1e-4
     - 148
     - 5.43 h / 21888 trajectories
   * - NFT
     - 1152
     - 1e-4
     - 111
     - Unreached within 29.50 h
   * - NFT
     - 576
     - 4e-4
     - 70
     - 7.83 h / 32832 trajectories

These are single-run training rewards, not held-out success rates. The NFT runs
change learning rate and sampling budget together; their different lengths also
prevent treating final rewards as equal-budget results.
