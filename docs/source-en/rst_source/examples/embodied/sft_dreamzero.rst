DreamZero SFT
=============

.. figure:: https://dreamzero0.github.io/images/project_overview.png
   :align: center
   :width: 90%

   DreamZero: a video-generation world model fine-tuned into a VLA policy.

Run DreamZero supervised fine-tuning (SFT) in RLinf — from model and data preparation
through configuration, training, evaluation, and troubleshooting — then deploy the trained
policy on a real Franka robot.

Overview
--------

Fine-tune a WAN-based DreamZero world model into a manipulation policy on LeRobot data, evaluate it in simulation, and deploy it on a Franka.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      WAN2.1 · WAN2.2

   .. grid-item-card:: Methods
      :text-align: center

      SFT · Mixture SFT

   .. grid-item-card:: Data
      :text-align: center

      LIBERO · DROID · Franka PnP

   .. grid-item-card:: Hardware
      :text-align: center

      1+ nodes · NVIDIA CUDA · :ref:`Huawei Ascend CANN <sft-dreamzero-hardware>` (SFT)

| **You'll do:** install → prepare model + LeRobot data → generate ``metadata.json`` → launch ``run_vla_sft.sh`` → evaluate in sim or on Franka.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · the `DreamZero repo <https://github.com/RLinf/dreamzero>`_ (``DREAMZERO_PATH``) · a LeRobot dataset.

**Supported setups**

- **Datasets:** LIBERO (``libero_sim``), OXE DROID (``oxe_droid``), Franka pick-and-place (``franka_pnp``); **mixture SFT** across embodiments (see ``libero_franka_mix_sft_dreamzero_5b.yaml``).
- **Backbones:** WAN2.1 (e.g. DreamZero-DROID 14B), WAN2.2 (e.g. Wan2.2-TI2V-5B cold start).

.. _sft-dreamzero-hardware:

Setup and Run
-------------

.. include:: _setup_common.rst

Reference SFT configs under ``examples/sft/config/``:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Recipe
     - Config
     - Backbone / Checkpoint
   * - LIBERO 14B
     - ``libero_sft_dreamzero_14b.yaml``
     - WAN2.1 (`DreamZero-DROID <https://huggingface.co/GEAR-Dreams/DreamZero-DROID>`_)
   * - LIBERO 5B
     - ``libero_sft_dreamzero_5b.yaml``
     - WAN2.2 cold start (`RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000 <https://huggingface.co/RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000>`_)
   * - DROID 14B
     - ``droid_sft_dreamzero_14b.yaml``
     - WAN2.1 (`DreamZero-DROID <https://huggingface.co/GEAR-Dreams/DreamZero-DROID>`_ / `DreamZero-AgiBot <https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot>`_)
   * - LIBERO + Franka mixture
     - ``libero_franka_mix_sft_dreamzero_5b.yaml``
     - WAN2.2 mixture SFT (``dreamzero_5b``)

.. tab-set::

   .. tab-item:: NVIDIA CUDA
      :sync: cuda

      **1. Install**

      Install RLinf with DreamZero and LIBERO evaluation dependencies (omit ``--env libero`` for SFT-only runs), clone the DreamZero package alongside it, and export ``DREAMZERO_PATH`` (read by ``examples/sft/run_vla_sft.sh``):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model dreamzero --env libero && source .venv/bin/activate
         git clone https://github.com/RLinf/dreamzero.git ../dreamzero
         export DREAMZERO_PATH="$(realpath ../dreamzero)"

      **2. Download**

      Download the DreamZero-DROID 14B base checkpoint and the LIBERO LeRobot dataset, then generate ``libero_metadata.json``:

      .. code-block:: bash

         hf download GEAR-Dreams/DreamZero-DROID --local-dir ./DreamZero-DROID
         hf download physical-intelligence/libero --repo-type dataset --local-dir ./libero
         python toolkits/lerobot/generate_dreamzero_metadata.py \
           --preset libero_sim \
           --dataset-root ./libero \
           --output-metadata ./libero_metadata.json

      .. dropdown:: Other datasets and cold-start backbones (DROID, Franka PnP, WAN2.2 5B)

         **Datasets and metadata presets** (pass ``--repo-type dataset`` to ``hf download`` and run ``toolkits/lerobot/generate_dreamzero_metadata.py``):

         .. list-table::
            :header-rows: 1
            :widths: 16 36 18 30

            * - Dataset
              - Hugging Face Repo
              - Metadata Preset
              - Extra Flags / Output
            * - LIBERO
              - ``physical-intelligence/libero``
              - ``libero_sim``
              - ``--output-metadata ./libero_metadata.json``
            * - DROID
              - ``GEAR-Dreams/DreamZero-DROID-Data``
              - ``oxe_droid``
              - ``--output-metadata ./droid_metadata.json --merge``
            * - Franka PnP
              - ``RLinf/dreamzero-franka-pnp``
              - ``franka_pnp``
              - ``--output-metadata ./franka_pnp_metadata.json``

         **Cold-start from scratch (WAN2.2 5B)** — download `Wan-AI/Wan2.2-TI2V-5B <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B>`_, the `Wan2.1 CLIP <https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P>`_ weights, and `google/umt5-xxl <https://huggingface.co/google/umt5-xxl>`_:

         .. code-block:: bash

            hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
            hf download Wan-AI/Wan2.1-I2V-14B-480P models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth --local-dir ./Wan2.1-CLIP
            hf download google/umt5-xxl --local-dir ./umt5-xxl

         Set ``model_path: null`` and each ``*_pretrained_path`` in ``examples/sft/config/libero_sft_dreamzero_5b.yaml``:

         .. code-block:: yaml

            actor:
              model:
                model_path: null
                tokenizer_path: ./umt5-xxl
                diffusion_model_pretrained_path: ./Wan2.2-TI2V-5B
                image_encoder_pretrained_path: ./Wan2.1-CLIP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
                text_encoder_pretrained_path: ./Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
                vae_pretrained_path: ./Wan2.2-TI2V-5B/Wan2.2_VAE.pth
                metadata_json_path: ./libero_metadata.json
                embodiment_tag: libero_sim

      **3. Launch**

      Set the downloaded dataset, checkpoint, and metadata paths in ``examples/sft/config/libero_sft_dreamzero_14b.yaml``:

      .. code-block:: yaml

         data:
           train_data_paths: ./libero
         actor:
           model:
             model_path: ./DreamZero-DROID
             metadata_json_path: ./libero_metadata.json

      Launch SFT from the repository root (logs go to ``logs/<timestamp>-<config_name>/run_embodiment.log``; set ``runner.resume_dir`` to resume):

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh libero_sft_dreamzero_14b

      Swap the config suffix for other recipes: ``libero_sft_dreamzero_5b``, ``droid_sft_dreamzero_14b``, or ``libero_franka_mix_sft_dreamzero_5b``.

   .. tab-item:: Huawei Ascend CANN
      :sync: ascend

      Huawei Ascend CANN supports DreamZero SFT with the same checkpoints, datasets, and configurations; CI runs WAN2.2 5B SFT on two Ascend 910B NPUs with PyTorch and ``torch-npu`` 2.6.0.

      **1. Install**

      Start with the Ascend LIBERO container or a host with CANN and the NPU driver installed:

      .. include:: _ascend_libero.rst

      Clone RLinf, install the Ascend DreamZero environment, and clone DreamZero (the installer selects the matching ``torch-npu`` package and skips CUDA flash-attention):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh --platform ascend embodied --model dreamzero && source .venv/bin/activate
         git clone https://github.com/RLinf/dreamzero.git ../dreamzero
         export DREAMZERO_PATH="$(realpath ../dreamzero)"

      **2. Download**

      Download the model weights, LeRobot dataset, and generate ``metadata.json`` as shown in the **NVIDIA CUDA** tab (for WAN2.2 5B cold start, expand the cold-start dropdown there).

      **3. Launch**

      On Ascend the installer removes ``torchcodec`` if its wheel cannot load, so set ``data.video_backend: decord`` in the SFT config. On 64 GB NPUs, lower ``actor.micro_batch_size`` (sized for 80 GB GPUs by default) and keep ``actor.global_batch_size`` divisible by ``micro_batch_size`` times the number of NPUs. Launch SFT from the repository root:

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh libero_sft_dreamzero_5b

      When an actor or rollout worker runs on an NPU, RLinf redirects CUDA calls inside the DreamZero package to NPU and runs compiled kernels eagerly across that worker process.

.. dropdown:: Advanced configuration (optional)

   Configs are managed by Hydra (entry script ``examples/sft/train_vla_sft.py``) and already ship with working defaults for sampling, temporal alignment, precision, and FSDP2.

   **Data-related settings**

   .. list-table::
      :header-rows: 1
      :widths: 28 72

      * - Field
        - Meaning and role
      * - ``train_data_paths``
        - **Single dataset**: LeRobot root or HF ``repo_id``. **Mixture SFT**: a YAML list; each item sets ``dataset_path`` (or a list of paths), ``weight``, ``embodiment_tag``, ``metadata_json_path``, etc. Optional ``distribute_weights: true`` splits weight by episode length when one spec lists multiple paths.
      * - ``lazy_load``
        - Lazy-load mp4 videos. **Must be ``True`` for ``multi_anchor`` sampling** (otherwise anchor-based frame lookup fails).
      * - ``sampling_mode``
        - ``multi_anchor`` (default, recommended): sample multiple temporal anchors within the same language span; macro block count comes from ``max_chunk_size``. ``fixed_window`` is a contiguous fixed window.
      * - ``video_backend``
        - LeRobot video decoder: ``pyav``, ``torchcodec`` or ``decord``. **``torchcodec`` is recommended**; use ``decord`` where ``torchcodec`` cannot load, such as Ascend.
      * - ``video_tolerance_s``
        - Timestamp tolerance (seconds) between video frames and target times.
      * - ``parquet_cache_size``
        - Max cached parquet episodes; affects memory and I/O.
      * - ``num_workers`` / ``prefetch_factor``
        - DataLoader parallelism and prefetch; affects throughput.

   **Temporal alignment (data sampling vs model blocks)**

   - Macro temporal block count: ``actor.model.action_head_cfg.config.diffusion_model_cfg.max_chunk_size`` (commonly **4**; official Groot DROID recipes may use **5**).
   - ``actor.model.action_horizon``: per-block action steps in the WAN transform (LIBERO 16, DROID 24), not the dataset macro stride.
   - Under ``multi_anchor``, dataset action length is roughly ``action_horizon * max_chunk_size`` (e.g. LIBERO 64, DROID 96).
   - Set video time dimension via ``action_head_cfg.config.num_frames`` in presets (DreamZero default 33 = ``8 * max_chunk_size + 1``); auto-derived if omitted.

   **Model and training settings**

   .. list-table::
      :header-rows: 1
      :widths: 28 72

      * - Field
        - Meaning and role
      * - ``model_type``
        - Must be ``dreamzero``.
      * - ``model_path``
        - Full checkpoint directory; when non-null, architecture loads from ``config.json`` and weights from that path. ``null`` uses YAML/preset + ``*_pretrained_path`` for cold start.
      * - ``tokenizer_path``
        - UMT5 tokenizer path (required for training and collate).
      * - ``diffusion_model_pretrained_path``
        - Causal DiT pretrained weights; required for cold start.
      * - ``image_encoder_pretrained_path``
        - WAN image encoder; WAN2.2 must point to **WAN2.1 CLIP** weights.
      * - ``text_encoder_pretrained_path``
        - T5 text encoder weights.
      * - ``vae_pretrained_path``
        - VAE weights; WAN2.2 uses the WAN2.2 VAE checkpoint.
      * - ``metadata_json_path``
        - Dataset ``metadata.json``; falls back to ``model_path/experiment_cfg/metadata.json`` if unset.
      * - ``embodiment_tag``
        - Selects data transform and collate template: ``libero_sim``, ``oxe_droid``, ``franka_pnp``. For mixture SFT, each list entry in ``train_data_paths`` sets its own tag while ``actor.model.embodiment_tag`` stays set to the primary source.
      * - ``action_horizon`` / ``state_horizon`` / ``num_action_per_block``
        - Per-block action steps (LIBERO 16, DROID 24), state rows per sample (usually 1), and DiT ``num_action_per_block``.
      * - ``max_action_dim`` / ``max_state_dim`` / ``max_seq_len``
        - Padding limits and max text sequence length.
      * - ``target_video_height`` / ``target_video_width``
        - WAN policy head target resolution **after multi-view concat** (5B preset e.g. 176×320; LIBERO often 160×320). Model-internal resize only; **do not** use for per-view data transform resize.
      * - ``droid_view_height`` / ``droid_view_width``
        - Optional per-view resize overrides for DROID.
      * - ``relative_action`` / ``relative_action_keys`` / ``relative_action_per_horizon``
        - Relative action settings; DROID often uses ``relative_action: True`` on keys like ``joint_position``.

   - ``precision``: keep ``precision: fp32`` with ``actor.fsdp_config.mixed_precision`` (``param_dtype``, ``reduce_dtype``, ``buffer_dtype: bf16``) so optimizer states and master weights stay in FP32 while FSDP computes in BF16.
   - ``is_lora``: LoRA fine-tuning (DreamZero SFT examples typically use full fine-tuning ``False``).
   - ``actor.micro_batch_size`` / ``actor.global_batch_size``, ``actor.optim.*``, ``actor.fsdp_config``: batch sizes, optimizer schedule, and FSDP2 sharding/recompute.

   **Example config sketch**

   .. code-block:: yaml

      # Single dataset
      data:
        train_data_paths: ./libero
        lazy_load: True
        sampling_mode: multi_anchor
        video_backend: torchcodec
        num_workers: 8

      # Mixture (see libero_franka_mix_sft_dreamzero_5b.yaml)
      data:
        train_data_paths:
          - dataset_path: ./libero
            weight: 4
            embodiment_tag: libero_sim
            metadata_json_path: ./libero_metadata.json
          - dataset_path: ./franka_pnp
            weight: 1
            embodiment_tag: franka_pnp
            metadata_json_path: ./franka_pnp_metadata.json

      # Model (resume from checkpoint)
      actor:
        model:
          model_path: ./DreamZero-DROID
          tokenizer_path: ./umt5-xxl
          embodiment_tag: oxe_droid
          action_horizon: 24
          metadata_json_path: ./droid_metadata.json

.. dropdown:: Practical recommendations

   - For stable convergence, prefer continuing SFT from released DreamZero weights (set ``model_path``).
   - Full WAN2.2 adaptation via cold start needs more data and longer training; after config changes, run 50–200 steps to validate shapes and loss.
   - Regenerate or update ``metadata.json`` whenever you change datasets or ``embodiment_tag``.
   - Do not mix LIBERO and DROID config templates; ``action_horizon``, ``embodiment_tag``, and multi-view concat logic differ.

.. dropdown:: Standalone evaluation and Franka deployment

   Use the unified Evaluation section for standalone simulator or real-robot evaluation. This page records only the DreamZero-specific handoff points:

   .. list-table::
      :header-rows: 1
      :widths: 26 34 40

      * - Target
        - Start from
        - DreamZero-specific fields
      * - LIBERO simulation
        - :doc:`LIBERO evaluation guide <../../evaluations/guides/libero>` with ``evaluations/libero/libero_spatial_dreamzero_eval.yaml``
        - Set ``runner.ckpt_path`` to ``full_weights.pt``; keep ``actor.model.metadata_json_path`` and ``actor.model.embodiment_tag: libero_sim`` aligned with SFT.
      * - Franka deployment / evaluation
        - :doc:`real-world evaluation guide <../../evaluations/guides/realworld>` with ``evaluations/realworld/realworld_pnp_eval_dreamzero.yaml``
        - Set the full DreamZero checkpoint directory, ``embodiment_tag: franka_pnp``, robot IP, camera serials, and task pose fields.

   For command syntax, Hydra overrides, logs, and result files, see the :doc:`Evaluation CLI reference <../../evaluations/reference/cli>` and :doc:`Evaluation results reference <../../evaluations/reference/results>`. If your SFT checkpoint is still sharded as ``.distcp``, convert it first with the :doc:`checkpoint conversion guide <../../guides/convertor>`.

   .. note::

      ``max_steps_per_rollout_epoch`` must be divisible by ``actor.model.num_action_chunks`` for DreamZero rollout evaluation.

.. dropdown:: Extend DreamZero to a new embodiment tag

   To train DreamZero SFT on another robot or LeRobot dataset, add an embodiment transform and register the tag in RLinf:

   1. **Implement and register the embodiment transform**: Mirror the built-in ``libero_sim`` (two views), ``franka_pnp`` (two views with custom ``num_frames``), or ``oxe_droid`` (three views with ``meta/modality.json`` slicing) transforms under RLinf's DreamZero data-transform package. Define the tag string, embodiment projector ID (must match ``action_loss_embodiment_ids`` in checkpoint ``config.json`` when resuming; 5B preset includes 17, 21, 26), default ``action_horizon``, modality keys (``video.<short_name>``, ``state.<name>``, ``action.<name>``, ``language``), multi-view concatenation layout matching the T5 prompt prefix, and rollout observation mapping, then register the tag in the DreamZero embodiment enum and registry.
   2. **Generate ``metadata.json``**: Add a preset in ``toolkits/lerobot/generate_dreamzero_metadata.py`` (or pass ``--embodiment-tag``, ``--state-key``, ``--action-key``, ``--video-keys``, ``--use-modality-json`` directly):

      .. code-block:: bash

         python toolkits/lerobot/generate_dreamzero_metadata.py \
           --preset YOUR_TAG \
           --dataset-root ./your_lerobot \
           --output-metadata ./your_metadata.json

   3. **Author the training config**: Copy ``libero_sft_dreamzero_14b.yaml``, ``libero_sft_dreamzero_5b.yaml``, or ``droid_sft_dreamzero_14b.yaml`` and set the dataset path, ``embodiment_tag``, ``metadata_json_path``, and ``action_horizon`` (for WAN cold start, also add the projector ID to ``action_head_cfg.config.action_loss_embodiment_ids`` in ``examples/sft/config/model/dreamzero_5b.yaml`` or ``dreamzero_14b.yaml``):

      .. code-block:: yaml

         data:
           train_data_paths: ./your_lerobot
           lazy_load: True              # required for multi_anchor with mp4
           sampling_mode: multi_anchor

         actor:
           model:
             embodiment_tag: "YOUR_TAG"
             metadata_json_path: ./your_metadata.json
             action_horizon: 16
             target_video_height: 160
             target_video_width: 320
             relative_action: False
             relative_action_keys: []

   4. **Validate with a 50–200 step run**: Verify ``metadata.json[<your_tag>]`` statistics match parquet dimensions, confirm finite ``train/action_loss``, and check batch concatenated ``images`` shape and projector ID.

   **Pitfall checklist**

   - Keep the ``embodiment_tag`` string identical across the embodiment enum, transform tag, YAML config / ``train_data_paths`` entry, and ``metadata.json`` top-level key.
   - Set ``data.lazy_load: True`` whenever ``sampling_mode: multi_anchor`` is used with mp4 data.
   - Dataset action length is ``action_horizon × max_chunk_size``; update both consistently.
   - Keep multi-view concatenation order aligned with the prompt text, and keep post-concat spatial shape (H×W) identical across sub-datasets during mixture training.
   - Do not change projector IDs arbitrarily when fine-tuning released weights.
   - Set the matching ``embodiment_tag`` in evaluation configs under ``examples/embodiment/config/`` or ``evaluations/``.

Visualization and Results
-------------------------

1. Inspect ``run_embodiment.log``: stable ``time/step``; reasonable ``train/loss``, ``train/action_loss``, ``train/dynamics_loss``.

2. Launch TensorBoard:

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

3. Check early in the run:

   - ``images`` / ``state`` / ``action`` shapes, dtypes, value ranges
   - Valid ratios for ``state_mask`` / ``action_mask`` / ``text_attention_mask``
   - For WAN2.2: input resolution and ``frame_seqlen`` match ``config.json`` or the preset

**Pretrained checkpoint evaluation results**

Evaluation on LIBERO Spatial for `RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000 <https://huggingface.co/RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000>`_ (``num_trajectory=512``):

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Training step
     - success_once
   * - 3000
     - 7.81%
   * - 6000
     - 66.41%
   * - 9000
     - 89.06%
   * - 12000
     - 88.48%
   * - 15000
     - 66.60%
   * - 18000
     - 96.68%
   * - 21000
     - 90.43%

.. dropdown:: Training acceleration benchmarks (8×H100)

   RLinf replaces DeepSpeed ZeRO-offload with FSDP2 ``full_shard``, operator fusion, CUDA Graph, and recompute tuning, reaching up to **~4×** training throughput over the official DreamZero baseline on the **Droid dataset** (3 views: left, right, wrist; 33 frames × 480 × 640) using 8×H100 GPUs.

   **DreamZero-14B** (MBS=1, GBS=8 — **2.7×** speedup over DeepSpeed baseline, **+35%** over native FSDP2):

   .. list-table::
      :header-rows: 1
      :widths: 38 22 22 18

      * - Configuration
        - Step Time
        - Throughput (samples/sec/GPU)
        - Speedup (vs. baseline)
      * - DeepSpeed ZeRO2 + Offload (official)
        - 18.0 s
        - 0.055
        - baseline
      * - FSDP2 Base (native)
        - 9.0 s
        - 0.111
        - +100% (2.0×)
      * - **RLinf optimized**
        - **6.7 s**
        - **0.150**
        - **+170% (2.7×)**

   **DreamZero-5B** (GBS=256 — throughput improves from 1.10 to 4.44 samples/sec/GPU, **4.0×** speedup):

   .. list-table::
      :header-rows: 1
      :widths: 38 22 22 18

      * - Configuration
        - Step Time
        - Throughput (samples/sec/GPU)
        - Speedup (vs. baseline)
      * - DeepSpeed ZeRO2 + Offload (official, mbs=32 × 8 GPU)
        - 30.0 s
        - 1.10
        - baseline
      * - FSDP2 Base (mbs=1 × 8 GPU)
        - 1.8 s
        - 0.56
        - -49% (constrained by small mbs and unhidden comm)
      * - **RLinf optimized (mbs=32 + recompute × 8 GPU)**
        - **7.2 s**
        - **4.44**
        - **+300% (4.0×)**

   .. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dream0acctime.jpg
      :align: center
      :width: 45%

      Speedup comparison for DreamZero 5B and 14B models

   .. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dream0accthpt.jpg
      :align: center
      :width: 45%

      Throughput improvement for DreamZero 5B and 14B models

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Symptom
     - Fix
   * - Missing weights (``No safetensors weights``)
     - Verify ``model.safetensors`` or a sharded index exists under ``model_path``. For cold start, confirm every ``*_pretrained_path`` is valid and matches the architecture.
   * - WAN2.2 dimension mismatch
     - Check ``model_path/config.json`` or the ``dreamzero_5b`` preset: ``diffusion_model_cfg`` must use ``ti2v`` with ``in_dim/out_dim=48``, the 38-channel WAN2.2 VAE, and WAN2.1 CLIP image-encoder weights.
   * - ``metadata.json`` not found
     - Run ``toolkits/lerobot/generate_dreamzero_metadata.py``, set ``actor.model.metadata_json_path``, and verify the JSON contains a top-level key matching ``embodiment_tag``.
   * - Abnormally high ``action_loss``
     - Verify normalization stats match the current dataset, check ``relative_action`` against the data, and align ``action_horizon``, ``max_chunk_size``, and ``sampling_mode``.
   * - DROID video size errors
     - Keep ``target_video_height`` / ``target_video_width`` for post-concat WAN resize only; adjust per-view DROID sizes with ``droid_view_height`` / ``droid_view_width`` or the ``oxe_droid`` transform.
   * - ``multi_anchor`` requires ``lazy_load``
     - Set ``data.lazy_load: True``.
   * - ``AttributeError: GR1_UNIFIED_SEGMENTATION`` or unknown embodiment tag
     - Register custom tags in RLinf's DreamZero embodiment enum and transform registry, and make sure the run uses RLinf's DreamZero transform chain (RLinf patches the upstream tag enum automatically on model load).
