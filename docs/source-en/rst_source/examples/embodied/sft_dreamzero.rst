DreamZero Supervised Fine-Tuning and Franka Real-World Deployment
=================================================================

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

      1+ nodes · GPUs

| **You'll do:** install → prepare model + LeRobot data → generate ``metadata.json`` → launch ``run_vla_sft.sh`` → evaluate in sim or on Franka.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · the `DreamZero repo <https://github.com/RLinf/dreamzero>`_ (``DREAMZERO_PATH``) · a LeRobot dataset.

**Currently supported**

- **Datasets:** LIBERO (``libero_sim``), OXE DROID (``oxe_droid``), Franka pick-and-place (``franka_pnp``); **mixture SFT** across embodiments (see ``libero_franka_mix_sft_dreamzero_5b.yaml``).
- **Backbones:** WAN2.1 (e.g. DreamZero-DROID 14B), WAN2.2 (e.g. Wan2.2-TI2V-5B cold start).

Installation
------------

.. include:: _setup_common.rst

**Option 1: SFT-only environment** — install DreamZero without simulator dependencies:

.. code-block:: bash

   # Add --use-mirror for faster downloads in mainland China.
   bash requirements/install.sh embodied --model dreamzero
   source .venv/bin/activate

**Option 2: SFT + LIBERO evaluation** — add LIBERO simulator dependencies:

.. code-block:: bash

   bash requirements/install.sh embodied --model dreamzero --env libero
   source .venv/bin/activate

Clone the DreamZero repository separately and set ``DREAMZERO_PATH`` before SFT or eval:

.. code-block:: bash

   git clone https://github.com/RLinf/dreamzero.git
   export DREAMZERO_PATH=/path/to/dreamzero

**What this does:**

1. Creates a DreamZero-specific uv virtual environment through ``requirements/install.sh``.
2. Installs only offline SFT dependencies by default, or adds LIBERO when you need simulator evaluation.
3. Makes the external DreamZero package importable through ``DREAMZERO_PATH``; ``examples/sft/run_vla_sft.sh`` also appends it to ``PYTHONPATH``.

Model Preparation
-----------------

Resume from a Checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~

Set ``actor.model.model_path`` to a downloaded checkpoint directory; architecture and weights load from that path. Options:

- DreamZero 14B (DROID / AgiBot): `DreamZero-DROID <https://huggingface.co/GEAR-Dreams/DreamZero-DROID>`_, `DreamZero-AgiBot <https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot>`_ — see ``droid_sft_dreamzero_14b.yaml``
- RLinf 5B (LIBERO SFT): `RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000 <https://huggingface.co/RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000>`_ — see ``libero_sft_dreamzero_5b.yaml`` and point ``model_path`` at that directory

Download example:

.. code:: bash

   pip install -U huggingface_hub
   huggingface-cli download GEAR-Dreams/DreamZero-DROID --local-dir ./DreamZero-DROID


YAML example (DROID + official 14B; see ``droid_sft_dreamzero_14b.yaml``):

.. code:: yaml

   defaults:
     - model/dreamzero_14b@actor.model

   actor:
     model:
       model_path: ./DreamZero-DROID
       tokenizer_path: google/umt5-xxl
       embodiment_tag: oxe_droid

For AgiBot data, set ``model_path`` to ``./DreamZero-AgiBot`` instead.

Train from Scratch (WAN2.2 Component Cold Start)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``model_path: null`` and fill each ``*_pretrained_path``. Download from Hugging Face:

- `Wan-AI/Wan2.2-TI2V-5B <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B>`_ (DiT, T5, VAE)
- `Wan2.1 CLIP <https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P>`_ file ``models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`` (not in the 5B repo)
- `google/umt5-xxl <https://huggingface.co/google/umt5-xxl>`_

Download example:

.. code:: bash

   huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
   huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
     models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth --local-dir ./Wan2.1-CLIP
   huggingface-cli download google/umt5-xxl --local-dir ./umt5-xxl

YAML example (LIBERO cold start; see ``libero_sft_dreamzero_5b.yaml``):

.. code:: yaml

   defaults:
     - model/dreamzero_5b@actor.model

   actor:
     model:
       model_path: null
       tokenizer_path: google/umt5-xxl
       diffusion_model_pretrained_path: Wan-AI/Wan2.2-TI2V-5B
       image_encoder_pretrained_path: Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
       text_encoder_pretrained_path: Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
       vae_pretrained_path: Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
       metadata_json_path: /path/to/metadata.json
       embodiment_tag: libero_sim


Data Preparation
----------------

Training data must follow the LeRobot v2/v3 layout (``meta/``, ``data/``, etc.). Set a local path or Hugging Face dataset ID via ``data.train_data_paths``.

Download Datasets
~~~~~~~~~~~~~~~~~

Supported datasets:

- LIBERO: `physical-intelligence/libero <https://huggingface.co/datasets/physical-intelligence/libero>`_ — ``embodiment_tag: libero_sim``; see ``libero_sft_dreamzero_14b.yaml`` / ``libero_sft_dreamzero_5b.yaml``
- DROID: `GEAR-Dreams/DreamZero-DROID-Data <https://huggingface.co/datasets/GEAR-Dreams/DreamZero-DROID-Data>`_ — ``embodiment_tag: oxe_droid``; see ``droid_sft_dreamzero_14b.yaml``
- Franka PnP: `RLinf/dreamzero-franka-pnp <https://huggingface.co/datasets/RLinf/dreamzero-franka-pnp>`_ — ``embodiment_tag: franka_pnp``; transforms in ``data_transforms/franka_pnp.py`` (extends ``libero_sim`` dual-view layout)
- Mixture SFT: ``libero_franka_mix_sft_dreamzero_5b.yaml`` uses a **list** for ``data.train_data_paths``; each entry can set ``dataset_path``, ``embodiment_tag``, ``metadata_json_path``, and ``weight``

Download example:

.. code:: bash

   pip install -U huggingface_hub
   # LIBERO
   huggingface-cli download physical-intelligence/libero --repo-type dataset --local-dir ./libero
   # DROID
   huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir ./DreamZero-DROID-Data
   # Franka PnP real-world data
   huggingface-cli download RLinf/dreamzero-franka-pnp --repo-type dataset --local-dir ./franka_pnp

Generate ``metadata.json``
~~~~~~~~~~~~~~~~~~~~~~~~~~

For a new dataset or cold start (no ``experiment_cfg/metadata.json``), generate normalization stats for the corresponding ``embodiment_tag`` first:

.. code:: bash

   # LIBERO
   python toolkits/lerobot/generate_dreamzero_metadata.py \
     --preset libero_sim \
     --dataset-root /path/to/libero \
     --output-metadata /path/to/metadata.json

   # DROID (use --merge for multiple datasets)
   python toolkits/lerobot/generate_dreamzero_metadata.py \
     --preset oxe_droid \
     --dataset-root /path/to/droid \
     --output-metadata /path/to/metadata.json \
     --merge

   # Franka PnP
   python toolkits/lerobot/generate_dreamzero_metadata.py \
     --preset franka_pnp \
     --dataset-root /path/to/franka_pnp \
     --output-metadata /path/to/franka_pnp_metadata.json

Then set ``actor.model.metadata_json_path`` in config (or place the file at ``model_path/experiment_cfg/metadata.json``).


Configure Further
-----------------

Configs are managed by Hydra; the entry script is ``examples/sft/train_vla_sft.py``. Below, **data fields** and **model/training fields** are explained separately.

Data-Related Settings
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning and role
   * - ``train_data_paths``
     - **Single dataset**: LeRobot root or HF ``repo_id``. **Mixture SFT**: a YAML list; each item sets ``dataset_path`` (or a list of paths), ``weight``, ``embodiment_tag``, ``metadata_json_path``, etc. ``build_dreamzero_mixture_dataset_from_spec`` samples by weight. Optional ``distribute_weights: true`` splits weight by episode length when one spec lists multiple paths.
   * - ``lazy_load``
     - Lazy-load mp4 videos. **Must be ``True`` for ``multi_anchor`` sampling** (otherwise anchor-based frame lookup fails).
   * - ``sampling_mode``
     - ``multi_anchor`` (default, recommended): sample multiple temporal anchors within the same language span; macro block count comes from ``max_chunk_size``. ``fixed_window`` is a contiguous fixed window.
   * - ``video_backend``
     - LeRobot video decoder: ``pyav`` or ``torchcodec``; affects lazy mp4 speed and compatibility. **``torchcodec`` is recommended.**
   * - ``video_tolerance_s``
     - Timestamp tolerance (seconds) between video frames and target times.
   * - ``parquet_cache_size``
     - Max cached parquet episodes; affects memory and I/O.
   * - ``num_workers`` / ``prefetch_factor``
     - DataLoader parallelism and prefetch; affects throughput.

**Temporal alignment (data sampling vs model blocks)**

- Macro temporal block count: ``actor.model.action_head_cfg.config.diffusion_model_cfg.max_chunk_size`` (commonly **4**; official Groot DROID recipes may use **5**).
- ``actor.model.action_horizon``: **per-block action steps in DreamTransform / WAN** (LIBERO 16, DROID 24), not the dataset macro stride.
- Under ``multi_anchor``, dataset action length is roughly ``action_horizon * max_chunk_size`` (e.g. LIBERO 64, DROID 96).
- Set video time dimension via ``action_head_cfg.config.num_frames`` in presets (DreamZero default 33 = ``8 * max_chunk_size + 1``); auto-derived if omitted.

Model and Training Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Identity and weight paths**

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
     - Causal DiT (diffusion backbone) pretrained weights; required for cold start.
   * - ``image_encoder_pretrained_path``
     - WAN image encoder; WAN2.2 must point to **WAN2.1 CLIP** weights.
   * - ``text_encoder_pretrained_path``
     - T5 text encoder weights.
   * - ``vae_pretrained_path``
     - VAE weights; WAN2.2 uses ``WanVideoVAE38``.
   * - ``metadata_json_path``
     - Dataset ``metadata.json``; falls back to ``model_path/experiment_cfg/metadata.json`` if unset.
   * - ``embodiment_tag``
     - Selects data transform and collate template: ``libero_sim``, ``oxe_droid``, ``franka_pnp`` (``data_transforms/embodiment_tag.py``). For a single dataset, must match the data. For mixture SFT, each list entry in ``train_data_paths`` sets its own tag; ``actor.model.embodiment_tag`` is still required (usually aligned with the primary source for policy-side metadata in ``get_model``).

**Temporal and action shape (must align with data and WAN capacity)**

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning and role
   * - ``action_horizon``
     - Action steps per WAN temporal block (LIBERO 16, DROID 24).
   * - ``state_horizon``
     - State rows per sample (usually 1, one state per macro anchor).
   * - ``num_action_per_block``
     - Align with DiT ``num_action_per_block`` in ``action_head_cfg`` (often equals ``action_horizon``).
   * - ``action_head_cfg...diffusion_model_cfg.max_chunk_size``
     - Multi-anchor macro temporal blocks / Causal DiT capacity; tied to ``data.sampling_mode: multi_anchor``. Video ``num_frames`` is derived as ``8 * max_chunk_size + 1``.
   * - ``max_action_dim`` / ``max_state_dim`` / ``max_seq_len``
     - Padding limits and max text sequence length in DreamTransform.

**Video size and DROID-specific options**

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning and role
   * - ``target_video_height`` / ``target_video_width``
     - WAN policy head target resolution **after multi-view concat** (5B preset e.g. 176×320; Libero often 160×320). Model-internal resize only; **do not** use for per-view data transform resize.
   * - ``droid_view_height`` / ``droid_view_width``
     - (Optional) per-view resize overrides for DROID.
   * - ``relative_action`` / ``relative_action_keys`` / ``relative_action_per_horizon``
     - Relative action settings; DROID often uses ``relative_action: True`` on keys like ``joint_position``.

**Other training options**

- ``precision``: main precision for Actor/optimizer (``fp32`` / ``bf16``). **Recommended: ``fp32``** with ``actor.fsdp_config.mixed_precision`` for mixed precision: ``precision: fp32`` keeps **optimizer states and master weights in FP32** (more stable), while FSDP runs forward/backward matmuls in **BF16** via ``mixed_precision`` (saves memory, faster). Example:

  .. code:: yaml

     actor:
       model:
         precision: fp32
       fsdp_config:
         mixed_precision:
           param_dtype: bf16
           reduce_dtype: bf16
           buffer_dtype: bf16

  Setting ``precision: bf16`` also lowers optimizer state precision and is usually less stable. With FSDP **CPU offload**, keep ``precision: fp32``.
- ``is_lora``: LoRA fine-tuning (DreamZero SFT examples typically use full fine-tuning ``False``).
- ``actor.micro_batch_size`` / ``actor.global_batch_size``: per-GPU micro-batch and global effective batch size.
- ``actor.optim.*``: learning rate, warmup, cosine schedule, etc.
- ``actor.fsdp_config``: FSDP2 sharding, gradient checkpointing; ``mixed_precision`` controls compute/comm dtypes (works with ``actor.model.precision`` above).

**Example config sketch**

.. code:: yaml

   # ---------- data (single dataset) ----------
   data:
     train_data_paths: /path/to/libero
     lazy_load: True
     sampling_mode: multi_anchor
     video_backend: torchcodec
     num_workers: 8

   # ---------- data (mixture; see libero_franka_mix_sft_dreamzero_5b.yaml) ----------
   data:
     train_data_paths:
       - dataset_path: /path/to/libero
         weight: 4
         embodiment_tag: libero_sim
         metadata_json_path: /path/to/libero_metadata.json
       - dataset_path: /path/to/franka_pnp
         weight: 1
         embodiment_tag: franka_pnp
         metadata_json_path: /path/to/franka_metadata.json

   # ---------- model (resume from checkpoint) ----------
   actor:
     model:
       model_path: /path/to/DreamZero-DROID
       tokenizer_path: /path/to/umt5-xxl
       embodiment_tag: oxe_droid
       action_horizon: 24
       metadata_json_path: /path/to/metadata.json   # if no experiment_cfg/metadata.json


Run It
------

From the repository root:

.. code:: bash

   # LIBERO + WAN2.1 (checkpoint, dreamzero_14b preset)
   bash examples/sft/run_vla_sft.sh libero_sft_dreamzero_14b

   # LIBERO + WAN2.2 (cold start, dreamzero_5b preset)
   bash examples/sft/run_vla_sft.sh libero_sft_dreamzero_5b

   # DROID + WAN2.1 (dreamzero_14b preset; model_path -> DreamZero-DROID)
   bash examples/sft/run_vla_sft.sh droid_sft_dreamzero_14b

   # LIBERO + Franka mixture (WAN2.2; see libero_franka_mix_sft_dreamzero_5b.yaml)
   bash examples/sft/run_vla_sft.sh libero_franka_mix_sft_dreamzero_5b

Equivalent command:

.. code:: bash

   python examples/sft/train_vla_sft.py \
     --config-path examples/sft/config/ \
     --config-name <config_name> \
     runner.logger.log_path=<auto_log_dir>

Logs:

- ``<repo>/logs/<timestamp>-<config_name>/run_embodiment.log``

Resume training with ``runner.resume_dir`` pointing to a checkpoint directory (field provided in example configs such as ``droid_sft_dreamzero_14b.yaml`` and ``libero_sft_dreamzero_5b.yaml``).


Standalone Evaluation
---------------------

Use the unified Evaluation section for standalone simulator or real-robot evaluation.
This SFT page only records the DreamZero-specific handoff points.

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

For command syntax, Hydra overrides, logs, and result files, use the
:doc:`Evaluation CLI reference <../../evaluations/reference/cli>` and
:doc:`Evaluation results reference <../../evaluations/reference/results>`. If your SFT
checkpoint is still sharded as ``.distcp``, convert it first with the
:doc:`checkpoint conversion guide <../../guides/convertor>`.

.. note::

   ``max_steps_per_rollout_epoch`` must be divisible by ``actor.model.num_action_chunks``
   for DreamZero rollout evaluation.

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

Visualization and Results
-------------------------

1. Inspect ``run_embodiment.log``: stable ``time/step``; reasonable ``train/loss``, ``train/action_loss``, ``train/dynamics_loss``.

2. TensorBoard:

.. code:: bash

   tensorboard --logdir ./logs --port 6006

3. Check early in the run:

   - ``images`` / ``state`` / ``action`` shapes, dtypes, value ranges
   - Valid ratios for ``state_mask`` / ``action_mask`` / ``text_attention_mask``
   - For WAN2.2: input resolution and ``frame_seqlen`` match ``config.json`` or the preset


Extend DreamZero to a New ``embodiment_tag``
--------------------------------------------

To train DreamZero SFT on a **new robot or LeRobot dataset**, add an ``embodiment_tag`` and register the corresponding transforms and metadata tooling in RLinf. Use existing modules as templates:

- ``rlinf/data/datasets/dreamzero/data_transforms/libero_sim.py`` (two views, simple state/action columns)
- ``rlinf/data/datasets/dreamzero/data_transforms/franka_pnp.py`` (two views, extends ``libero_sim``, custom ``num_frames``, etc.)
- ``rlinf/data/datasets/dreamzero/data_transforms/oxe_droid.py`` (three views, ``meta/modality.json`` slicing)

Data flow:

.. code:: text

   LeRobot dataset
        → DreamZeroLeRobotDataset (reads parquet/mp4 via transform keys)
        → ComposedModalityTransform + DreamTransform (normalize, multi-view concat, tokenize)
        → DreamZeroCollator → training

Step 1: Implement the Embodiment Transform Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new transform module under ``rlinf/data/datasets/dreamzero/data_transforms/`` named for your tag, for example ``your_tag.py``. Implement ``DreamZeroEmbodimentTransform`` (see ``base.py``), including at least:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Member / method
     - Description
   * - ``TAG``
     - String id; must **exactly match** ``actor.model.embodiment_tag`` and the top-level key in ``metadata.json``.
   * - ``DEFAULT_TAG_MAPPING``
     - ``{TAG: <int>}`` maps to the WAN action head **embodiment projector ID**. When fine-tuning released DreamZero weights, the ID must appear in ``action_loss_embodiment_ids`` in checkpoint ``config.json`` (5B preset includes 17, 21, 26). A **new ID** implies random projector init or model config changes.
   * - ``DEFAULT_ACTION_HORIZON``
     - Default per-block action steps (LIBERO 16, DROID 24); align with ``actor.model.action_horizon``.
   * - ``get_modality_config()``
     - Returns ``ModalityConfig`` for ``video`` / ``state`` / ``action`` / ``language`` (``delta_indices``, ``modality_keys``). ``language`` keys must exist in the dataset. Video/action ``delta_indices`` should match Groot recipes (current code uses video ``range(25)``, action ``range(24)``); mismatch breaks ``multi_anchor`` alignment.
   * - ``get_transform(...)``
     - Build ``Video*`` → ``StateAction*`` → ``ConcatTransform`` → ``DreamTransform``; RLinf's ``DreamTransform`` (``dream_transform.py``) calls the registry for multi-view concat.
   * - ``format_training_prompt(instruction)``
     - T5 prompt prefix describing the multi-view layout (consistent with Groot training templates).
   * - ``concat_multiview_video(images)``
     - Concatenate ``(v, t, c, h, w)`` to ``(1, t, c, H, W)``; layout must match ``format_training_prompt``.
   * - ``ROLLOUT_OBS_LAYOUT``
     - A ``RolloutObsLayout`` mapping RLinf rollout fields (``main_images``, ``wrist_images``, ``states``, ``task_descriptions``) to the ``modality_keys`` above. Used at inference via ``convert_rollout_env_obs(embodiment_tag, env_obs)`` in ``data_transforms/__init__.py``.

**``modality_keys`` naming** (wired to ``DreamZeroLeRobotDataset``):

- Video: ``video.<short_name>`` (e.g. ``video.image``); short names resolve via ``meta/modality.json`` ``original_key`` or ``info.json`` ``observation.images.*`` / bare column names.
- State/action: ``state.<name>``, ``action.<name>``; with ``meta/modality.json``, use ``start``/``end`` slices; otherwise fallback to full ``observation.state`` / ``action`` columns or heuristics (see ``_build_component_sources`` in ``lerobot_dataset.py``).
- Keys in training YAML must match ``*_concat_order`` in ``ConcatTransform``.

Step 2: Register in RLinf
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Add a member to ``EmbodimentTag`` in ``rlinf/data/datasets/dreamzero/data_transforms/embodiment_tag.py`` (value must equal your ``TAG`` string).
2. Edit ``rlinf/data/datasets/dreamzero/data_transforms/__init__.py``:

   - ``from ...<your_tag> import YourEmbodimentDataTransform``
   - Add ``YourEmbodimentDataTransform.TAG: YourEmbodimentDataTransform`` to ``_EMBODIMENT_REGISTRY``

No manual Groot patch is required: ``get_model()`` replaces ``groot.vla.data.schema.embodiment_tags.EmbodimentTag`` with the RLinf enum via ``Patcher.add_patch``.

If unregistered, ``build_dreamzero_composed_transform`` errors and lists known tags.

Step 3: Generate ``metadata.json``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute normalization stats; the output key must equal ``TAG``:

**Option A (recommended)**: add an entry to ``PRESETS`` in ``toolkits/lerobot/generate_dreamzero_metadata.py`` (mirror ``libero_sim`` / ``oxe_droid``: ``state_key``, ``action_key``, ``video_keys``, ``use_modality_json``), then:

.. code:: bash

   python toolkits/lerobot/generate_dreamzero_metadata.py \
     --preset <your_tag> \
     --dataset-root /path/to/lerobot_dataset \
     --output-metadata /path/to/metadata.json

**Option B**: use CLI flags without editing the script (``--embodiment-tag``, ``--state-key``, ``--action-key``, ``--video-keys``, ``--use-modality-json``).

Set ``actor.model.metadata_json_path`` in training config (or ``model_path/experiment_cfg/metadata.json``).

Step 4: Author / Adjust the Training Config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Copy ``libero_sft_dreamzero_14b.yaml``, ``libero_sft_dreamzero_5b.yaml``, or ``droid_sft_dreamzero_14b.yaml`` and update at least:

.. code:: yaml

   data:
     train_data_paths: /path/to/your_lerobot
     lazy_load: True              # required for multi_anchor with mp4
     sampling_mode: multi_anchor

   actor:
     model:
       embodiment_tag: "<your_tag>"
       metadata_json_path: /path/to/metadata.json
       action_horizon: <match DEFAULT_ACTION_HORIZON>
       # when resuming: verify action_loss_embodiment_ids includes your projector ID
       target_video_height: ...
       target_video_width: ...
       relative_action: ...
       relative_action_keys: [...]

For WAN cold start, add the new ID to ``action_head_cfg.config.action_loss_embodiment_ids`` in ``examples/sft/config/model/dreamzero_5b.yaml`` (or ``dreamzero_14b.yaml``).

Step 5: Validate with a Short Run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Run the metadata script alone; confirm ``metadata.json[<your_tag>]`` statistics/modalities match parquet dimensions.
2. Run 50–200 SFT steps; ensure no ``Could not map transform video keys`` or ``embodiment_tag not found in metadata`` errors.
3. Check finite ``train/action_loss``; verify batch ``images`` concat shape and ``embodiment_id`` vs ``DEFAULT_TAG_MAPPING``.

**Pitfall checklist**

- ``embodiment_tag`` string must match in **four places**: ``embodiment_tag.py`` enum value, Python ``TAG``, config / ``train_data_paths`` entry, and **metadata.json** top-level key.
- ``multi_anchor`` + mp4 data: **must** set ``data.lazy_load: True``.
- Dataset action length is ``action_horizon × max_chunk_size``; do not change only one.
- Multi-view **concat order** must match **prompt text** or training signal is wrong.
- Do not change ``DEFAULT_TAG_MAPPING`` integer IDs arbitrarily when fine-tuning official weights.
- Per-view ``VideoResize`` lives in each embodiment's ``data_transforms`` module (e.g. ``libero_sim`` and ``franka_pnp`` both use 256×256); ``target_video_height/width`` is for WAN resize **after** multi-view concat only—do not mix the two. **Mix dataset training** requires identical post-concat ``images`` spatial shape (H×W) from ``DreamTransform`` across sub-datasets, or collate will fail; align ``VideoResize`` in the corresponding transform modules when concat layouts differ (e.g. ``oxe_droid`` uses a 2×2 grid) or per-view defaults differ.
- Inference/eval: set ``embodiment_tag`` correctly in DreamZero eval configs under ``examples/embodiment/config/``.

For inference only (no RLinf code changes) when upstream Groot/DreamZero already supports the tag, ``metadata.json`` and eval config may suffice; **SFT on new data** requires the enum member, registry entry, and transform module above (``get_model`` patches Groot ``EmbodimentTag`` automatically).


Common Issues
-------------

1. **Missing weights (No safetensors weights)**

   - Check ``model.safetensors`` or a sharded index under ``model_path``
   - For cold start, ensure all ``*_pretrained_path`` entries are valid and match the architecture

2. **WAN2.2 dimension mismatch**

   - Verify effective config (``model_path/config.json`` or ``dreamzero_5b`` preset): ``diffusion_model_cfg`` is ti2v, ``in_dim/out_dim=48``, ``vae_cfg`` is ``WanVideoVAE38``
   - Image encoder must use WAN2.1 CLIP paths

3. **metadata.json not found**

   - Run ``toolkits/lerobot/generate_dreamzero_metadata.py`` and set ``metadata_json_path``
   - Confirm JSON contains a key matching ``embodiment_tag``

4. **Abnormally high action_loss**

   - Check normalization stats match the current dataset
   - Check ``relative_action`` settings vs data
   - Align ``action_horizon``, ``max_chunk_size``, and ``sampling_mode``

5. **DROID video size errors**

   - Do not use ``target_video_height/width`` for per-view data transform resize; adjust DROID view sizes in the ``oxe_droid`` transform code

6. **multi_anchor requires lazy_load**

   - Set ``data.lazy_load: True``

7. **``AttributeError: GR1_UNIFIED_SEGMENTATION`` or unknown ``EmbodimentTag``**

   - Use ``dream_transform.DreamTransform`` (RLinf subclass) in the transform chain, not the Groot base class directly
   - Register new tags in ``embodiment_tag.py`` and ``_EMBODIMENT_REGISTRY``; ``get_model()`` patches the Groot enum at model load


Practical Recommendations
-------------------------

- For stable convergence, prefer continuing SFT from released DreamZero weights (set ``model_path``).
- Full WAN2.2 adaptation via cold start needs more data and longer training; after config changes, run 50–200 steps to validate shapes and loss.
- Regenerate or update ``metadata.json`` whenever you change datasets or ``embodiment_tag``.
- Do not mix LIBERO and DROID config templates; ``action_horizon``, ``embodiment_tag``, and multi-view concat logic differ.
