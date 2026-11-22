OpenPI SFT
==========

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
   :align: center
   :width: 40%

   OpenPI π₀ / π₀.₅ vision-language-action models.

Run **full-parameter** or **LoRA** SFT on π₀ / π₀.₅ in RLinf. Policies use ``model_type: openpi``, a PyTorch port aligned with the JAX reference (the official OpenPI PyTorch code is not). SFT is the usual RL cold start: imitate demonstrations first, then continue with RL from a stronger prior.

For visual-history conditioning on LIBERO-Mem, follow :doc:`OpenPI Short Memory Evaluation and Training </rst_source/examples/embodied/pi0/short_memory>`. For dual-Franka SFT and deployment, see :doc:`dual_franka_openpi_pytorch`.

Overview
--------

Full-parameter or LoRA SFT for OpenPI π₀ / π₀.₅ across LIBERO, real-world Franka, BEHAVIOR-1K, RoboTwin, and custom LeRobot datasets.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀ · π₀.₅

   .. grid-item-card:: Methods
      :text-align: center

      Full SFT · LoRA

   .. grid-item-card:: Data
      :text-align: center

      LeRobot · BEHAVIOR

   .. grid-item-card:: Hardware
      :text-align: center

      1+ nodes · GPUs

| **You'll do:** install → prepare data and norm stats → ``bash examples/sft/run_vla_sft.sh <launch-name>`` → watch the training loss.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · a matching dataset and base checkpoint.

Configurations
~~~~~~~~~~~~~~

Each experiment YAML imports a path-free model template through Hydra ``defaults``, then overrides paths and task fields. Replace ``/path/to/...`` before launch.

.. list-table::
   :header-rows: 1
   :widths: 18 38 22 22

   * - Task
     - Experiment config (``examples/sft/config/``)
     - Model template
     - Launch name
   * - LIBERO π₀
     - ``libero_sft_openpi.yaml``
     - ``model/pi0``
     - ``libero_sft_openpi``
   * - Real-world Franka bin-relocation
     - ``realworld_bin_relocation_sft_openpi.yaml``
     - ``model/pi0``
     - ``realworld_bin_relocation_sft_openpi``
   * - Custom LeRobot
     - ``custom_sft_openpi.yaml``
     - ``model/pi0``
     - ``custom_sft_openpi``
   * - BEHAVIOR-1K π₀.₅
     - ``behavior_sft_openpi_pi05.yaml``
     - ``model/pi0_5``
     - ``behavior_sft_openpi_pi05``
   * - RoboTwin adjust_bottle π₀
     - ``robotwin_adjust_bottle_sft_openpi.yaml``
     - ``model/pi0``
     - ``robotwin_adjust_bottle_sft_openpi``

Setup and Run
-------------

.. include:: _setup_common.rst

``--model openpi`` installs the official OpenPI runtime (transforms and data pipeline) alongside RLinf's ``openpi`` network.

.. tab-set::

   .. tab-item:: LIBERO & Franka
      :sync: libero-franka

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate

      **2. Configure**

      For newly collected real-world or custom LeRobot datasets, compute ``norm_stats.json`` first (see *Normalization statistics for new LeRobot datasets* below). Then set ``data.train_data_paths`` and ``actor.model.model_path`` in ``examples/sft/config/libero_sft_openpi.yaml`` (or ``realworld_bin_relocation_sft_openpi.yaml``):

      .. code-block:: yaml

         data:
           train_data_paths: /path/to/libero-data

         actor:
           model:
             model_path: /path/to/pi0-model

      **3. Launch**

      Launch training from the repository root (logs and checkpoints go under ``runner.logger.log_path`` every ``runner.save_interval`` steps at ``.../checkpoints/global_step_<N>/``):

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh libero_sft_openpi
         # Real-world Franka:
         bash examples/sft/run_vla_sft.sh realworld_bin_relocation_sft_openpi

      .. dropdown:: Advanced configuration (optional)

         Optional placement, LoRA, and OpenPI overrides (already set with working defaults in each shipped YAML):

         .. code-block:: yaml

            cluster:
              num_nodes: 1
              component_placement:
                actor,env,rollout: 0-0

            actor:
              model:
                num_action_chunks: 4          # env-executed length, not network horizon
                is_lora: False                # set True and lora_rank for LoRA
                openpi:
                  task: sft
                  embed_dtype: bf16
                  config_name: "pi0_libero"   # use pi0_realworld for Franka

   .. tab-item:: BEHAVIOR-1K
      :sync: behavior

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env behavior && source .venv/bin/activate

      **2. Download and Convert**

      Download the π₀.₅ JAX base checkpoint from ``gs://openpi-assets/checkpoints/pi05_base`` and convert it to the RLinf OpenPI layout (using BEHAVIOR's 32-step horizon; prepare task-specific ``norm_stats.json`` first; the converter defaults to π₀.₅ so omit ``--no-pi05``):

      .. code-block:: bash

         mkdir -p ./checkpoints/jax
         gcloud storage cp --recursive \
           gs://openpi-assets/checkpoints/pi05_base ./checkpoints/jax/

         python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
           --input-model ./checkpoints/jax/pi05_base \
           --input-norm-stats ./assets/behavior-1k/2025-challenge-demos/norm_stats.json \
           --output-model ./pi05_base_openpi \
           --output-norm-stats ./pi05_base_openpi/task/norm_stats.json \
           --action-dim 32 \
           --action-horizon 32 \
           --max-token-len 200

      Verify ``./pi05_base_openpi`` contains ``model.safetensors`` and ``config.json``. Other π₀.₅ SFT configs use the same converter with their own ``--action-horizon`` (10 for ManiSkill RLT; 20 for Franka RLT and dual-Franka). Official OpenPI PyTorch checkpoints with ``paligemma_with_expert.*`` keys require ``--mode openpi_pytorch_to_openpi`` even when they contain ``model.safetensors``.

      Set dataset, base model, and stats paths in ``examples/sft/config/behavior_sft_openpi_pi05.yaml``:

      .. code-block:: yaml

         data:
           train_data_paths: /path/to/2025-challenge-demos
           behavior_dataset_root: /path/to/2025-challenge-demos

         actor:
           model:
             model_path: ./pi05_base_openpi
             openpi_data:
               norm_stats_path: /path/to/assets/behavior-1k/2025-challenge-demos/norm_stats.json

      **3. Launch**

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh behavior_sft_openpi_pi05

      .. dropdown:: Advanced configuration (optional)

         The shipped config imports ``model/pi0_5`` and ``hybrid_engines/fsdp`` and sets BEHAVIOR streaming options:

         .. code-block:: yaml

            data:
              repo_id: "behavior-1k/2025-challenge-demos"
              modalities: ["rgb"]
              num_workers: 8
              tasks: ["turning_on_radio"]
              use_skill: false
              task_subtasks:
                turning_on_radio:
                  - "move to radio"
                  - "pick up radio from coffee table"
                  - "press radio"
                  - "place radio on coffee table"

            actor:
              model:
                openpi:
                  task: sft
                  embed_dtype: bf16

         - ``use_skill: false`` trains on main-task text; ``true`` uses per-frame skill text from ``task_subtasks`` (collapsed orchestrators in the dataset are not valid labels).
         - ``fine_grained_level`` / ``tolerance_s`` control streaming time alignment.

   .. tab-item:: RoboTwin
      :sync: robotwin

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env robotwin && source .venv/bin/activate

      **2. Download**

      Download the processed `adjust_bottle <https://huggingface.co/datasets/RLinf/RoboTwin-adjust_bottle-official-demo_clean50-Pi0_processed-data>`_ dataset (and optionally the reference SFT weights `RLinf-Pi0-NEW-RoboTwin-SFT-adjust_bottle <https://huggingface.co/RLinf/RLinf-Pi0-NEW-RoboTwin-SFT-adjust_bottle>`_):

      .. code-block:: bash

         hf download RLinf/RoboTwin-adjust_bottle-official-demo_clean50-Pi0_processed-data --repo-type dataset --local-dir ./robotwin-adjust-bottle-data
         hf download RLinf/RLinf-Pi0-NEW-RoboTwin-SFT-adjust_bottle --local-dir ./RLinf-Pi0-NEW-RoboTwin-SFT-adjust_bottle

      Set the dataset and base model paths in ``examples/sft/config/robotwin_adjust_bottle_sft_openpi.yaml``:

      .. code-block:: yaml

         data:
           train_data_paths: ./robotwin-adjust-bottle-data

         actor:
           model:
             model_path: /path/to/pi0_base_pytorch_new

      **3. Launch**

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh robotwin_adjust_bottle_sft_openpi

      .. dropdown:: Advanced configuration (optional)

         Uses 14-D ALOHA actions and three cameras (OpenPI pads actions to 32-D before the model). Update the task name in ``norm_stats_path`` when switching tasks:

         .. code-block:: yaml

            data:
              num_workers: 4

            actor:
              model:
                num_action_chunks: 50
                action_dim: 14
                openpi:
                  task: sft
                  embed_dtype: bf16
                  config_name: "pi0_aloha_robotwin"
                  num_images_in_input: 3
                openpi_data:
                  norm_stats_path: ${actor.model.model_path}/physical-intelligence/robotwin/adjust_bottle/norm_stats.json

   .. tab-item:: Custom LeRobot Dataset
      :sync: custom-lerobot

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate

      **2. Prepare Dataset and Dataconfig**

      Place your LeRobot dataset directory (containing ``meta/info.json`` and ``data/``) on the training host with observation keys (``image``, ``state``), ``actions``, and task text (``task`` or ``prompt``). ``examples/sft/config/custom_sft_openpi.yaml`` selects ``actor.model.openpi.config_name: "pi0_custom"`` by default; adapt that custom dataconfig (or register your own entry in the OpenPI dataconfig registry) to match your camera keys, state/action dimensions, and delta or absolute action convention.

      Compute ``norm_stats.json`` for the dataset using the same ``--config-name`` as training:

      .. code-block:: bash

         python toolkits/lerobot/calculate_norm_stats.py \
           --config-name pi0_custom \
           --repo-id ./custom-lerobot-data

      The command writes ``./assets/pi0_custom/custom-lerobot-data/norm_stats.json``.

      **3. Launch**

      Pass the dataset, base checkpoint, and normalization stats paths as Hydra CLI overrides:

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh custom_sft_openpi \
           data.train_data_paths=./custom-lerobot-data \
           actor.model.model_path=./pi0_base_openpi \
           actor.model.openpi_data.norm_stats_path=./assets/pi0_custom/custom-lerobot-data/norm_stats.json

      .. dropdown:: Advanced configuration (optional)

         Key fields in ``examples/sft/config/custom_sft_openpi.yaml``:

         .. code-block:: yaml

            data:
              train_data_paths: ./custom-lerobot-data

            actor:
              model:
                model_path: ./pi0_base_openpi
                num_action_chunks: 4
                openpi:
                  task: sft
                  config_name: "pi0_custom"
                openpi_data:
                  norm_stats_path: ./assets/pi0_custom/custom-lerobot-data/norm_stats.json

.. dropdown:: Shared conventions (precision, FSDP, action horizon)

   .. list-table::
      :header-rows: 1
      :widths: 26 74

      * - Topic
        - Convention
      * - Task mode
        - Set ``openpi.task: sft`` during training and ``eval`` during evaluation.
      * - Action chunk vs. horizon
        - ``num_action_chunks`` (and ``openpi.action_chunk``) sets env-executed steps per call; ``openpi.action_horizon`` sets network horizon (**32** for ``pi05_behavior``, **50** for ``pi0_aloha_robotwin`` and ``pi0_libero``, **10** for ``pi0_realworld``).
      * - Data window sizing
        - Standard LeRobot SFT sizes its window from the dataconfig horizon; BEHAVIOR and dual-Franka streaming loaders use ``num_action_chunks``, so keep it equal to the network horizon there.
      * - Precision
        - Set ``actor.model.precision: fp32`` for FP32 master weights and ``actor.model.openpi.embed_dtype: bf16`` for BF16 backbone activations (omitted ``embed_dtype`` follows ``model.precision``; ``precision: null`` selects BF16).
      * - FSDP1 & memory
        - Use ``hybrid_engines/openpi_fsdp`` to keep projections, RMSNorm, vision embeddings, and RLT Stage 1 weights in FP32 while the backbone computes in BF16. OpenPI supports ``no_shard``, ``shard_grad_op``, and ``full_shard`` with ``use_orig_params: true``; enable ``gradient_checkpointing: true`` with ``gradient_checkpointing_use_reentrant: false`` to reduce activation memory. BEHAVIOR and RoboTwin use the ``openpi_cosine`` LR schedule.
      * - Tokenizer
        - Loaded automatically from the base model; no separate SentencePiece path is needed.

   .. code-block:: yaml

      defaults:
        - model/pi0_5@actor.model
        - hybrid_engines/openpi_fsdp@actor.fsdp_config
        - _self_

      actor:
        model:
          precision: fp32
          openpi:
            task: sft
            embed_dtype: bf16
        fsdp_config:
          sharding_strategy: no_shard
          use_orig_params: true
          mixed_precision:
            param_dtype: bf16
            reduce_dtype: fp32
            buffer_dtype: fp32

.. dropdown:: Supported datasets & custom LeRobot config

   Select the data format with ``actor.model.openpi.config_name``:

   .. list-table::
      :header-rows: 1
      :widths: 44 56

      * - ``config_name``
        - Dataset / environment
      * - ``pi0_libero`` · ``pi05_libero``
        - LIBERO
      * - ``pi0_realworld``
        - Real-world Franka
      * - ``pi05_yam_joint``
        - Real-world Dual YAM
      * - ``pi0_aloha_robotwin``
        - RoboTwin (ALOHA)
      * - ``pi05_behavior``
        - BEHAVIOR-1K (streaming loader)
      * - ``pi0_maniskill`` · ``pi05_maniskill``
        - ManiSkill
      * - ``pi05_metaworld``
        - MetaWorld
      * - ``pi05_calvin``
        - CALVIN
      * - ``pi0_custom``
        - Custom LeRobot

   **Custom LeRobot dataset:** set ``actor.model.openpi.config_name: "pi0_custom"`` in ``examples/sft/config/custom_sft_openpi.yaml``, then register ``pi0_custom`` in the OpenPI dataconfig registry and adapt the custom Franka/LeRobot data config class to your observation keys and action space.

.. dropdown:: Normalization statistics for new LeRobot datasets

   For a newly collected LeRobot dataset, compute ``norm_stats`` for ``state`` and ``actions`` before SFT (especially important for real-robot data):

   .. code-block:: bash

      # Local dataset directory (contains meta/info.json):
      python toolkits/lerobot/calculate_norm_stats.py \
          --config-name pi0_realworld \
          --repo-id ./realworld_franka_bin_relocation

      # Or a Hugging Face repo id (cached under ~/.cache/huggingface/lerobot):
      python toolkits/lerobot/calculate_norm_stats.py \
          --config-name pi0_realworld \
          --repo-id realworld_franka_bin_relocation

   - ``--repo-id`` accepts a local path or LeRobot HF repo id; ``HF_LEROBOT_HOME`` changes the cache directory.
   - ``--config-name`` must match the training dataconfig.
   - The script writes to ``./assets/<config_name>/<repo_id>/norm_stats.json``. Training and evaluation do **not** look beside SFT ``full_weights.pt`` for this file, so set ``openpi_data.norm_stats_path`` explicitly (omitting it falls back to ``{assets_dir}/{asset_id}/norm_stats.json`` with a warning; RoboTwin pins stats beside base weights at ``physical-intelligence/robotwin/<task>/norm_stats.json``).
   - If standard deviations are tiny or the q99–q01 range is very narrow, widening them stabilizes training, especially before online RL.
   - BEHAVIOR does not use this script; point ``actor.model.openpi_data.norm_stats_path`` at an existing ``{assets}/{asset_id}/norm_stats.json``.

.. dropdown:: Evaluating SFT checkpoints

   ``openpi`` evaluation loads ``.../checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt`` (or the ``global_step_<N>`` directory) directly without converting to ``model.safetensors`` first. Always keep ``openpi.task: eval`` in eval YAMLs, match ``config_name`` and ``num_action_chunks`` to training, and explicitly point ``openpi_data.norm_stats_path`` back at the training stats file:

   .. code-block:: bash

      bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_eval \
        rollout.model.model_path=./logs/checkpoints/global_step_30000 \
        rollout.model.openpi_data.norm_stats_path=./pi0_base_pytorch_new/physical-intelligence/robotwin/adjust_bottle/norm_stats.json

   See :doc:`BEHAVIOR-1K evaluation <../../evaluations/guides/behavior>` and :doc:`RoboTwin evaluation <../../evaluations/guides/robotwin>`. To export a standalone ``model.safetensors`` + ``config.json`` + ``norm_stats.json`` directory, run the OpenPI checkpoint converter in ``sft_to_openpi`` mode (see :doc:`Checkpoint conversion <../../guides/convertor>`).

Visualization and Results
-------------------------

Watch the **training loss** in TensorBoard (``tensorboard --logdir ./logs --port 6006``). For all logged metrics and logger backends, see :doc:`Training metrics <../../reference/metrics>`.
