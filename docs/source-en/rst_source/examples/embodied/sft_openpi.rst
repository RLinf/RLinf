OpenPI Supervised Fine-Tuning
=============================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
   :align: center
   :width: 40%

   OpenPI π₀ / π₀.₅ vision-language-action models.

Run **full-parameter** or **LoRA** SFT on π₀ / π₀.₅ in RLinf. Policies use
``model_type: openpi``, a PyTorch port aligned with the JAX reference
(the official OpenPI PyTorch code is not). SFT is the usual RL cold start:
imitate demonstrations first, then continue with RL from a stronger prior.

Recipes
-------

Each experiment YAML imports a path-free model template through Hydra
``defaults``, then overrides paths and task fields. Replace ``/path/to/...``
before launch.

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

For dual-Franka SFT and deployment, see :doc:`dual_franka_openpi_pytorch`.

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
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · a matching dataset.

Shared conventions
------------------

``openpi.task`` must be ``sft`` during training. Eval YAMLs use ``eval``;
do not copy the training value.

``num_action_chunks`` is the env-executed chunk, **not** the network horizon:

- ``num_action_chunks`` / ``openpi.action_chunk``: how many actions the env
  executes per step.
- Network ``action_horizon``: ``openpi.action_horizon`` when set; otherwise the
  official ``TrainConfig.model.action_horizon`` for ``openpi.config_name``.
  Default templates only interpolate ``num_action_chunks`` into
  ``action_chunk``.

Official LeRobot SFT also sizes its data window from
``TrainConfig.model.action_horizon``; it does not read ``num_action_chunks``.
Typical official horizons: BEHAVIOR ``pi05_behavior`` **32**, RoboTwin
``pi0_aloha_robotwin`` / LIBERO ``pi0_libero`` **50**, real-world
``pi0_realworld`` **10**. Override ``openpi.action_horizon`` only when the
checkpoint horizon differs from that ``TrainConfig``.

The BEHAVIOR / dual-Franka streaming SFT loaders currently still use
``num_action_chunks`` as the data window, so those recipes keep it equal to
the official horizon.

Precision and FSDP:

- ``actor.model.precision: null`` (OpenPI default: Gemma / SigLIP in bf16,
  action head in fp32).
- ``actor.fsdp_config.sharding_strategy`` **must** be ``no_shard``
  (``full_shard`` / ``shard_grad_op`` fail at startup). Each rank holds full
  parameters, gradients, and optimizer state.
- Bind FSDP dtypes to ``actor.model.precision``. BEHAVIOR / RoboTwin recipes
  also set ``gradient_checkpointing: True`` and
  ``actor.optim.lr_scheduler: openpi_cosine`` (warmup from
  ``peak / (warmup + 1)``, then cosine decay to ``min_lr``).

.. code:: yaml

   actor:
     model:
       precision: null
       openpi:
         task: sft
     fsdp_config:
       sharding_strategy: no_shard
       mixed_precision:
         param_dtype: ${actor.model.precision}
         reduce_dtype: ${actor.model.precision}
         buffer_dtype: ${actor.model.precision}

The tokenizer is loaded by OpenPI's ``ModelTransformFactory`` from the base
model; the SFT YAML does not need a SentencePiece path.

Datasets
--------

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

Custom LeRobot dataset
~~~~~~~~~~~~~~~~~~~~~~

1. Set ``config_name`` in ``examples/sft/config/custom_sft_openpi.yaml``:

   .. code:: yaml

      actor:
        model:
          openpi:
            task: sft
            config_name: "pi0_custom"

2. Register ``pi0_custom`` in
   ``rlinf/models/embodiment/openpi/dataconfig/__init__.py`` (a template
   is already there):

   .. code:: python

      TrainConfig(
          name="pi0_custom",
          model=pi0_config.Pi0Config(),
          data=CustomDataConfig(
              repo_id="physical-intelligence/custom_dataset",
              base_config=DataConfig(prompt_from_task=True),
              assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_base/assets"),
              extra_delta_transform=False,
              action_train_with_rotation_6d=False,
          ),
          pytorch_weight_path="checkpoints/torch/pi0_base",
      ),

3. Adapt ``CustomDataConfig`` in
   ``rlinf/models/embodiment/openpi/dataconfig/franka_dataconfig.py`` to
   your keys and action space.

Normalization statistics
~~~~~~~~~~~~~~~~~~~~~~~~

For a newly collected LeRobot dataset, compute ``norm_stats`` for ``state``
and ``actions`` before SFT (especially important for real-robot data):

.. code:: bash

   # Local dataset directory (contains meta/info.json):
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi0_realworld \
       --repo-id /path/to/realworld_franka_bin_relocation

   # Or a Hugging Face repo id (cached under ~/.cache/huggingface/lerobot):
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi0_realworld \
       --repo-id realworld_franka_bin_relocation

- ``--repo-id`` is a local path or a LeRobot HF repo id; ``HF_LEROBOT_HOME``
  changes the cache parent.
- ``--config-name`` must match the training dataconfig.
- ``calculate_norm_stats.py`` writes to
  ``./assets/<config_name>/<repo_id>/norm_stats.json``
  (``TrainConfig.assets_dirs / repo_id``). Training and eval do **not** look
  next to SFT ``full_weights.pt`` for that file. Set
  ``openpi_data.norm_stats_path`` to that ``norm_stats.json``. If the key is
  omitted, OpenPI loads ``{assets_dir}/{asset_id}/norm_stats.json`` from the
  TrainConfig (after ``model_path`` is applied) and logs a warning. The
  RoboTwin recipe pins stats beside the base weights at
  ``physical-intelligence/robotwin/<task>/norm_stats.json``.

If standard deviations are tiny or the q99–q01 range is very narrow, widening
them often stabilizes training, especially when SFT is followed by online RL.

BEHAVIOR does not use that script. Point
``actor.model.openpi_data.norm_stats_path`` at an existing
``{assets}/{asset_id}/norm_stats.json``.

Installation
------------

.. include:: _setup_common.rst

**Option 1: Docker** — image ``agentic-rlinf0.4-maniskill_libero``:

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.4-maniskill_libero
      # Mainland China mirror: infinigence-ai-registry.cn-beijing.cr.aliyuncs.com/rlinf/rlinf:agentic-rlinf0.4-maniskill_libero

   source switch_env openpi

**Option 2: Custom environment**:

.. code:: bash

   # Add --use-mirror in mainland China.
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

``--model openpi`` installs the official OpenPI **runtime** (transforms and
data pipeline). The trained network is still ``openpi``.

LIBERO and real-world Franka
----------------------------

Edit data paths and ``actor.model.model_path`` in the experiment YAML.
Typical overrides:

.. code:: yaml

   data:
     train_data_paths: /path/to/libero-data

   cluster:
     num_nodes: 1
     component_placement:
       actor,env,rollout: 0-0

   actor:
     model:
       model_path: /path/to/pi0-model
       num_action_chunks: 4          # env-executed length, not network horizon
       is_lora: False                # set True and lora_rank for LoRA
       openpi:
         task: sft
         config_name: "pi0_libero"   # use pi0_realworld for Franka

Launch:

.. code:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_openpi
   bash examples/sft/run_vla_sft.sh realworld_bin_relocation_sft_openpi

Logs and checkpoints go under ``runner.logger.log_path``, saved every
``runner.save_interval`` steps at ``.../checkpoints/global_step_<N>/``.

Pi0.5 + BEHAVIOR-1K
-------------------

Filesystem paths live in the experiment config; the model shape comes from
``model/pi0_5``:

.. code:: yaml

   defaults:
     - model/pi0_5@actor.model
     - hybrid_engines/fsdp@actor.fsdp_config

The streaming loader reads only ``data:`` (no hidden defaults). Change at least:

.. code:: yaml

   data:
     train_data_paths: /path/to/2025-challenge-demos
     behavior_dataset_root: /path/to/2025-challenge-demos
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
       model_path: /path/to/pi05_base_pytorch_new   # new-format fp32 base weights
       openpi:
         task: sft
       openpi_data:
         norm_stats_path: /path/to/assets/behavior-1k/2025-challenge-demos/norm_stats.json

- ``use_skill: false`` trains on the main-task text; ``true`` uses per-frame
  skill text from ``task_subtasks`` (collapsed orchestrators in the dataset
  are not valid labels).
- ``fine_grained_level`` / ``tolerance_s`` control streaming time alignment.

.. code:: bash

   bash examples/sft/run_vla_sft.sh behavior_sft_openpi_pi05

Pi0 + RoboTwin
--------------

Uses the official OpenPI / LeRobot map-style loader with
``config_name: pi0_aloha_robotwin``. Example data:
`adjust_bottle <https://huggingface.co/datasets/RLinf/RoboTwin-adjust_bottle-official-demo_clean50-Pi0_processed-data>`_;
matching HF SFT weights:
`RLinf-Pi0-NEW-RoboTwin-SFT-adjust_bottle <https://huggingface.co/RLinf/RLinf-Pi0-NEW-RoboTwin-SFT-adjust_bottle>`_.

.. code:: yaml

   data:
     train_data_paths: /path/to/robotwin-data
     num_workers: 4

   actor:
     model:
       model_path: /path/to/pi0_base_pytorch_new
       num_action_chunks: 50
       action_dim: 14
       openpi:
         task: sft
         config_name: "pi0_aloha_robotwin"
         num_images_in_input: 3
       openpi_data:
         norm_stats_path: ${actor.model.model_path}/physical-intelligence/robotwin/adjust_bottle/norm_stats.json

14-D ALOHA actions and three cameras; OpenPI pads actions to 32-D before the
model. ``openpi_data.norm_stats_path`` pins SFT and eval to the same
``norm_stats.json``. Change the task name in that path when switching tasks.

.. code:: bash

   bash examples/sft/run_vla_sft.sh robotwin_adjust_bottle_sft_openpi

Evaluation
----------

SFT writes ``full_weights.pt`` under
``.../checkpoints/global_step_<N>/actor/model_state_dict/``. ``openpi``
eval can load that ``.pt`` (or the ``global_step_<N>`` directory) directly:
wrapper / FSDP prefixes are stripped in memory, and the eval YAML supplies the
model shape. You do **not** need to convert to ``model.safetensors`` first.

Do not copy training ``openpi.task: sft`` into eval. Eval configs must use
``task: eval`` (that selects ``Pi0Eval`` and observation transforms). Keep
``config_name`` and ``num_action_chunks`` aligned with training.

Norm stats also do **not** travel with ``full_weights.pt``. Stock eval YAMLs
often interpolate ``openpi_data.norm_stats_path`` as
``${rollout.model.model_path}/.../norm_stats.json``, which assumes a converted
directory that already bundles stats. If you only retarget ``model_path`` at
an SFT checkpoint, also point stats back at the training copy (RoboTwin: next
to the base weights at ``physical-intelligence/robotwin/<task>/norm_stats.json``;
BEHAVIOR: the training ``openpi_data.norm_stats_path``).

.. code:: bash

   # Example: evaluate the SFT .pt in place (keep training-time norm stats)
   bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_eval \
     rollout.model.model_path=/path/to/logs/.../checkpoints/global_step_30000 \
     rollout.model.openpi_data.norm_stats_path=/path/to/pi0_base_pytorch_new/physical-intelligence/robotwin/adjust_bottle/norm_stats.json

Eval steps: :doc:`BEHAVIOR-1K <../../evaluations/guides/behavior>` and
:doc:`RoboTwin <../../evaluations/guides/robotwin>`. To export a shareable
bare ``Pi0`` directory (``model.safetensors`` + ``config.json`` + copied
``norm_stats.json``), use ``sft_to_openpi``; commands are in
``rlinf/utils/ckpt_convertor/openpi/README.md``.

Visualization
-------------

Watch the **training loss**. Metric definitions:
:doc:`Training metrics <../../reference/metrics>`.

.. code:: bash

   tensorboard --logdir ./logs
