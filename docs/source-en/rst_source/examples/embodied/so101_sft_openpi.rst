SO101 Pi0 Supervised Fine-Tuning
=================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
   :align: center
   :width: 40%

   OpenPI π₀ supervised fine-tuning on SO101 teleoperation data.

This recipe runs OpenPI π₀ supervised fine-tuning on data collected from the
:doc:`SO101 6-DOF arm <so101>`. It is the second stage of the SO101 workflow:
once you have ``run_<timestamp>/`` directories from teleoperation, this guide
takes you all the way through merging them, computing normalization
statistics, and launching SFT on a GPU server.

Overview
--------

Fine-tune π₀ on a merged SO101 LeRobot dataset, on a single GPU host
disjoint from the robot.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      OpenPI π₀

   .. grid-item-card:: Methods
      :text-align: center

      Full SFT

   .. grid-item-card:: Data
      :text-align: center

      LeRobot v3.0 (multi-run)

   .. grid-item-card:: Hardware
      :text-align: center

      1+ GPUs · no robot needed

| **You'll do:** install OpenPI on the GPU host → merge per-run datasets → place the merge under HF_LEROBOT_HOME → compute norm stats → launch ``run_so101_sft_smoke.sh`` → watch the training loss.
| **Prerequisites:** :doc:`so101` (data collection complete) · :doc:`sft_openpi` (familiarity with the generic OpenPI SFT flow) · a π₀ base checkpoint.

Where the pieces live
~~~~~~~~~~~~~~~~~~~~~

The SO101 SFT pipeline is wired together by these RLinf-side files:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - File
     - Purpose
   * - ``rlinf/models/embodiment/openpi/policies/so101_policy.py``
     - ``SO101Inputs`` / ``SO101Outputs`` transforms — pad 6-D state /
       6-D action to the model dim, expose ``image`` + ``extra_view_image``
       as ``base_0_rgb`` + ``left_wrist_0_rgb``.
   * - ``rlinf/models/embodiment/openpi/dataconfig/so101_dataconfig.py``
     - ``LeRobotSO101DataConfig`` — repacks the flat-key LeRobot dataset
       (``state``, ``actions``, ``image``, ``extra_view_image``, ``prompt``)
       into the ``observation/`` tree the π₀ policy expects.
   * - ``rlinf/models/embodiment/openpi/dataconfig/__init__.py`` (entry ``pi0_so101``)
     - Registers the SO101 ``TrainConfig`` with ``action_horizon=10``,
       ``repo_id="so101_data"``, and the standard ``pi0_base`` assets dir.
   * - ``rlinf/models/embodiment/openpi/_compat.py``
     - OpenPI ↔ LeRobot compatibility shims (``lerobot.common`` alias,
       ``PromptFromLeRobotTask`` DataFrame patch).  Activated at site init
       via a ``.pth`` file dropped into ``site-packages/``.
   * - ``examples/sft/config/so101_sft_openpi.yaml``
     - SFT recipe — ``action_dim: 6``, ``num_action_chunks: 10``,
       ``openpi.config_name: pi0_so101``, ``num_images_in_input: 2``.
   * - ``examples/sft/run_so101_sft_smoke.sh``
     - Launcher with sane defaults (``HF_HUB_OFFLINE=1``,
       ``EMBODIED_PATH``, ``model_path``, ``train_data_paths``,
       ``max_steps``).
   * - ``toolkits/lerobot/merge_lerobot_datasets.py``
     - Merger that auto-detects v2.x / v3.0 layouts.

Installation (GPU host)
-----------------------

The robot side already has ``--env so101`` installed for data collection.  The
GPU host needs the OpenPI bundle plus the same RLinf checkout.  These steps
also drop the persistent compat ``.pth`` shim into the venv, so DataLoader
spawn workers see the OpenPI ↔ LeRobot adapters automatically.

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

   # Adds OpenPI + RLinf editable into a fresh .venv, installs the
   # transformers_replace patches, and drops the openpi compat .pth into
   # site-packages.  Use --use-mirror inside mainland China.
   bash requirements/install.sh embodied --model openpi --env so101
   source .venv/bin/activate

If you previously installed via a different ``--env`` and want to add OpenPI
support, the same command is idempotent — it only re-runs missing steps.

.. note::

   If the persistent shim isn't picked up automatically (e.g. you brought a
   prebuilt venv from elsewhere), run::

      python -m rlinf.models.embodiment.openpi._compat install

   This writes ``rlinf_openpi_compat.pth`` into the active site-packages, so
   :func:`install_compat_shims` fires in every Python process that uses the
   venv — including the ``multiprocessing.spawn`` workers used by torch
   DataLoader.

Data preparation
----------------

Merge per-run datasets
~~~~~~~~~~~~~~~~~~~~~~

SO101 data collection writes one ``run_<timestamp>/`` subdirectory per
recording session.  Each run is a self-contained LeRobot v3.0 dataset.
OpenPI's loader expects a single dataset, so merge them first:

.. code-block:: bash

   # Source: parent dir containing rank_0/run_<timestamp>/...
   # Output: a single merged LeRobot v3.0 dataset.
   python toolkits/lerobot/merge_lerobot_datasets.py \
       --source-dir /path/to/so101_data \
       --output-dir /path/to/so101_data_merged

The merge tool detects the codebase version automatically (``v2.x``
jsonlines or ``v3.0`` parquet) and writes the same layout back.  For SO101
v3.0 input it produces:

.. code-block::

   so101_data_merged/
   ├── data/chunk-000/file-000.parquet      # all frames concatenated
   ├── meta/
   │   ├── episodes/chunk-000/file-000.parquet
   │   ├── tasks.parquet
   │   ├── stats.json                       # carried over from first run
   │   └── info.json                        # totals patched globally

Place the merge under ``HF_LEROBOT_HOME``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The OpenPI loader looks up datasets by repo id under
``$HF_LEROBOT_HOME`` (default ``~/.cache/huggingface/lerobot``).  Symlink
the merged dataset there with the canonical name ``so101_data`` so it
matches the ``repo_id`` baked into ``pi0_so101``:

.. code-block:: bash

   mkdir -p ~/.cache/huggingface/lerobot
   ln -sfn /path/to/so101_data_merged ~/.cache/huggingface/lerobot/so101_data

You can also pass an absolute path to ``train_data_paths`` instead — but the
symlink keeps the ``norm_stats`` lookup path tidy (see next section).

Compute normalization statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-feature normalization stats live next to the model checkpoint.  The
helper script reads the dataset, runs every transform in the pi0_so101
config, and writes ``norm_stats.json`` to ``<assets_dir>/<repo_id>/``.

.. code-block:: bash

   # HF_HUB_OFFLINE=1 prevents the lerobot loader from contacting the Hub
   # to validate the dataset version.
   HF_HUB_OFFLINE=1 python toolkits/lerobot/calculate_norm_stats.py \
       --config_name pi0_so101 \
       --repo_id so101_data

For the standard 8-episode / 2011-frame collection the script processes
62 batches in roughly 50 seconds and writes stats like::

   state mean(0..5):  [53.37, 1.58, -88.92, 82.34, 40.29, 0.87]
   state std (0..5):  [7.65, 8.7, 20.86, 21.47, 15.66, 2.59]

Place the resulting JSON next to the π₀ base checkpoint so the SFT runtime
can load it from ``<model_path>/so101_data/norm_stats.json``:

.. code-block:: bash

   PI0_BASE=/path/to/pi0_base    # writable copy or fresh dir
   mkdir -p $PI0_BASE/so101_data
   cp assets/pi0_so101/so101_data/norm_stats.json \
      $PI0_BASE/so101_data/norm_stats.json

A clean way to set up ``$PI0_BASE`` without copying the 14 GB safetensors is
to make a directory of symlinks to the read-only base checkpoint:

.. code-block:: bash

   PI0_BASE_RO=/path/to/shared/pi0_base    # the original 14 GB ckpt
   PI0_BASE=/path/to/pi0_base_so101         # writable mirror
   mkdir -p $PI0_BASE
   for f in model.safetensors config.json policy_postprocessor.json policy_preprocessor.json README.md; do
       ln -sf $PI0_BASE_RO/$f $PI0_BASE/$f
   done
   # Then drop the per-asset_id norm_stats into a real subdirectory:
   mkdir -p $PI0_BASE/so101_data
   cp assets/pi0_so101/so101_data/norm_stats.json $PI0_BASE/so101_data/

.. note::

   ``q01`` / ``q99`` quantile inspection is just as relevant here as for any
   other OpenPI dataset.  See :doc:`sft_openpi` for the rationale and the
   recommended widening procedure for narrow ranges.

Run It
------

Configuration
~~~~~~~~~~~~~

The recipe lives at ``examples/sft/config/so101_sft_openpi.yaml``.  Inherit
from it via override on the CLI for ad-hoc runs, or copy it for a permanent
variant.

.. code-block:: yaml

   data:
     train_data_paths: "so101_data"   # resolves to ~/.cache/huggingface/lerobot/so101_data

   actor:
     model:
       model_path: "/path/to/pi0_base_so101"
       num_action_chunks: 10
       action_dim: 6
       openpi:
         config_name: "pi0_so101"
         num_images_in_input: 2

   runner:
     max_steps: 2000

The SO101 default ``micro_batch_size: 4`` × ``global_batch_size: 32`` matches
the realworld Franka recipe.  Bump ``cluster.component_placement.actor`` to
the GPU range you have (e.g. ``0-7``) and switch ``fsdp_config.sharding_strategy``
back to ``full_shard`` to use multiple GPUs.

.. _so101-sft-launch:

Launch
~~~~~~

Standard launcher
^^^^^^^^^^^^^^^^^

Like every other OpenPI SFT recipe, the standard ``run_vla_sft.sh`` works for
SO101:

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh so101_sft_openpi

This picks up ``EMBODIED_PATH`` from the script directory, resolves the config
from ``examples/sft/config/so101_sft_openpi.yaml``, timestamps the log
directory automatically, and forwards any Hydra overrides you append.
Override ``model_path`` and ``train_data_paths`` on the command line or in a
permanent copy of the YAML:

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh so101_sft_openpi \
       actor.model.model_path=/path/to/pi0_base_so101 \
       data.train_data_paths=so101_data

Smoke-test launcher (offline-friendly)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The convenience launcher ``run_so101_sft_smoke.sh`` wraps the same call but
**forces ``HF_HUB_OFFLINE=1``** (for GPU hosts without internet), exposes the
four most common knobs as env vars, and defaults to a 20-step smoke run so a
new install can be validated without editing any YAML:

.. code-block:: bash

   # Defaults: model_path=/root/pi0_base_so101, train_data_paths=so101_data,
   # max_steps=20, log_dir=/root/so101_sft_run.
   bash examples/sft/run_so101_sft_smoke.sh

   # Customised:
   SO101_MODEL_PATH=/abs/path/to/pi0_base_so101 \
   SO101_DATA_REPO_ID=so101_data \
   SO101_MAX_STEPS=2000 \
   SO101_LOG_DIR=/abs/path/to/results \
       bash examples/sft/run_so101_sft_smoke.sh

Smoke test (1×A800)
~~~~~~~~~~~~~~~~~~~

A 20-step single-GPU smoke run on the standard 8-episode SO101 collection
takes about 2 min 40 s and produces a real checkpoint:

.. list-table::
   :header-rows: 1
   :widths: 10 15 18 18 18

   * - Step
     - Loss
     - Grad norm
     - LR
     - Step time
   * - 13
     - 0.141
     - 2.90
     - 3.25e-7
     - 4.61 s
   * - 16
     - 0.147
     - 3.39
     - 4.00e-7
     - 4.17 s
   * - 18
     - 0.139
     - 3.66
     - 4.50e-7
     - 4.13 s
   * - 20
     - 0.174
     - 5.16
     - 5.00e-7
     - 4.11 s

Loss bounces around 0.10–0.27 with grad norms 2.9–5.2, learning rate ramps
2.5e-7 → 5.0e-7 along the 1000-step cosine warmup inherited from the
realworld recipe.  The 20-step checkpoint lands at about 31 GB under
``${SO101_LOG_DIR}/so101_sft_openpi/checkpoints/global_step_20/actor/``
(``dcp_checkpoint`` + ``model_state_dict`` flavours).

Visualisation
-------------

TensorBoard events land under ``${SO101_LOG_DIR}/tensorboard/``:

.. code-block:: bash

   tensorboard --logdir ${SO101_LOG_DIR}/tensorboard

Watch ``train/loss``, ``train/grad_norm``, ``train/learning_rate`` for SFT
health.  See :doc:`Training metrics <../../reference/metrics>` for the full
namespace.

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Problem
     - Solution
   * - ``ModuleNotFoundError: No module named 'lerobot.common'``
     - The persistent compat shim isn't installed in this venv.  Run
       ``python -m rlinf.models.embodiment.openpi._compat install`` from
       inside the activated venv.  This writes
       ``site-packages/rlinf_openpi_compat.pth`` and the next Python
       process picks both shims up automatically.
   * - ``ValueError: task_index=0 not found in task mapping``
     - The DataFrame-tasks shim didn't fire in a worker process.  Verify
       the ``.pth`` file exists in ``site-packages/`` (see above), and that
       no shell alias points the venv's Python at a different interpreter.
   * - ``transformers_replace is not installed correctly``
     - OpenPI ships replacement modules under
       ``openpi/models_pytorch/transformers_replace/``.  ``install.sh``
       copies them on top of the installed ``transformers`` package; if you
       installed by hand, run::

          cp -r /path/to/openpi/src/openpi/models_pytorch/transformers_replace/* \
                .venv/lib/python3.11/site-packages/transformers/
   * - ``UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa3``
     - macOS ``._*`` resource-fork files leaked into the ``transformers/``
       tree from a tarball you brought across.  Purge with
       ``find .venv/lib/python3.11/site-packages/transformers -name '._*' -delete``.
   * - ``HFValidationError: Repo id must be in the form 'repo_name'`` …
     - You passed an absolute path to ``--repo_id``.  Either set up the
       ``HF_LEROBOT_HOME`` symlink (recommended) or use a path with no
       slashes / preserved by the loader.
   * - Norm stats not loading at training time
     - The runtime asset_id is the original ``repo_id`` baked into the
       ``TrainConfig`` (``so101_data``).  Make sure the JSON lives at
       ``${model_path}/so101_data/norm_stats.json`` — not under the name
       you passed to ``--repo_id`` if those differ.

Cross-references
----------------

- :doc:`so101` — data collection on the SO101 hardware.
- :doc:`sft_openpi` — generic OpenPI SFT recipe (LIBERO, ManiSkill, Franka,
  etc.).
- :doc:`/rst_source/extending/new_realworld_robot` — extending RLinf to a
  new real-world robot.
- :doc:`Training metrics <../../reference/metrics>` — full metric namespace.
