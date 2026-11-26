ABot-M0 RL
==========

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/ABot-M0.png
   :align: center
   :width: 80%

   ABot-M0: a VGGT-grounded VLA policy.

Run evaluation and **PPO** training for
`ABot-M0 <https://github.com/amap-cvlab/ABot-Manipulation>`__ in RLinf, on
standard **LIBERO** and **LIBERO-Plus**. The integration uses the HuggingFace
rollout backend and FSDP actor training: ABot-M0 generates action chunks during
rollout, and RLinf recomputes log-probabilities and value estimates from the
stored rollout inputs during actor updates.

Overview
--------

Fine-tune ABot-M0 on LIBERO-10 / LIBERO-Plus with PPO (actor-critic).

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      LIBERO · LIBERO-Plus

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      LIBERO-10

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · GPUs

| **You'll do:** install → download the ABot-M0 checkpoint + backbones → set ``model_path`` → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · an ABot-M0 LIBERO checkpoint and its backbone weights (steps below).

ABot-M0 is the VLA policy: the RLinf wrapper keeps pretrained perception
components frozen, trains the action model through the RL objective, and adds a
value head for actor-critic PPO (GAE advantages/returns, ratio clipping, value
clipping, optional entropy regularization).

Tasks
~~~~~

Select the model page by matching the environment, task family, and config or checkpoint artifact.

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - Environment
     - Task / Suite
     - Config / Weights
     - Focus
   * - LIBERO
     - LIBERO-10
     - ``libero_10_ppo_abot_m0``
     - PPO fine-tuning for the ABot-M0 release checkpoint.
   * - LIBERO
     - LIBERO-10+
     - ``libero_10_plus_ppo_abot_m0``
     - Long-horizon LIBERO-10+ training with ABot-M0.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - Field
     - Description
   * - Observation
     - LIBERO RGB observations and robot state expected by ABot-M0.
   * - Action
     - Continuous robot actions decoded from ABot-M0 policy outputs.
   * - Reward
     - LIBERO success signal or task reward used by PPO.
   * - Prompt
     - Natural-language instruction associated with each LIBERO task.

Setup and Run
-------------

.. include:: _setup_common.rst

The installer clones ABot-M0 and VGGT automatically; set ``ABOT_PATH`` and ``VGGT_PATH`` first to reuse local checkouts.

.. tab-set::

   .. tab-item:: LIBERO
      :sync: libero

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model abot_m0 --env maniskill_libero && source .venv/bin/activate

      **2. Download**

      Download the 10k-step LIBERO RL baseline (~40% initial LIBERO success rate) and the Qwen3-VL backbone:

      .. code-block:: bash

         hf download HaoyunOvO/ABot-m0-LIBERO-10k-step --local-dir ./ABot-m0-LIBERO-10k-step
         hf download StarVLA/Qwen3-VL-4B-Instruct-Action --local-dir ./Qwen3-VL-4B-Instruct-Action

      Update ``qwenvl.base_vlm`` in the downloaded checkpoint's ``config.yaml`` to your local ``Qwen3-VL-4B-Instruct-Action`` path:

      .. code-block:: yaml

         qwenvl:
           base_vlm: /path/to/Qwen3-VL-4B-Instruct-Action

      **3. Launch**

      Launch PPO training, passing ``actor.model.model_path`` on the command line (rollout inherits the same path automatically):

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_10_ppo_abot_m0 \
           actor.model.model_path=./ABot-m0-LIBERO-10k-step/checkpoints/steps_10000_pytorch_model.pt

      .. dropdown:: Advanced configuration (optional)

         - **Released SFT checkpoint for evaluation**: download ``acvlab/ABot-M0-LIBERO`` instead of the 10k-step RL baseline:

           .. code-block:: bash

              hf download acvlab/ABot-M0-LIBERO --local-dir ./ABot-M0-LIBERO

         - **Offline VGGT weights**: for machines without Hugging Face access at runtime, download ``facebook/VGGT-1B`` and export ``VGGT_MODEL_PATH`` (``VGGT_PATH`` selects the source checkout at install time; ``VGGT_MODEL_PATH`` selects the weights at runtime):

           .. code-block:: bash

              hf download facebook/VGGT-1B --local-dir ./VGGT-1B
              export VGGT_MODEL_PATH=$PWD/VGGT-1B

         - **Hydra configuration reference**: see :doc:`Training configuration <../../reference/configuration>`.

   .. tab-item:: LIBERO-Plus
      :sync: liberoplus

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model abot_m0 --env liberoplus && source .venv/bin/activate

      **2. Download**

      Download the LIBERO-Plus scene assets, the 10k-step LIBERO RL baseline, and the Qwen3-VL backbone (see the :ref:`LIBERO-Pro & LIBERO-Plus section <liberopro-plus-benchmark>` for asset details):

      .. code-block:: bash

         export LIBERO_PLUS_PACKAGE_DIR=$(python -c "import pathlib; import liberoplus.liberoplus as l_plus; print(pathlib.Path(l_plus.__file__).resolve().parent)" | tail -n 1)
         hf download --repo-type dataset Sylvest/LIBERO-plus assets.zip --local-dir "${LIBERO_PLUS_PACKAGE_DIR}"
         hf download HaoyunOvO/ABot-m0-LIBERO-10k-step --local-dir ./ABot-m0-LIBERO-10k-step
         hf download StarVLA/Qwen3-VL-4B-Instruct-Action --local-dir ./Qwen3-VL-4B-Instruct-Action

      Extract the ``assets/`` subtree from ``assets.zip`` into the installed ``liberoplus.liberoplus`` package directory:

      .. code-block:: bash

         python - <<'PY'
         import os, zipfile
         from pathlib import Path
         pkg = Path(os.environ["LIBERO_PLUS_PACKAGE_DIR"])
         with zipfile.ZipFile(pkg / "assets.zip") as z:
             for info in z.infolist():
                 if "/assets/" in info.filename and (rel := info.filename.split("/assets/", 1)[1]):
                     target = pkg / "assets" / rel
                     target.mkdir(parents=True, exist_ok=True) if info.is_dir() else (target.parent.mkdir(parents=True, exist_ok=True), target.write_bytes(z.read(info)))
         PY

      Update ``qwenvl.base_vlm`` in the downloaded checkpoint's ``config.yaml`` to your local ``Qwen3-VL-4B-Instruct-Action`` path:

      .. code-block:: yaml

         qwenvl:
           base_vlm: ./Qwen3-VL-4B-Instruct-Action

      **3. Launch**

      Select the LIBERO-Plus suite and launch PPO training, passing ``actor.model.model_path`` on the command line (rollout inherits the same path automatically):

      .. code-block:: bash

         export LIBERO_TYPE=plus
         bash examples/embodiment/run_embodiment.sh libero_10_plus_ppo_abot_m0 \
           actor.model.model_path=./ABot-m0-LIBERO-10k-step/checkpoints/steps_10000_pytorch_model.pt

      .. dropdown:: Advanced configuration (optional)

         - **Released SFT checkpoint for evaluation**: download ``acvlab/ABot-M0-LIBERO`` instead of the 10k-step RL baseline:

           .. code-block:: bash

              hf download acvlab/ABot-M0-LIBERO --local-dir ./ABot-M0-LIBERO

         - **Offline VGGT weights**: for machines without Hugging Face access at runtime, download ``facebook/VGGT-1B`` and export ``VGGT_MODEL_PATH`` (``VGGT_PATH`` selects the source checkout at install time; ``VGGT_MODEL_PATH`` selects the weights at runtime):

           .. code-block:: bash

              hf download facebook/VGGT-1B --local-dir ./VGGT-1B
              export VGGT_MODEL_PATH=$PWD/VGGT-1B

         - **Hydra configuration reference**: see :doc:`Training configuration <../../reference/configuration>`.

.. dropdown:: Standalone evaluation

   Use the unified Evaluation section to verify ABot-M0 checkpoints before training. Start from the :doc:`LIBERO evaluation guide <../../evaluations/guides/libero>` and set ``actor.model.model_path`` to the ABot-M0 checkpoint.

   .. list-table::
      :header-rows: 1
      :widths: 28 36 36

      * - Suite
        - Config source
        - What to change
      * - LIBERO-10
        - ``libero_10_ppo_abot_m0`` via the Evaluation config fallback
        - Point ``actor.model.model_path`` at the ABot-M0 checkpoint.
      * - LIBERO-10+
        - ``libero_10_plus_ppo_abot_m0`` via the Evaluation config fallback
        - Set ``LIBERO_TYPE=plus`` and point ``actor.model.model_path`` at the ABot-M0 checkpoint.

   For CLI usage, Hydra overrides, logs, and video output, use the :doc:`Evaluation CLI reference <../../evaluations/reference/cli>` and :doc:`Evaluation results reference <../../evaluations/reference/results>`.

Visualization and Results
-------------------------

Launch TensorBoard with ``tensorboard --logdir ./logs --port 6006`` and watch ``env/success_once``. Pass ``env.eval.video_cfg.save_video=True`` to record evaluation videos; see :doc:`Training metrics <../../reference/metrics>` for all logged metrics and logger backends.
