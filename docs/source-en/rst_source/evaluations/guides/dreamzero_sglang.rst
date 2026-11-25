DreamZero SGLang Evaluation
===========================

Overview
--------

Evaluate DreamZero on LIBERO through RLinf's SGLang embodied backend using an SGLang-native Diffusers checkpoint. The eval driver starts standalone ``sglang serve`` servers and assigns each rollout worker a server URL; workers run the DreamZero data transforms, request action chunks over HTTP, and step LIBERO.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Benchmark
      :text-align: center

      LIBERO-Spatial (``libero``)

   .. grid-item-card:: Models
      :text-align: center

      DreamZero WAN2.2-5B (SGLang backend)

   .. grid-item-card:: Config Dir
      :text-align: center

      ``evaluations/libero/``

   .. grid-item-card:: Related Training
      :text-align: center

      :doc:`DreamZero SFT <../../examples/embodied/sft_dreamzero>`

Setup and Run
-------------

Requires CUDA 13.0 or later. Until `sgl-project/sglang#30679 <https://github.com/sgl-project/sglang/pull/30679>`_ lands upstream, place a local SGLang checkout containing that PR at ``./sglang`` and install its ``diffusion`` extra into the same virtual environment.

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``.

.. tab-set::

   .. tab-item:: LIBERO

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env libero --model dreamzero && source .venv/bin/activate
         pip install -e "./sglang/python[diffusion]"

      **2. Download LIBERO SFT Diffusers Checkpoint**

      .. code-block:: bash

         hf download RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers --local-dir ./RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

      **3. Evaluate LIBERO-Spatial**

      .. code-block:: bash

         bash evaluations/run_eval.sh libero libero_spatial_dreamzero_eval_sglang \
           rollout.model.model_path=./RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

The default config is ``evaluations/libero/libero_spatial_dreamzero_eval_sglang.yaml``, using `RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers <https://huggingface.co/RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers>`_.

.. dropdown:: Generate Missing Metadata

   The checkpoint normally ships with ``experiment_cfg/metadata.json``. If it is missing, generate it from your LIBERO dataset directory (``./libero_dataset``) and pass ``rollout.model.metadata_json_path=./metadata.json`` to ``run_eval.sh``:

   .. code-block:: bash

      python toolkits/lerobot/generate_dreamzero_metadata.py \
        --preset libero_sim \
        --dataset-root ./libero_dataset \
        --output-metadata ./metadata.json

.. dropdown:: Key Configuration and Parallel Overrides

   .. list-table::
      :header-rows: 1
      :widths: 38 62

      * - Field
        - Purpose
      * - ``rollout.rollout_backend: sglang``
        - Selects the SGLang rollout backend.
      * - ``rollout.sglang.serving_mode: embodied``
        - Launches the SGLang server group and assigns each rollout worker a server URL.
      * - ``rollout.sglang.server.pipeline_config.default_num_inference_steps``
        - DreamZero denoising steps on the server.
      * - ``rollout.sglang.server.pipeline_config.cfg_scale``
        - Classifier-free guidance scale for action inference.
      * - ``rollout.sglang.server.cfg_parallel_degree``
        - Splits positive and negative CFG branches across ranks when set to ``2``.
      * - ``rollout.sglang.server.tp_size``
        - Tensor-parallel size for the DreamZero DiT.
      * - ``rollout.sglang.server.sp_degree``
        - Sequence-parallel size for the DreamZero DiT attention sequence.
      * - ``rollout.model.model_path``
        - DreamZero Diffusers checkpoint directory loaded by SGLang.
      * - ``rollout.model.metadata_json_path``
        - Normalization statistics used before and after action inference.
      * - ``rollout.model.num_action_chunks``
        - Actions returned per request; ``env.eval.max_steps_per_rollout_epoch`` must be divisible by this value.

   Override parallelism directly on ``libero_spatial_dreamzero_eval_sglang``:

   .. code-block:: bash

      bash evaluations/run_eval.sh libero libero_spatial_dreamzero_eval_sglang \
        rollout.sglang.server.num_gpus=2 \
        rollout.sglang.server.cfg_parallel_degree=2 \
        rollout.model.model_path=./RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

Results and Troubleshooting
---------------------------

Logs go to ``logs/<timestamp>-<config>/eval_embodiment.log``, reporting server startup, endpoint readiness, per-episode outcomes, and ``eval/success_once`` (see :doc:`../reference/results`). The config uses ``auto_reset: True`` and ordered reset states to cover the full suite with fewer parallel environments; see :ref:`LIBERO evaluation configuration <libero-eval-config>` for trajectory counting.

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Symptom
     - Fix
   * - SGLang cannot find model components
     - Confirm ``rollout.model.model_path`` points to the downloaded Diffusers checkpoint directory.
   * - Metadata loading fails
     - Set ``rollout.model.metadata_json_path`` to a ``metadata.json`` generated for ``libero_sim``.
   * - Local HTTP requests hit a proxy
     - Export ``NO_PROXY=127.0.0.1,localhost`` before launching evaluation.
