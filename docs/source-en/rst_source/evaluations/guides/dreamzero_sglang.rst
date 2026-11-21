DreamZero SGLang Evaluation
===========================

Overview
--------

Evaluate DreamZero on LIBERO through the RLinf SGLang embodied backend using an SGLang-native Diffusers checkpoint layout. Instead of loading the DreamZero network inside each rollout worker (as in :doc:`../../examples/embodied/sft_dreamzero`), the eval driver launches a standalone ``sglang serve`` server group and assigns each rollout worker a server URL; workers keep the DreamZero data-transform stack (including ``groot``), post batched observations over the VLA action API, and denormalize the returned action chunks before stepping LIBERO.

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

DreamZero SGLang evaluation requires CUDA 13.0 or later. Because DreamZero support is under review in SGLang, install SGLang from a local checkout containing `sgl-project/sglang#30679 <https://github.com/sgl-project/sglang/pull/30679>`_ with the ``diffusion`` extra into the same virtual environment.

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``.

.. tab-set::

   .. tab-item:: DreamZero SGLang + LIBERO

      .. code-block:: bash

         # 1. Install
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env libero --model dreamzero && source .venv/bin/activate
         pip install -e "/path/to/sglang_dreamzero/python[diffusion]"

         # 2. Download LIBERO SFT Diffusers checkpoint
         hf download RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers --local-dir ./RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

         # 3. Evaluate LIBERO-Spatial
         bash evaluations/run_eval.sh libero libero_spatial_dreamzero_eval_sglang \
           rollout.model.model_path=./RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

The default config is ``evaluations/libero/libero_spatial_dreamzero_eval_sglang.yaml``, using `RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers <https://huggingface.co/RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers>`_.

.. dropdown:: Generate Missing Metadata

   The checkpoint should include ``experiment_cfg/metadata.json``. If it is missing, generate it from the LIBERO dataset and pass ``rollout.model.metadata_json_path=./metadata.json`` to ``run_eval.sh``:

   .. code-block:: bash

      python toolkits/lerobot/generate_dreamzero_metadata.py \
        --preset libero_sim \
        --dataset-root /path/to/libero \
        --output-metadata ./metadata.json

.. dropdown:: Key Configuration and Parallel Overrides

   .. list-table::
      :header-rows: 1
      :widths: 34 66

      * - Field
        - Purpose
      * - ``rollout.rollout_backend: sglang``
        - Selects the SGLang rollout backend.
      * - ``rollout.sglang.serving_mode: embodied``
        - Launches the SGLang server group and assigns each rollout worker a server URL.
      * - ``rollout.sglang.server.pipeline_config.default_num_inference_steps``
        - DreamZero denoising steps used by the server.
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

   **Parallel overrides**

   Adjust parallelism directly on ``libero_spatial_dreamzero_eval_sglang`` through Hydra overrides:

   .. code-block:: bash

      bash evaluations/run_eval.sh libero libero_spatial_dreamzero_eval_sglang \
        rollout.sglang.server.num_gpus=2 \
        rollout.sglang.server.cfg_parallel_degree=2 \
        rollout.model.model_path=./RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

Results and Troubleshooting
---------------------------

Logs go to ``logs/<timestamp>-<config>/eval_embodiment.log``, reporting the SGLang server command, endpoint readiness, per-episode outcomes, and ``eval/success_once`` (see :doc:`../reference/results`). The config uses ``auto_reset: True`` and ordered reset states to cover the full suite with fewer parallel environments; see :ref:`LIBERO evaluation configuration <libero-eval-config>` for trajectory counting rules.

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Symptom
     - Fix
   * - SGLang cannot find model components
     - Confirm ``rollout.model.model_path`` points to the downloaded Diffusers checkpoint directory.
   * - Metadata loading fails
     - Set ``rollout.model.metadata_json_path`` to an existing ``metadata.json`` generated for ``libero_sim``.
   * - Local HTTP requests hit a proxy
     - Export ``NO_PROXY=127.0.0.1,localhost`` before launching evaluation.
