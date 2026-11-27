DreamZero SGLang Evaluation
===========================

Evaluate DreamZero on LIBERO through the RLinf SGLang embodied backend using an SGLang-native DreamZero Diffusers checkpoint.

Compared with the native evaluation path in :doc:`../../examples/embodied/sft_dreamzero`, each rollout worker launches and owns a dedicated SGLang Diffusion server process on its assigned GPUs instead of loading the large DreamZero network inside the worker. The rollout worker retains the DreamZero data transforms, posts batched observations to its local server over the VLA action API, and denormalizes the returned action chunks before stepping LIBERO.

Install the Test Environment
----------------------------

DreamZero SGLang evaluation requires CUDA 13.0 or later.

Clone RLinf and install the embodied, LIBERO, and DreamZero dependencies:

.. tab-set::

   .. tab-item:: Global

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git ./RLinf-dreamzero
         cd ./RLinf-dreamzero
         bash requirements/install.sh embodied --env libero --model dreamzero --venv .venv-dreamzero
         source .venv-dreamzero/bin/activate

   .. tab-item:: China Mirror

      .. code-block:: bash

         git clone https://gh-proxy.com/github.com/RLinf/RLinf.git ./RLinf-dreamzero
         cd ./RLinf-dreamzero
         bash requirements/install.sh embodied --env libero --model dreamzero --venv .venv-dreamzero --use-mirror
         source .venv-dreamzero/bin/activate

Clone ``RLinf/dreamzero`` and export ``DREAMZERO_PATH`` if it is not already present in the virtualenv:

.. code-block:: bash

   git clone https://github.com/RLinf/dreamzero.git ../dreamzero
   export DREAMZERO_PATH="$PWD/../dreamzero"

DreamZero serving support is tracked in `sgl-project/sglang#30679 <https://github.com/sgl-project/sglang/pull/30679>`_. Clone that branch and install it with the ``diffusion`` extra:

.. code-block:: bash

   git clone https://github.com/sgl-project/sglang.git ../sglang-dreamzero
   git -C ../sglang-dreamzero fetch origin pull/30679/head:pr-30679
   git -C ../sglang-dreamzero checkout pr-30679
   uv pip install -e "../sglang-dreamzero/python[diffusion]"

Prepare the Checkpoint
----------------------

Download the LIBERO SFT Diffusers checkpoint from `RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers <https://huggingface.co/RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers>`_ and point ``rollout.model.model_path`` at the downloaded directory:

.. tab-set::

   .. tab-item:: Global

      .. code-block:: bash

         hf download RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers \
           --local-dir ./assets/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

   .. tab-item:: China Mirror

      .. code-block:: bash

         export HF_ENDPOINT=https://hf-mirror.com
         hf download RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers \
           --local-dir ./assets/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

The checkpoint should contain ``experiment_cfg/metadata.json``. If metadata is missing, generate it from the LIBERO dataset and set ``rollout.model.metadata_json_path`` explicitly:

.. code-block:: bash

   python toolkits/lerobot/generate_dreamzero_metadata.py \
     --preset libero_sim \
     --dataset-root ./data/libero \
     --output-metadata ./assets/metadata.json

Run LIBERO-Spatial
------------------

Set ``rollout.model.model_path`` in ``evaluations/libero/libero_spatial_dreamzero_eval_sglang.yaml`` (or pass it on the command line) and launch evaluation:

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_dreamzero_eval_sglang \
     rollout.model.model_path=./assets/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

For a custom metadata file, append ``rollout.model.metadata_json_path=./assets/metadata.json``.

Important Config Fields
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Field
     - Purpose
   * - ``rollout.rollout_backend: sglang``
     - Selects the SGLang rollout backend.
   * - ``rollout.sglang.serving_mode: embodied``
     - Runs the embodied SGLang rollout worker, which launches and owns its local server process on its assigned GPUs.
   * - ``rollout.sglang.server.pipeline_config.default_num_inference_steps``
     - Number of DreamZero denoising steps executed on the server.
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

Parallel Overrides
------------------

RLinf derives GPUs per server instance automatically from ``dp_size * tp_size * sp_degree * cfg_parallel_degree``. To run with 2-way CFG parallelism on ``libero_spatial_dreamzero_eval_sglang``, override ``cfg_parallel_degree`` directly:

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_dreamzero_eval_sglang \
     rollout.sglang.server.cfg_parallel_degree=2 \
     rollout.model.model_path=./assets/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

Validation
----------

Evaluation logs are written under ``logs/<timestamp>-<config>/``. Check ``eval_embodiment.log`` for server startup, readiness, per-episode outcomes, and the final ``eval/success_once`` metric.

The LIBERO-Spatial SGLang config uses ``auto_reset: True`` and ordered reset states to cover the full suite with fewer parallel environments. See :ref:`libero-eval-config` for LIBERO trajectory accounting rules.

Troubleshooting
---------------

- If SGLang cannot find model components, confirm that ``rollout.model.model_path`` points to the downloaded Diffusers checkpoint directory.
- If metadata loading fails, set ``rollout.model.metadata_json_path`` to an existing ``metadata.json`` generated for ``libero_sim``.
- If local HTTP requests unexpectedly use a proxy, export ``NO_PROXY=127.0.0.1,localhost`` before launching evaluation.
