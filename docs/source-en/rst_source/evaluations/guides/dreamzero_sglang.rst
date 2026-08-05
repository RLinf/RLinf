DreamZero SGLang Evaluation
===========================

This guide runs DreamZero LIBERO evaluation through the RLinf SGLang embodied backend. Use this path for inference-only evaluation with an SGLang-native DreamZero checkpoint layout.

Compared with the original DreamZero eval path in :doc:`../../examples/embodied/sft_dreamzero`, this backend runs the large DreamZero network in a separately launched ``sglang serve`` process instead of loading it in the rollout worker. The worker still needs the DreamZero transform stack installed, though: ``rlinf.models.embodiment.dreamzero.sglang_adapter`` imports ``rlinf.data.datasets.dreamzero.data_transforms``, which pulls in the ``groot`` package installed by ``install_dreamzero_deps``. The eval driver launches the SGLang server group and pushes each server URL to the rollout workers; each worker is a thin client that posts batched observations to the URL assigned to its rank over the VLA action API, then denormalizes the returned action chunks before stepping LIBERO.

Install the Test Environment
----------------------------

Set up RLinf with the embodied, LIBERO, and DreamZero SGLang dependencies:

.. code-block:: bash

   cd /path/to/RLinf
   bash requirements/install.sh embodied --env libero --model dreamzero \
     --venv /path/to/dreamzero_test

DreamZero support is still under review in SGLang, so use SGLang code that
includes `sgl-project/sglang#30679 <https://github.com/sgl-project/sglang/pull/30679>`_
and install it with the ``diffusion`` extra:

.. code-block:: bash

   source /path/to/dreamzero_test/bin/activate
   cd /path/to/sglang_dreamzero
   pip install -e "python[diffusion]"

Prepare the Checkpoint
----------------------

Download the LIBERO SFT Diffusers checkpoint from
`RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers <https://huggingface.co/RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers>`_.
Point ``rollout.model.model_path`` at the downloaded checkpoint directory.

.. code-block:: bash

   hf download RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers \
     --local-dir /path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

The checkpoint should contain ``experiment_cfg/metadata.json``. If metadata is not available in the checkpoint, generate it from the LIBERO dataset and set ``rollout.model.metadata_json_path`` explicitly:

.. code-block:: bash

   python toolkits/lerobot/generate_dreamzero_metadata.py \
     --preset libero_sim \
     --dataset-root /path/to/libero \
     --output-metadata /path/to/metadata.json

Run LIBERO-Spatial
------------------

The default SGLang eval config is ``evaluations/libero/libero_spatial_dreamzero_eval_sglang.yaml``.

.. code-block:: bash

   cd /path/to/RLinf
   bash evaluations/run_eval.sh libero libero_spatial_dreamzero_eval_sglang \
     rollout.model.model_path=/path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

For a custom metadata file, add:

.. code-block:: bash

   rollout.model.metadata_json_path=/path/to/metadata.json

Important Config Fields
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Field
     - Purpose
   * - ``rollout.rollout_backend: sglang``
     - Selects the SGLang rollout backend.
   * - ``rollout.sglang.serving_mode: embodied``
     - Selects ``SGLangEmbodiedWorker``; the SGLang serve group is launched by the
       driver script (``launch_sglang_router_and_server``), and the worker connects
       to the driver-assigned server URL for inference.
   * - ``rollout.sglang.server.pipeline_config.default_num_inference_steps``
     - Controls the DreamZero denoising steps used by the server.
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
     - Number of actions returned per model request; ``env.eval.max_steps_per_rollout_epoch`` must be divisible by this value.

Parallel Overrides
------------------

The supported DreamZero SGLang evaluation entry is ``libero_spatial_dreamzero_eval_sglang``. For local experiments, adjust parallelism by overriding fields on this config instead of switching to a different YAML:

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_dreamzero_eval_sglang \
     rollout.sglang.server.num_gpus=2 \
     rollout.sglang.server.cfg_parallel_degree=2 \
     rollout.model.model_path=/path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

Validation
----------

Evaluation logs are written under ``logs/<timestamp>-<config>/``. Check ``eval_embodiment.log`` for the SGLang server command, endpoint readiness, per-episode results, and the final ``eval/success_once`` metric.

The LIBERO-Spatial SGLang config uses ``auto_reset: True`` and ordered reset states to cover the full suite with fewer parallel environments. See :ref:`libero-eval-config` for the LIBERO trajectory accounting rules.

Troubleshooting
---------------

- If SGLang cannot find model components, confirm that ``rollout.model.model_path`` points to the downloaded Diffusers checkpoint directory.
- If metadata loading fails, set ``rollout.model.metadata_json_path`` to an existing ``metadata.json`` generated for ``libero_sim``.
- If local HTTP requests unexpectedly use a proxy, set ``NO_PROXY=127.0.0.1,localhost`` before launching evaluation.
