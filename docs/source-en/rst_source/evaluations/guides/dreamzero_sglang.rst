DreamZero SGLang Evaluation
===========================

This guide runs DreamZero LIBERO evaluation through the RLinf SGLang embodied backend. Use this path for inference-only evaluation with an SGLang-native DreamZero checkpoint layout.

Compared with the original DreamZero eval path in :doc:`../../examples/embodied/sft_dreamzero`, this backend does not require ``DREAMZERO_PATH`` or the external DreamZero Python package at runtime. The RLinf rollout worker starts and owns an SGLang action server, sends batched observations over the VLA action API, and denormalizes the returned action chunks before stepping LIBERO.

Install the Test Environment
----------------------------

Set up RLinf with the embodied, LIBERO, and DreamZero SGLang dependencies:

.. code-block:: bash

   cd /path/to/RLinf
   bash requirements/install.sh embodied --env libero --model dreamzero-sglang \
     --torch 2.11.0 --python 3.11.14 --venv /path/to/dreamzero_test

Install the SGLang build that contains DreamZero support, including the ``diffusion`` extra:

.. code-block:: bash

   source /path/to/dreamzero_test/bin/activate
   cd /path/to/sglang_dreamzero
   pip install -e "python[diffusion]"

Prepare the Repacked Checkpoint
-------------------------------

Download the LIBERO SFT checkpoint from `RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000 <https://huggingface.co/RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000>`_.
The DreamZero SGLang backend expects a repacked checkpoint directory. Convert the raw checkpoint once from the RLinf repository:

.. code-block:: bash

   cd /path/to/RLinf
   python toolkits/difusser-like-weight-convert/dreamzero_repack.py \
     --path /path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000

By default, the output is written next to the source checkpoint with a ``-repacked`` suffix. Point ``rollout.model.model_path`` at this repacked directory.

The repacked checkpoint should contain ``experiment_cfg/metadata.json``. If metadata is not available in the checkpoint, generate it from the LIBERO dataset and set ``rollout.model.metadata_json_path`` explicitly:

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
     rollout.model.model_path=/path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000-repacked \
     rollout.model.tokenizer_path=/path/to/umt5-xxl

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
     - Starts ``SGLangEmbodiedWorker``, which launches and controls the local SGLang action server.
   * - ``rollout.sglang.http_payload_format: msgpack``
     - Uses the binary action payload path used by DreamZero evaluation.
   * - ``rollout.sglang.num_inference_steps``
     - Controls the DreamZero denoising steps used by the server.
   * - ``rollout.sglang.cfg_scale``
     - Classifier-free guidance scale for action inference.
   * - ``rollout.sglang.cfg_parallel_degree``
     - Splits positive and negative CFG branches across ranks when set to ``2``.
   * - ``rollout.sglang.tp_size``
     - Tensor-parallel size for the DreamZero DiT.
   * - ``rollout.sglang.sp_size``
     - Sequence-parallel size for the DreamZero DiT attention sequence.
   * - ``rollout.model.model_path``
     - Repacked DreamZero checkpoint directory loaded by SGLang.
   * - ``rollout.model.metadata_json_path``
     - Normalization statistics used before and after action inference.
   * - ``rollout.model.num_action_chunks``
     - Number of actions returned per model request; ``env.eval.max_steps_per_rollout_epoch`` must be divisible by this value.

Parallel Overrides
------------------

The supported DreamZero SGLang evaluation entry is ``libero_spatial_dreamzero_eval_sglang``. For local experiments, adjust parallelism by overriding fields on this config instead of switching to a different YAML:

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_dreamzero_eval_sglang \
     rollout.sglang.num_gpus=2 \
     rollout.sglang.cfg_parallel_degree=2 \
     rollout.model.model_path=/path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000-repacked \
     rollout.model.tokenizer_path=/path/to/umt5-xxl

Validation
----------

Evaluation logs are written under ``logs/<timestamp>-<config>/``. Check ``eval_embodiment.log`` for the SGLang server command, endpoint readiness, per-episode results, and the final ``eval/success_once`` metric.

The LIBERO-Spatial SGLang config uses ``auto_reset: True`` and ordered reset states to cover the full suite with fewer parallel environments. See :ref:`libero-eval-config` for the LIBERO trajectory accounting rules.

Troubleshooting
---------------

- If SGLang cannot find model components, confirm that ``rollout.model.model_path`` points to the repacked checkpoint directory rather than the raw DreamZero checkpoint.
- If metadata loading fails, set ``rollout.model.metadata_json_path`` to an existing ``metadata.json`` generated for ``libero_sim``.
- If local HTTP requests unexpectedly use a proxy, set ``NO_PROXY=127.0.0.1,localhost`` before launching evaluation.
