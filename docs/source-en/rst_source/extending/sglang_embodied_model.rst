Adapting an Embodied Model in SGLang Server
===============================================

This document describes how to integrate an embodied model already adapted to SGLang Server into
RLinf's evaluation rollout and evaluate the model using the various simulators supported by RLinf.

The document is divided into two parts:

- The first part describes the steps and interface conventions required when adapting any new model;
- The second part uses DreamZero as an example and explains, item by item, the code and YAML configuration that need to be modified.

.. note::

   This document only covers the **eval rollout / sglang-serve** path. This path is responsible for converting environment observations
   into actions during evaluation and does not include training-side model registration, FSDP Policy, or SFT adaptation. For training-side adaptation, refer to
   :doc:`Adding a New Model with FSDP <new_model_fsdp>` and
   :doc:`Adding a New SFT Model <new_model_sft>`.


Part One: Adapting a New Model
==============================

Overall Architecture
--------------------

The SGLang embodied evaluation path separates general logic from model-specific logic:

- ``SGLangEmbodiedWorker`` is responsible for starting and stopping the ``sglang serve`` subprocess, polling
  ``/health``, allocating ports, and exchanging observations and actions with the environment Worker through channels;
- Subclasses of ``EmbodiedActionPolicy`` are responsible for model-specific observation preprocessing, HTTP requests,
  response parsing, and action postprocessing.

Therefore, integrating a new model usually **does not require modifying**
``rlinf/workers/rollout/sglang/sglang_embodied_worker.py``. The model is selected by
``rollout.model.model_type``, and the call flow is as follows:

.. code-block:: text

   rollout.model.model_type: "<your_model>"
                 │
                 ▼
   SGLangEmbodiedWorker.init_worker()
                 │
                 ├── get_action_policy_cls("<your_model>")
                 ├── start sglang serve
                 └── create YourActionPolicy
                              │
                              ├── convert env_obs to model input
                              ├── request the model's action endpoint
                              └── return [N, H, D] actions

To enter this call flow, all four of the following conditions must be satisfied in the configuration:

.. code-block:: yaml

   runner:
     task_type: embodied_eval
     only_eval: true

   rollout:
     rollout_backend: sglang
     sglang:
       serving_mode: embodied
     model:
       model_type: "<your_model>"

Here, ``serving_mode: embodied`` cannot be omitted. Otherwise, RLinf creates a regular
``SGLangWorker`` instead of ``SGLangEmbodiedWorker``, preventing the model from working correctly.


Adaptation Steps
----------------

The following describes, in the recommended order, the work required for a new model.

Step 1: Confirm the SGLang Server Action Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf's Action Policy is a client of SGLang Server. Before writing RLinf code, first confirm
that the SGLang side already has the following capabilities:

1. ``sglang serve`` can load the target model or target Pipeline;
2. SGLang Server provides a VLA interface for this model that accepts batched observations and returns batched actions;
3. The request and response formats are fixed and can represent the images, text, states, and cache information required by the model.

.. warning::

   The Action Policy in the RLinf repository is only responsible for action requests and conversion. The model Pipeline, action route, and
   related ``sglang serve`` parameters still need to be implemented in the SGLang version being used.


Step 2: Register ``model_type``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Register the model in ``rlinf/config.py`` so that configuration validation recognizes the name:

.. code-block:: python

   SupportedModel.YOUR_MODEL = SupportedModel.register("your_model", force=True)

If the model needs to pass embodied configuration validation, it must also be added to ``EMBODIED_MODEL``:

.. code-block:: python

   EMBODIED_MODEL = {
       # ...
       SupportedModel.YOUR_MODEL,
   }

``"your_model"``, the name in the Action Policy decorator, and
``rollout.model.model_type`` in the YAML must be identical. Policy Registry lookup is case-insensitive,
but using lowercase everywhere is still recommended.


Step 3: Implement the Action Policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a model file in the following directory:

.. code-block:: text

   rlinf/workers/rollout/sglang/action_policies/<your_model>.py

The Policy needs to inherit from ``EmbodiedActionPolicy`` and be registered with a decorator:

.. code-block:: python

   import torch

   from rlinf.workers.rollout.sglang.action_policies.base import (
       EmbodiedActionPolicy,
   )
   from rlinf.workers.rollout.sglang.action_policies.registry import (
       register_action_policy,
   )


   @register_action_policy("your_model")
   class YourActionPolicy(EmbodiedActionPolicy):
       def __init__(self, cfg, server_url, rank):
           super().__init__(cfg, server_url, rank)
           # Create lightweight transforms and an HTTP client here.

       def infer(self, env_obs, mode="eval"):
           # 1. Convert RLinf env_obs to model input.
           # 2. Normalize it and send a request to SGLang Server.
           # 3. Parse the response and denormalize the actions.
           actions = ...
           info = {
               "prev_logprobs": ...,
               "prev_values": ...,
               "forward_inputs": ...,
           }
           return torch.as_tensor(actions, dtype=torch.float32), info

``infer`` is the only method that must be implemented. Its interface is:

- The input ``env_obs`` is a dictionary of environment observations organized by batch;
- General fields include ``main_images`` and ``task_descriptions``; the model can also use
  ``wrist_images``, ``states``, or other views;
- The output ``actions`` must be a Tensor with shape
  ``[N, num_action_chunks, action_dim]``;
- The output ``info`` is an additional information dictionary. This interface is reserved for future training extensions; currently, it can return
  ``prev_logprobs``, ``prev_values``, and ``forward_inputs`` as DreamZero does;
- The current SGLang embodied Worker is used only for evaluation. If the Policy does not support training mode, then when
  ``mode != "eval"`` it should explicitly raise ``NotImplementedError``. Support for using SGLang
  as a rollout worker for embodied model training is planned.

.. important::

   Do not load the model in the Action Policy. The model should exist only in the ``sglang serve``
   subprocess. Keep only data transformations, the HTTP Client, and a small amount of request context in the Policy; otherwise,
   weights will be loaded repeatedly and additional GPU memory will be consumed.


Step 4: Import the Policy to Trigger Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add the following in ``rlinf/workers/rollout/sglang/action_policies/__init__.py``:

.. code-block:: python

   from rlinf.workers.rollout.sglang.action_policies import your_model  # noqa: F401,E401

The decorator executes only when the module is imported. If this step is omitted, Worker initialization reports that no
Action Policy is registered for the corresponding ``model_type``.


Step 5: Add the Model YAML and Evaluation YAML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is recommended to maintain the model configuration and SGLang evaluation configuration separately:

.. code-block:: text

   examples/embodiment/config/model/<your_model>.yaml
   evaluations/<benchmark>/<your_model>_eval_sglang.yaml

The model configuration describes the model's fixed structure, such as action dimensions, action horizon, and input image size;
the evaluation YAML describes the checkpoint, environment, resources, Server startup parameters, and HTTP parameters. This allows the same
model configuration to be reused by multiple YAML configuration files.


Step 6: Test and Debug
~~~~~~~~~~~~~~~~~~~~~~

For the first run, it is recommended to reduce ``env.eval.total_num_envs`` and confirm the following in order:

1. The Worker type in the logs is ``SGLangEmbodiedWorker``;
2. The ``sglang serve`` command printed in the logs contains model-specific parameters;
3. ``/health`` responds within ``spawn_timeout``;
4. The request sent by the Policy can be parsed correctly by the action endpoint;
5. The action dimensions and dtype output by the Server meet the convention;
6. The denormalized action shape matches the simulator's requirements;
7. Increase the number of parallel environments and the degree of model parallelism only after a small-scale run succeeds.


Part Two: DreamZero as an Example
=================================

DreamZero's SGLang evaluation path consists of the following files:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - File
     - Purpose
   * - ``rlinf/config.py``
     - Register ``dreamzero`` and add it to ``EMBODIED_MODEL``
   * - ``rlinf/workers/rollout/sglang/action_policies/dreamzero.py``
     - Observation transformation, HTTP Client, Serve parameters, and action postprocessing
   * - ``rlinf/workers/rollout/sglang/action_policies/__init__.py``
     - Import the DreamZero Policy
   * - ``examples/embodiment/config/model/dreamzero_5b.yaml``
     - DreamZero 5B model configuration
   * - ``evaluations/libero/libero_spatial_dreamzero_eval_sglang.yaml``
     - SGLang evaluation YAML for LIBERO-Spatial


Code Adaptation
---------------

Register the Model
~~~~~~~~~~~~~~~~~~

DreamZero is registered in ``rlinf/config.py`` as follows:

.. code-block:: python

   SupportedModel.DREAMZERO = SupportedModel.register("dreamzero", force=True)

At the same time, ``SupportedModel.DREAMZERO`` is added to ``EMBODIED_MODEL``. Therefore, the evaluation YAML
can use:

.. code-block:: yaml

   rollout:
     model:
       model_type: dreamzero


Policy Adaptation
~~~~~~~~~~~~~~~~~

``dreamzero.py`` divides model adaptation into four steps:

1. ``DreamZeroActionRequest`` and ``DreamZeroActionResult`` define the request and response;
2. ``HttpDreamZeroActionClient`` is responsible for encoding, retrying, sending, and parsing HTTP requests;
3. ``_DreamZeroActionAdapter`` reuses training data transformations to perform observation and action conversion;
4. ``DreamZeroActionPolicy`` implements the RLinf interface and generates DreamZero Server startup parameters.

The Policy registration code is:

.. code-block:: python

   @register_action_policy("dreamzero")
   class DreamZeroActionPolicy(EmbodiedActionPolicy):
       ...

The complete data flow for one inference is:

.. code-block:: text

   RLinf env_obs
       │
       ├── observation_convert()
       │     main_images       → video.image
       │     wrist_images      → video.wrist_image
       │     states            → state.state
       │     task_descriptions → annotation.task
       │
       ├── normalize_obs()
       │     dataset transform + metadata normalization + prompt tokenize
       │
       ├── POST /v1/actions/generations
       │
       ├── unapply()
       │     [B, H, max_action_dim] → environment-scale actions
       │
       └── actions [B, H, action_dim]

Using ``embodiment_tag: libero_sim`` as an example, ``main_images`` and
``wrist_images`` are converted into two video modalities, an external camera and a wrist camera; ``states`` is converted into
robot state, and ``task_descriptions`` is converted into language instructions. The inverse transformation slices out the action dimensions required by LIBERO according to
the metadata and binarizes gripper actions to ``-1`` or ``1``.

To make DreamZero support a new simulator, it is usually also necessary to add, under
``rlinf/data/datasets/dreamzero/data_transforms/``, the corresponding
``embodiment_tag``, ``RolloutObsLayout``, modality definitions, training prompt format, and
embodiment id.


HTTP Requests
~~~~~~~~~~~~~

The DreamZero Client calls ``POST /v1/actions/generations``. The JSON form is as follows;
when using msgpack, the logical fields remain the same, but Tensors and ndarrays do not need to be expanded into large lists first:

.. code-block:: json

   {
     "model": "/path/to/dreamzero_checkpoint",
     "parameters": {
       "action_input": {},
       "session_ids": [
         "rlinf-eval-r0-stage0-slot0"
       ],
       "reset_mask": [
         false
       ],
       "prompts": [
         "<training-format prompt>"
       ],
       "negative_prompts": [
         "text_negative:missing"
       ],
       "seed": 1140
     },
     "runtime": {
       "response_format": "envelope",
       "output_format": "numpy"
     }
   }

Where:

- ``action_input`` is the output of ``_DreamZeroActionAdapter.normalize_obs``;
- ``session_ids`` identifies each logical environment slot and is used by the Server to reuse video or text caches;
- ``reset_mask`` is used to clear the cache for the corresponding session before the next request;
- ``prompt_cache_keys`` in the Python dataclass is sent as
  ``prompts`` in the HTTP payload, and ``negative_prompt_cache_keys`` is sent as ``negative_prompts``;
- ``seed`` comes from ``rollout.sglang.seed``.

The Client reads the normalized actions returned by the Server from:

.. code-block:: python

   response["data"][0]["action"]["values"]

These actions are still in DreamZero's normalized and padded action space and cannot be sent directly to the environment; they must pass through
``_DreamZeroActionAdapter.unapply``.


Server Parameters and Pipeline Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``DreamZeroActionPolicy.build_sglang_serve_args`` appends the following types of parameters:

.. code-block:: text

   --backend sglang
   --pipeline DreamZeroPipeline
   --pipeline-config-path <tmpdir>/dreamzero_pipeline_rank<rank>.json
   --sp-degree <sp_size>
   --cfg-parallel-size <cfg_parallel_degree>
   --dreamzero-dit-path <model_path>
   --dreamzero-vae-path <model_path>
   --dreamzero-text-encoder-path <model_path>
   --dreamzero-image-encoder-path <model_path>

All these model paths point to ``rollout.model.model_path``, and the Server loads
different model components according to the checkpoint layout.

The Policy also generates the JSON used by ``DreamZeroPipelineConfig``. The main mappings are as follows:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Pipeline JSON Field
     - Source
   * - ``dreamzero_compile_components``
     - ``rollout.sglang.compile_components``
   * - ``dreamzero_sequence_parallel_size``
     - ``rollout.sglang.sp_degree``; reads ``sp_size`` when not set
   * - ``dreamzero_max_sessions``
     - ``rollout.sglang.max_sessions``; defaults to this Worker's eval batch size
   * - ``cfg_scale``
     - ``rollout.sglang.cfg_scale``
   * - ``action_horizon``
     - ``rollout.model.action_horizon``
   * - ``num_inference_steps``
     - ``rollout.sglang.num_inference_steps``
   * - ``num_frames``, tile parameters
     - ``rollout.model.action_head_cfg.config``
   * - ``synthetic_height`` / ``synthetic_width``
     - ``rollout.model.target_video_height`` / ``target_video_width``

This means that model shape-related fields must remain consistent with the checkpoint training configuration and cannot be
changed arbitrarily based only on GPU memory availability.


Model YAML
----------

The DreamZero model configuration is located at
``examples/embodiment/config/model/dreamzero_5b.yaml``. Fields directly related to SGLang evaluation
can be summarized as:

.. code-block:: yaml

   model_type: "dreamzero"

   model_path: null
   tokenizer_path: null
   metadata_json_path: null

   action_dim: 32
   state_horizon: 1
   action_horizon: 24
   num_action_per_block: 24
   max_action_dim: 32
   max_state_dim: 64

   target_video_height: 176
   target_video_width: 320

   action_head_cfg:
     config:
       num_frames: 33
       tile_size_height: 34
       tile_size_width: 34
       tile_stride_height: 18
       tile_stride_width: 16
       tiled: false

The meanings of the main fields are as follows:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``model_type``
     - Lookup key for the Action Policy Registry; must be ``dreamzero``
   * - ``model_path``
     - Full checkpoint path; must be overridden in the evaluation YAML
   * - ``tokenizer_path``
     - Text tokenizer path; the DreamZero example uses ``google/umt5-xxl``
   * - ``metadata_json_path``
     - Dataset statistics used for state and action normalization and denormalization
   * - ``action_dim`` / ``max_action_dim``
     - Model action width; ``max_action_dim`` is the padded width used for multiple embodiments
   * - ``action_horizon``
     - Action horizon predicted by the Server in one request
   * - ``num_action_per_block``
     - Number of actions used by each action block in the DreamZero DiT
   * - ``target_video_height`` / ``target_video_width``
     - Video resolution used by the data transformations and Pipeline
   * - ``action_head_cfg.config``
     - DreamZero network structure and video and tile parameters; these should usually remain consistent with the training checkpoint

The values in the model configuration are defaults. The evaluation YAML can override these values, but
``action_horizon``, ``num_action_per_block``, video dimensions, and model structure-related fields must match
the current checkpoint. For example, the LIBERO SGLang configuration in this section overrides the horizon from the default
``24`` to ``16``; this is a requirement of that evaluation checkpoint, not a general recommendation.


Detailed Evaluation YAML
------------------------

The complete example is located at
``evaluations/libero/libero_spatial_dreamzero_eval_sglang.yaml``. The configuration blocks are explained below.

Hydra defaults
~~~~~~~~~~~~~~

.. code-block:: yaml

   defaults:
     - env/libero_spatial@env.eval
     - model/dreamzero_5b@rollout.model
     - override hydra/job_logging: stdout

   hydra:
     searchpath:
       - file://${oc.env:EMBODIED_PATH}/config/

This composes the ``libero_spatial`` simulator configuration into ``env.eval`` and the
``dreamzero_5b`` model configuration into ``rollout.model``. ``run_eval.sh`` sets
``EMBODIED_PATH``.


Cluster and Runner Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement:
       env,rollout: all

   runner:
     task_type: embodied_eval
     max_epochs: 1
     only_eval: True
     ckpt_path: null

Field descriptions:

- ``component_placement`` specifies how env and rollout are placed;
- ``task_type: embodied_eval`` selects ``EmbodiedEvalRunner``;
- ``only_eval: True`` is required, and ``SGLangEmbodiedWorker`` asserts it;
- ``ckpt_path`` can be ``null`` in this path, and the Server loads weights from
  ``rollout.model.model_path``;
- The current script supports evaluation only, not training.


Environment Parallelism and Evaluation Steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   env:
     eval:
       rollout_epoch: 1
       total_num_envs: 128
       auto_reset: True
       ignore_terminations: True
       max_episode_steps: 480
       max_steps_per_rollout_epoch: 1920
       group_size: 1
       use_fixed_reset_state_ids: True
       use_ordered_reset_state_ids: True
       is_eval: True

``total_num_envs`` is the total number of parallel environments and must be divisible by the actual number of environment Workers,
``pipeline_stage_num``, and ``group_size``.

``max_steps_per_rollout_epoch`` must be divisible by
``rollout.model.num_action_chunks``. The Worker calculates how many action requests are needed per epoch using the following
formula:

.. code-block:: python

   n_eval_chunk_steps = (
       env.eval.max_steps_per_rollout_epoch
       // rollout.model.num_action_chunks
   )

In the example, ``1920 // 16 = 120``. ``num_action_chunks`` must be consistent with the action chunk length returned by one inference and
actually executed by the environment.


SGLang Worker Dispatch
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   rollout:
     rollout_backend: "sglang"
     pipeline_stage_num: 1
     return_logprobs: false

     sglang:
       serving_mode: "embodied"

- ``rollout_backend: sglang`` selects the SGLang backend;
- ``serving_mode: embodied`` further selects ``SGLangEmbodiedWorker``.

``pipeline_stage_num`` participates in calculating the eval batch size for each rollout rank;
``return_logprobs: false`` indicates that policy probabilities are not needed for evaluation.


Server Startup and Parallel Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   rollout:
     sglang:
       spawn_timeout: 900
       attention_backend: "TORCH_SDPA"
       compile_components: true
       num_gpus: 1
       tp_size: 1
       sp_size: 1
       cfg_parallel_degree: 1
       dit_cpu_offload: false
       cfg_scale: 5.0
       num_inference_steps: 16
       seed: 1140

Field descriptions:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Field
     - Meaning
   * - ``spawn_timeout``
     - Maximum number of seconds the Worker waits for ``/health`` to succeed
   * - ``attention_backend``
     - Attention backend passed to ``sglang serve``
   * - ``compile_components``
     - Whether to compile the relevant components of the DreamZero Pipeline
   * - ``num_gpus``
     - Number of GPUs used by each Server
   * - ``tp_size``
     - Tensor parallel size passed to the Server
   * - ``sp_size`` / ``sp_degree``
     - DreamZero sequence parallel size; if both are set, ``sp_degree`` takes precedence
   * - ``cfg_parallel_degree``
     - Classifier-free guidance parallel size
   * - ``dit_cpu_offload``
     - Whether to offload DiT-related parts to the CPU
   * - ``cfg_scale``
     - CFG scale used for DreamZero inference
   * - ``num_inference_steps``
     - Number of flow-matching/diffusion inference steps
   * - ``seed``
     - Random seed used for each action request

If there are multiple rollout ranks, each rank starts an independent Server. The default service port is
``port_base + rank * port_stride``. To customize it, add:

.. code-block:: yaml

   rollout:
     sglang:
       host: 127.0.0.1
       port_base: 30010
       port_stride: 100
       master_port_base: 30100

The port ranges must avoid conflicts with other tasks or other ranks.


HTTP Client Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   rollout:
     sglang:
       http_timeout_s: 600
       http_max_retries: 5
       http_retry_backoff_s: 1.0
       http_payload_format: "msgpack"
       debug_sessions: false
       debug_batch: false

Field descriptions:

- ``http_timeout_s`` is the timeout for a single action request;
- ``http_max_retries`` is the number of retries for connection errors or retryable 5xx responses;
- ``http_retry_backoff_s`` is the base wait time for linear backoff;
- ``http_payload_format`` supports ``json`` and ``msgpack``. ``msgpack`` is recommended for images and large batches
  to avoid expanding ndarrays into enormous JSON lists;
- ``debug_sessions`` and ``debug_batch`` can be enabled when jointly debugging cache or shape issues, but should not be
  kept enabled for large-scale evaluation.


DreamZero Model and Data Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   rollout:
     model:
       model_type: "dreamzero"
       precision: bf16
       model_path: /path/to/dreamzero_checkpoint
       tokenizer_path: google/umt5-xxl
       metadata_json_path: /path/to/metadata.json
       embodiment_tag: "libero_sim"

       action_horizon: 16
       num_action_chunks: 16
       num_action_per_block: 16
       target_video_height: 160
       target_video_width: 320

Among these fields:

- ``model_path`` is the checkpoint loaded by SGLang Server;
- ``tokenizer_path`` is used only for the text tokenizer; do not use it in place of ``model_path``;
- ``metadata_json_path`` provides normalization statistics from the training data. If it is not explicitly specified, the code only attempts
  ``model_path/experiment_cfg/metadata.json``;
- ``embodiment_tag`` selects the observation layout, modality transformations, action postprocessing, and embodiment id;
- ``action_horizon`` is the action length generated by the model at one time;
- ``num_action_chunks`` is the action length RLinf sends to and executes in the environment each time;
- ``num_action_per_block`` is a DreamZero network structure parameter;
- ``target_video_height`` and ``target_video_width`` must be consistent with the checkpoint and Server
  Pipeline.

All three action lengths in the current example are ``16``. If a new checkpoint's generation horizon differs from the chunk that is actually
executed, the Server output, Policy slicing, and environment execution strategy must all be confirmed together; changing only
one of these fields is insufficient.


Generating Metadata
~~~~~~~~~~~~~~~~~~~

DreamZero's observation normalization and action denormalization depend on dataset metadata. The LIBERO example can use:

.. code-block:: bash

   python toolkits/lerobot/generate_dreamzero_metadata.py \
     --preset libero_sim \
     --dataset-root /path/to/libero \
     --output-metadata /path/to/metadata.json

After generation, write the path to ``rollout.model.metadata_json_path``. The metadata must come from data and an embodiment matching the training
checkpoint; using incorrect statistics may not cause an immediate error, but it results in incorrect action
scales.


Running Evaluation
------------------

After preparing the DreamZero dependencies, an SGLang environment that supports ``DreamZeroPipeline``, the checkpoint, and
metadata, run the following from the repository root:

.. code-block:: bash

   export DREAMZERO_PATH=/path/to/DreamZero

   bash evaluations/run_eval.sh \
     libero \
     libero_spatial_dreamzero_eval_sglang \
     rollout.model.model_path=/path/to/dreamzero_checkpoint \
     rollout.model.metadata_json_path=/path/to/metadata.json

For initial joint debugging, the number of environments can be overridden:

.. code-block:: bash

   bash evaluations/run_eval.sh \
     libero \
     libero_spatial_dreamzero_eval_sglang \
     rollout.model.model_path=/path/to/dreamzero_checkpoint \
     rollout.model.metadata_json_path=/path/to/metadata.json \
     env.eval.total_num_envs=4

``run_eval.sh`` sets ``EMBODIED_PATH`` and adds ``DREAMZERO_PATH`` to
``PYTHONPATH``. Make sure ``DREAMZERO_PATH`` points to the DreamZero
code directory containing the ``groot`` package.

Common Issues
-------------

Worker Is Not ``SGLangEmbodiedWorker``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check whether both of the following are set:

.. code-block:: yaml

   rollout:
     rollout_backend: sglang
     sglang:
       serving_mode: embodied


Action Policy Not Registered
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Confirm that the following three names are identical and that the Policy module has been imported in ``action_policies/__init__.py``:

.. code-block:: text

   SupportedModel.register("dreamzero")
   @register_action_policy("dreamzero")
   rollout.model.model_type: dreamzero


Server Fails to Start
~~~~~~~~~~~~~~~~~~~~~

The Worker prints the complete ``sglang serve`` command and the Server log path. Check the following first:

- Whether the current SGLang installation contains ``DreamZeroPipeline`` and the action endpoint;
- Whether the checkpoint path and component layout are correct;
- Whether ``num_gpus``, ``tp_size``, and ``sp_size`` match the available GPUs;
- Whether the port is occupied;
- Whether ``spawn_timeout`` is sufficient to cover initial compilation and weight loading.


Request to Local Server Times Out
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``NO_PROXY`` must include ``127.0.0.1,localhost``; otherwise, ``/health`` and action requests may
be sent to an upstream proxy. The Worker sets it automatically when starting a local Server; when starting one manually or testing the Client separately,
you need to check the proxy environment variables yourself.


Incorrect Action Shape or Scale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check the following in order:

1. Whether the Server output is ``[B, H, max_action_dim]``;
2. Whether ``action_horizon``, ``num_action_chunks``, and ``num_action_per_block`` are consistent with the
   checkpoint;
3. Whether ``embodiment_tag`` selects the correct data transformation;
4. Whether ``metadata_json_path`` comes from matching training data;
5. Whether the action dimensions after ``unapply`` meet the environment's requirements;
6. Whether the image resolution and view order are consistent with training.


Abnormal GPU Memory Usage
~~~~~~~~~~~~~~~~~~~~~~~~~

Confirm that the Action Policy does not import and create a local large DreamZero model. The model can only be loaded by
``sglang serve``. Also check ``max_sessions``, eval batch size, compilation options, and
parallel configuration; DreamZero sets ``max_sessions`` to the current Worker's eval batch size by default, and
increasing the number of parallel environments also increases the Server-side cache requirements.
