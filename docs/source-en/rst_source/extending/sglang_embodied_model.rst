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

- The driver script (e.g. ``eval_embodied_agent.py``) starts one or more sglang
  server processes via :func:`launch_sglang_router_and_server` and pushes each
  server's URL to the rollout workers; server startup, ``/health`` polling, and
  port allocation are handled by ``SGLangServerWorker`` (driver side) — the
  rollout worker itself no longer owns a server subprocess;
- ``SGLangEmbodiedWorker`` picks the driver-assigned server URL for its own rank,
  uses it to create an :class:`InferenceHTTPClient`, loads the model's **sglang
  adapter**, and exchanges observations and actions with the environment Worker
  through channels. **The HTTP request to the server (msgpack encoding + retries)
  is performed by the worker itself**;
- The **sglang adapter** (a plain standalone class, e.g. ``DreamZeroSGLangAdapter``)
  only holds model-specific pure logic: ``build_request`` turns an env
  observation into the request payload, ``parse_response`` turns the server
  response into actions, and ``action_path`` declares the action endpoint. It
  **does not hold an HTTP client and does not send requests itself**.

Therefore, integrating a new model usually **does not require modifying**
``rlinf/workers/rollout/sglang/sglang_embodied_worker.py``, and you do not need to
care how the server is started — server parameters come entirely from the YAML
``rollout.sglang.server`` block passed straight to the sglang server process. The
model is selected by ``rollout.model.model_type``, and the call flow is as follows:

.. code-block:: text

   rollout.model.model_type: "<your_model>"
                 │
                 ▼
   driver: launch_sglang_router_and_server()
                 │  (rollout.sglang.server_type == "embodied" → SGLangServerWorker)
                 ▼
   each SGLangServerWorker.init_server()
                 │  (rollout.sglang.server → ServerArgs.from_kwargs → dispatch_launch)
                 ▼
   SGLangEmbodiedWorker.init_worker()
                 │
                 ├── get_sglang_adapter_cls("<your_model>")
                 ├── pick this rank's server URL, create InferenceHTTPClient
                 └── create YourSglangAdapter(cfg, rank)

   SGLangEmbodiedWorker.predict(env_obs)   # driven by EmbodiedEvalRunner
                 │
                 ├── adapter.build_request(env_obs)  → (payload, state)
                 ├── http_client.post(adapter.action_path, payload, msgpack=True, ...)
                 └── adapter.parse_response(resp, state)  → [N, H, D] actions


To enter this call flow, the following conditions must all be satisfied in the
configuration:

.. code-block:: yaml

   runner:
     task_type: embodied_eval
     only_eval: true

   rollout:
     rollout_backend: sglang
     sglang:
       serving_mode: embodied      # rollout worker = SGLangEmbodiedWorker
       server_type: embodied      # server dispatch = embodied (via dispatch_launch)
       launch_server: true        # driver launches the server group
     model:
       model_type: "<your_model>"

Note that ``rollout.sglang.server_type`` and ``rollout.model.model_type`` are two
**orthogonal** fields with different names and different responsibilities:

- ``rollout.sglang.server_type`` decides **which sglang dispatch branch the
  server subprocess runs** (within the same ``SGLangServerWorker`` class:
  ``srt`` = language model via
  ``sglang.srt.entrypoints.http_server.launch_server``; ``embodied`` = VLA /
  diffusion via ``sglang.multimodal_gen.runtime.launch_server.dispatch_launch``,
  serving the ``/v1/actions/generations`` endpoint);
- ``rollout.model.model_type`` decides **which sglang adapter the rollout worker
  loads** (the adapter registry lookup key).

``serving_mode: embodied`` also cannot be omitted — otherwise RLinf creates a
regular ``SGLangWorker`` instead of ``SGLangEmbodiedWorker``.


Adaptation Steps
----------------

The following describes, in the recommended order, the work required for a new model.

Step 1: Confirm the SGLang Server Action Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf's sglang adapter builds requests for and parses responses from SGLang
Server (the actual HTTP round-trip is performed by the worker). Before writing
RLinf code, first confirm that the SGLang side already has the following
capabilities:

1. ``sglang serve`` can load the target model or target Pipeline;
2. SGLang Server provides a VLA interface for this model that accepts batched observations and returns batched actions;
3. The request and response formats are fixed and can represent the images, text, states, and cache information required by the model.

.. warning::

   The sglang adapter in the RLinf repository is only responsible for building action requests and parsing responses. The model Pipeline, action route, and
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

``"your_model"``, the name used in ``register_sglang_adapter``, and
``rollout.model.model_type`` in the YAML must be identical. The adapter registry
lookup is case-insensitive, but using lowercase everywhere is still recommended.


Step 3: Implement the sglang adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create the adapter file under the model's own directory (in the model layer,
next to the training-side model code):

.. code-block:: text

   rlinf/models/embodiment/<your_model>/sglang_adapter.py

The adapter is a **plain standalone class** (it does not need to inherit any base
class) with the constructor signature ``(cfg, rank)``. It must implement
``build_request`` / ``parse_response`` and declare ``action_path``:

.. code-block:: python

   import numpy as np
   import torch


   class YourSglangAdapter:
       action_path = "/v1/actions/generations"

       def __init__(self, cfg, rank):
           self.cfg_rollout = cfg.rollout
           self.model_cfg = cfg.rollout.model
           self.rank = rank
           # Create lightweight transforms here (do not load large model weights).

       def build_request(self, env_obs, mode="eval"):
           # 1. Convert RLinf env_obs to model input and normalize it;
           # 2. Assemble the server request payload (a dict);
           # 3. Return (payload, state); state is passed back to parse_response as-is.
           payload = {...}
           state = ...            # e.g. the intermediate observation used for denormalization
           return payload, state

       def parse_response(self, resp, state):
           # 1. Read the normalized action from the server response;
           # 2. Denormalize it to environment-scale actions.
           actions = ...
           info = {
               "prev_logprobs": ...,
               "prev_values": ...,
               "forward_inputs": ...,
           }
           return torch.as_tensor(actions, dtype=torch.float32), info

Interface contract:

- **HTTP is handled by the worker** — the adapter does not hold an HTTP client
  and does not send requests. The worker calls ``build_request`` to get the
  payload, sends it via ``http_client.post(adapter.action_path, payload,
  msgpack=True, ...)``, then hands the response to ``parse_response``;
- The ``env_obs`` passed to ``build_request`` is a dictionary of environment
  observations organized by batch; general fields include ``main_images`` and
  ``task_descriptions``, and the model can also use ``wrist_images``, ``states``
  or other views. It returns ``(payload, state)``: ``payload`` is the request
  body sent to the server, and ``state`` is any intermediate value to be reused
  in ``parse_response``;
- ``parse_response`` returns ``(actions, info)``: ``actions`` must be a Tensor
  with shape ``[N, num_action_chunks, action_dim]``; ``info`` is an additional
  information dictionary reserved for future training extensions — currently it
  can return ``prev_logprobs``, ``prev_values`` and ``forward_inputs`` as
  DreamZero does;
- The current SGLang embodied Worker is used only for evaluation. If the adapter
  does not support training mode, it should explicitly raise
  ``NotImplementedError`` when ``mode != "eval"``. Support for using SGLang as a
  rollout worker for embodied model training is planned.

.. important::

   Do not load the model in the adapter. The model should exist only in the
   ``sglang serve`` subprocess. Keep only data transformations and a small amount
   of request context in the adapter; otherwise weights will be loaded repeatedly
   and additional GPU memory will be consumed.


Step 4: Register the adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The adapter is registered with a **lazy builder** in
``_register_builtin_sglang_adapters()`` inside
``rlinf/models/embodiment/sglang_adapter.py`` (mirroring the model-registration
style of ``rlinf.models``): put the heavy imports inside the builder so your
adapter module is imported only when that ``model_type`` is actually looked up:

.. code-block:: python

   # rlinf/models/embodiment/sglang_adapter.py
   def _register_builtin_sglang_adapters():
       def _build_your_model():
           from rlinf.models.embodiment.your_model.sglang_adapter import (
               YourSglangAdapter,
           )

           return YourSglangAdapter

       register_sglang_adapter(
           SupportedModel.YOUR_MODEL.value,
           _build_your_model,
           force=True,
       )

The worker looks it up via ``get_sglang_adapter_cls(model_type)`` in
``init_worker``. If registration is missing, Worker initialization reports that
no sglang adapter is registered for the corresponding ``model_type``.


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

1. The rollout Worker type in the logs is ``SGLangEmbodiedWorker`` and the server
   Worker type is ``SGLangServerWorker`` (``server_type=embodied``);
2. The ``multimodal_gen sglang serve: launching in-process ...`` line printed in
   the logs carries the correct ``pipeline``, ``backend``, ``tp_size`` and GPU;
3. ``/health`` responds before the timeout (server first-time compilation +
   weight loading is slow, so allow enough time);
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
   * - ``rlinf/models/embodiment/dreamzero/sglang_adapter.py``
     - ``DreamZeroSGLangAdapter``: observation transforms, request building, response parsing, and action postprocessing
   * - ``rlinf/models/embodiment/sglang_adapter.py``
     - adapter registry (``register_sglang_adapter`` / ``get_sglang_adapter_cls``); lazily registers the DreamZero adapter
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


Adapter Adaptation
~~~~~~~~~~~~~~~~~~~

``sglang_adapter.py`` uses a **single class** ``DreamZeroSGLangAdapter`` to hold
all of the DreamZero adaptation logic (observation/action transforms + request
assembly + response parsing):

1. ``__init__(cfg, rank)`` reuses the training data transforms
   (``build_dreamzero_composed_transform``) to prepare observation conversion and
   action inverse-transform; it does not load the large model nor build an HTTP client;
2. ``build_request(env_obs, mode)`` turns an env observation into the server
   request payload and returns a ``state`` (the converted observation) for
   ``parse_response`` to reuse;
3. ``parse_response(resp, state)`` reads the normalized action from the response
   and denormalizes it to environment-scale actions;
4. ``action_path`` declares the action endpoint ``/v1/actions/generations``.

**The HTTP request is performed by ``SGLangEmbodiedWorker``** — the adapter only
builds the payload and parses the response, it does not send requests; server
parameters also come entirely from the YAML ``rollout.sglang.server`` block.

The adapter is registered via a lazy builder (see
``rlinf/models/embodiment/sglang_adapter.py``):

.. code-block:: python

   register_sglang_adapter(
       SupportedModel.DREAMZERO.value,
       _build_dreamzero_sglang_adapter,   # lazily imports DreamZeroSGLangAdapter inside
       force=True,
   )

The complete data flow for one inference is:

.. code-block:: text

   RLinf env_obs
       │
       ├── build_request()
       │     ├── _observation_convert()
       │     │     main_images       → video.image
       │     │     wrist_images      → video.wrist_image
       │     │     states            → state.state
       │     │     task_descriptions → annotation.task
       │     ├── _normalize_obs()   dataset transform + metadata normalization
       │     │     (text is NOT tokenized on the client; the SGLang server does it)
       │     └── assemble payload (raw prompt text + normalized observation)
       │
       ├── worker: POST /v1/actions/generations (msgpack)
       │
       └── parse_response()
             ├── _unapply()  [B, H, max_action_dim] → environment-scale actions
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

The adapter's ``build_request`` assembles the payload for
``POST /v1/actions/generations``, which the worker sends via
``http_client.post(..., msgpack=True)``. The logical structure of the payload is
as follows (transported with msgpack, so Tensors and ndarrays do not need to be
expanded into large lists first):

.. code-block:: json

   {
     "model": "/path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-repacked",
     "input": {
       "prompt": [
         "<training-format prompt>"
       ],
       "observation": {}
     },
     "parameters": {
       "session_ids": [
         "rlinf-eval-r0-stage0-slot0"
       ],
       "reset_mask": [
         false
       ],
       "negative_prompts": [
         "<negative prompt>"
       ],
       "seed": 1140
     },
     "runtime": {
       "response_format": "envelope",
       "output_format": "numpy"
     }
   }

Where:

- ``input.prompt`` is the **raw instruction text** for each environment (it is
  not tokenized on the client; the SGLang server tokenizes it);
- ``input.observation`` is the normalized model input (the output of
  ``_normalize_obs`` with the prompt/token-related keys removed);
- ``parameters.session_ids`` identifies each logical environment slot and is used by the Server to reuse video or text caches;
- ``parameters.reset_mask`` is used to clear the cache for the corresponding
  session before the next request (all ``false`` by default in evaluation);
- ``parameters.negative_prompts`` are the negative prompts;
- ``parameters.seed`` comes from ``rollout.sglang.seed``.

The worker reads the normalized actions returned by the Server from the following
location and hands them to ``parse_response``:

.. code-block:: python

   response["data"][0]["action"]["values"]

These actions are still in DreamZero's normalized and padded action space and cannot be sent directly to the environment; they must pass through
``DreamZeroSGLangAdapter._unapply`` (called inside ``parse_response``).


Server Parameters and Pipeline Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The server's startup parameters come entirely from the YAML
``rollout.sglang.server`` block. That block is forwarded verbatim to
``sglang.multimodal_gen``'s ``ServerArgs.from_kwargs(**)``: top-level keys are
``ServerArgs`` fields, and the ``pipeline_config`` sub-block is dispatched by
``from_kwargs`` into ``DreamZeroPipelineConfig`` (matched by
``pipeline_class_name``), which also converts ``backend`` / ``disagg_role``
strings to enums. A typical block looks like:

.. code-block:: yaml

   rollout:
     sglang:
       server:
         model_path: ${rollout.model.model_path}
         backend: sglang
         pipeline_class_name: DreamZeroPipeline
         tp_size: ${..tensor_parallel_size}
         num_gpus: 1
         attention_backend: TORCH_SDPA
         dit_cpu_offload: false
         cfg_parallel_degree: 1
         sp_degree: 1
         pipeline_config:            # → DreamZeroPipelineConfig
           cfg_scale: 5.0
           default_num_inference_steps: 16
           action_horizon: 16
           num_frames: 33
           synthetic_height: 160
           synthetic_width: 320
           dreamzero_compile_components: true
           dreamzero_max_sessions: 128

Key points:

- ``host`` / ``port`` / ``master_port`` are filled at runtime by
  ``SGLangServerWorker`` (free ports allocated serially under a PortLock);
  do not set them in YAML;
- the top-level ``ServerArgs`` fields map directly to ``sglang serve`` arguments
  (``backend``, ``attention_backend``, ``tp_size``, ``num_gpus``,
  ``dit_cpu_offload``, ``cfg_parallel_degree``, ``sp_degree``, ...);
- the ``pipeline_config`` sub-block maps to ``DreamZeroPipelineConfig`` fields,
  named identically to the sglang side (``cfg_scale``,
  ``default_num_inference_steps``, ``dreamzero_compile_components``,
  ``dreamzero_max_sessions``, ...);
- all model paths point to ``rollout.model.model_path``; the Server loads the
  different model components per the checkpoint layout.

Model shape-related fields (``num_frames``, tile parameters, ``synthetic_*``,
``action_horizon``, ...) must remain consistent with the checkpoint training
configuration and cannot be changed arbitrarily based only on GPU memory
availability.


Model YAML
----------

The DreamZero model configuration is located at
``examples/embodiment/config/model/dreamzero_5b.yaml``. Fields directly related to SGLang evaluation
can be summarized as:

.. code-block:: yaml

   model_type: "dreamzero"

   model_path: null
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
     - Lookup key for the adapter registry; must be ``dreamzero``
   * - ``model_path``
     - Full checkpoint path; must be overridden in the evaluation YAML
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
       serving_mode: "embodied"      # rollout worker = SGLangEmbodiedWorker
       server_type: "embodied"      # server dispatch = embodied (via dispatch_launch)
       launch_server: true          # driver launches the server group
       launch_router: false         # no router (action endpoint not forwarded by router)
       group_name: SGLangServerGroup
       router_group_name: SGLangRouterGroup

- ``rollout_backend: sglang`` selects the SGLang backend;
- ``serving_mode: embodied`` selects the rollout worker = ``SGLangEmbodiedWorker``;
- ``server_type: embodied`` selects the server dispatch branch = ``embodied``
  (VLA / diffusion, via ``sglang.multimodal_gen``'s ``dispatch_launch``).
  ``serving_mode`` and ``server_type`` are orthogonal: the former controls how
  the worker connects, the latter which sglang dispatch the server subprocess
  runs; both take effect within the same ``SGLangServerWorker`` class — there
  is no longer a separate server class;
- ``launch_router: false`` is required: the sglang router only forwards fixed
  LLM endpoints (``/generate``, ``/v1/chat/completions``), not the dreamzero
  ``/v1/actions/generations``, so the embodied path disables the router and
  rollout workers hit their rank-assigned server URL directly.

``pipeline_stage_num`` participates in calculating the eval batch size for each rollout rank;
``return_logprobs: false`` indicates that policy probabilities are not needed for evaluation.


Server Startup and Parallel Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``rollout.sglang`` top level holds RLinf-private keys (HTTP client, parallelism,
launch toggles); the ``server`` sub-block holds the ``ServerArgs`` fields
forwarded to sglang:

.. code-block:: yaml

   rollout:
     sglang:
       tensor_parallel_size: 1
       pipeline_parallel_size: 1
       launch_server: true
       launch_router: false
       server_type: embodied
       seed: 1140

       server:                       # → ServerArgs.from_kwargs
         backend: sglang
         pipeline_class_name: DreamZeroPipeline
         tp_size: ${..tensor_parallel_size}
         num_gpus: 1
         attention_backend: TORCH_SDPA
         dit_cpu_offload: false
         cfg_parallel_degree: 1
         sp_degree: 1
         pipeline_config:            # → DreamZeroPipelineConfig
           cfg_scale: 5.0
           default_num_inference_steps: 16
           dreamzero_compile_components: true

Top-level private-key fields:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Field
     - Meaning
   * - ``tensor_parallel_size``
     - TP size per server engine; the launcher packs hardware ranks into engines
       of ``tp×pp`` (4 GPUs tp=2 → two 2-GPU servers; tp=4 → one 4-GPU server)
   * - ``pipeline_parallel_size``
     - PP size per server engine
   * - ``launch_server`` / ``launch_router``
     - Whether the driver launches the server group / router
   * - ``server_type``
     - server dispatch branch: ``srt`` / ``embodied``
   * - ``seed``
     - Random seed used for each action request

The ``server`` sub-block fields are ``ServerArgs`` fields (``backend``,
``attention_backend``, ``tp_size``, ``num_gpus``, ``dit_cpu_offload``,
``cfg_parallel_degree``, ``sp_degree``, ...); the ``pipeline_config`` sub-block
holds ``DreamZeroPipelineConfig`` fields (``cfg_scale``,
``default_num_inference_steps``, ``dreamzero_*``, ...). Refer to the
``ServerArgs`` / ``DreamZeroPipelineConfig`` of the SGLang version in use for
the exact field meanings.

With multiple rollout ranks, ``launch_sglang_router_and_server`` packs the
hardware ranks into multiple server engines of
``tensor_parallel_size × pipeline_parallel_size`` and launches them in parallel.
**The HTTP port and master_port are allocated by the worker's PortLock as free
ports serially** — do not set ``host`` / ``port`` / ``port_base`` /
``master_port_base`` in YAML; this avoids concurrent ranks grabbing the same
port (an EADDRINUSE on master_port=30005 was seen before).


HTTP Client Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   rollout:
     sglang:
       http_timeout_s: 600
       http_max_retries: 5
       http_retry_backoff_s: 1.0

These fields are read by ``SGLangEmbodiedWorker`` to create the
:class:`InferenceHTTPClient` that sends requests to the server. Field
descriptions:

- ``http_timeout_s`` is the timeout for a single action request;
- ``http_max_retries`` is the number of retries for connection errors or retryable 5xx responses;
- ``http_retry_backoff_s`` is the base wait time for linear backoff.

.. note::

   Embodied action requests always use **msgpack** (so ndarrays are not expanded
   into enormous JSON lists for images / large batches); the worker calls
   ``http_client.post(..., msgpack=True)``, and there is no longer an
   ``http_payload_format`` config option.


DreamZero Model and Data Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   rollout:
     model:
       model_type: "dreamzero"
       precision: bf16
       model_path: /path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-repacked
       metadata_json_path: /path/to/metadata.json
       embodiment_tag: "libero_sim"

       action_horizon: 16
       num_action_chunks: 16
       num_action_per_block: 16
       target_video_height: 160
       target_video_width: 320

Among these fields:

- ``model_path`` is the checkpoint loaded by SGLang Server;
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
     rollout.model.model_path=/path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers \
     rollout.model.metadata_json_path=/path/to/metadata.json

For the detailed DreamZero SGLang evaluation workflow, see :doc:`../evaluations/guides/dreamzero_sglang`.

For initial joint debugging, the number of environments can be overridden:

.. code-block:: bash

   bash evaluations/run_eval.sh \
     libero \
     libero_spatial_dreamzero_eval_sglang \
     rollout.model.model_path=/path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers \
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


sglang adapter Not Registered
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Confirm that the following three names are identical and that the adapter is
registered in ``_register_builtin_sglang_adapters()`` inside
``rlinf/models/embodiment/sglang_adapter.py``:

.. code-block:: text

   SupportedModel.register("dreamzero")
   register_sglang_adapter("dreamzero", _build_dreamzero_sglang_adapter, force=True)
   rollout.model.model_type: dreamzero


Server Fails to Start
~~~~~~~~~~~~~~~~~~~~~

The logs print a ``multimodal_gen sglang serve: launching in-process ...`` line
and the server subprocess output (streamed straight to the Ray actor's
stdout/stderr). Check the following first:

- Whether the current SGLang installation contains ``DreamZeroPipeline`` and the action endpoint;
- Whether the checkpoint path and component layout are correct;
- Whether ``num_gpus``, ``tp_size``, and ``sp_degree`` match the available GPUs;
- Whether the port is occupied (HTTP/master_port are auto-allocated by PortLock
  but may still clash with external processes);
- Whether enough startup wait time is given for first-time compilation and
  weight loading.


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

Confirm that the sglang adapter does not import and create a local large DreamZero model. The model can only be loaded by
``sglang serve``. Also check ``max_sessions``, eval batch size, compilation options, and
parallel configuration; DreamZero sets ``max_sessions`` to the current Worker's eval batch size by default, and
increasing the number of parallel environments also increases the Server-side cache requirements.
