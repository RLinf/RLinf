Adding New Models with FSDP+HuggingFace
========================================

This document focuses on how to use the HuggingFace Transformers library together
with PyTorch FSDP (Fully Sharded Data Parallel) to train and generate with
models. It supports any model implemented in HuggingFace as long as it is
compatible with PyTorch. As an example, this guide provides a step-by-step
workflow showing how to integrate a new HuggingFace model into RLinf following
the OpenVLA pattern.

Prerequisites
-------------

* Familiarity with **HuggingFace Transformers**
* Understanding of the **RLinf** framework architecture
* Knowledge of **PyTorch** and distributed training

Step-by-Step Implementation
---------------------------

1. Model Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended way to integrate a model into RLinf is to register a model
builder. This is also the preferred approach when another repository depends on
RLinf.

.. code-block:: python

  from rlinf.models import register_model

  def build_my_model(cfg, torch_dtype):
      from your_package.your_model_action_model import (
          YourModelForRLActionPrediction,
      )

      return YourModelForRLActionPrediction.from_pretrained(
          cfg.model_path,
          torch_dtype=torch_dtype,
          hidden_size=cfg.hidden_size,
          unnorm_key=cfg.unnorm_key,
          action_dim=cfg.action_token_len,
          attn_implementation=cfg.attn_implementation,
          low_cpu_mem_usage=cfg.low_cpu_mem_usage,
          trust_remote_code=cfg.trust_remote_code,
      )

  register_model("your_model_type", build_my_model, category="embodied")

This single registration does two things:

- Makes ``model_type`` pass RLinf config validation.
- Makes ``rlinf.models.get_model(cfg)`` build your model correctly.

Distributed registration when RLinf is a dependency (``RLINF_EXT_MODULE``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When RLinf is pulled in as a third-party dependency and **Ray** launches
distributed workers, calling ``register_model(...)`` only in the main (driver)
process **does not** carry over to worker processes: workers are separate Python
processes and will not re-execute the registration code from your entry script.
As a result, code paths that actually run on workers (rollout, training, etc.)
may still see an unregistered model and fail to build or resolve it.

Set the environment variable ``RLINF_EXT_MODULE`` to the import path of an
**extension module**. On **each worker initialization**, RLinf imports that module
and, when defined, invokes its ``register()`` so workers perform the same model
(or other extension) registration as the driver, without patching RLinf.

The extension module should implement ``register()`` and call ``register_model``
and any other needed hooks there. For the exact env var format and examples, see
the docstring of ``ClusterEnvVar.EXT_MODULE`` in
``rlinf/scheduler/cluster/cluster.py``, e.g.:

.. code-block:: bash

   export RLINF_EXT_MODULE=rlinf_ext
   # or a fully qualified package path, e.g.:
   export RLINF_EXT_MODULE=workflows.scripts.rlinf_ext

Standalone checkpoint utilities (for example
``python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf``) also
load ``RLINF_EXT_MODULE`` before calling ``get_model``, matching worker behavior.
If your ``model_type`` is registered only via an extension module, set this
environment variable when running those tools as well.

2. Model Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create your class in `rlinf/models/embodiment/your_model_action_model.py` and
inherit from a HuggingFace base class.
For a custom VLA model integrated into RLinf, you should follow the interface
contract of `rlinf.models.embodiment.base_policy.BasePolicy` and provide at
least `forward` and `predict_action_batch`. If you inherit directly from
`BasePolicy`, you will usually need to implement `default_forward` and
`predict_action_batch`, while `forward` is dispatched by the base class.
`predict_action_batch` is used to encapsulate generation, decoding, and optional
value computation so the RL-specific logic stays inside the model.

.. code-block:: python

  from typing import Any

  from rlinf.models.embodiment.base_policy import BasePolicy

  class YourModelForRLActionPrediction(BasePolicy):
      def default_forward(
          self,
          forward_inputs: dict[str, Any],
          **kwargs,
      ) -> dict[str, Any]:
          return {
              "logprobs": ...,  # [B, ...]
              "values": ...,    # [B] or [B, 1]
              "entropy": ...,   # [B] or [B, 1]
          }

      def predict_action_batch(
          self,
          env_obs: dict[str, Any],
          mode: str = "train",
          compute_values: bool = True,
          **kwargs,
      ) -> tuple[Any, dict[str, Any]]:
          actions = ...  # [B, action_chunk, action_dim] or flattened env action
          result = {
              "prev_logprobs": ...,
              "prev_values": ...,
              "forward_inputs": {
                  "chains": ...,
                  "denoise_inds": ...,
                  "action": ...,
                  "model_action": ...,
              },
          }
          return actions, result

Refer to `rlinf/models/embodiment/openpi/openpi_action_model.py`. The inputs and
outputs of these interfaces are best understood as follows:

- ``forward(forward_type=..., **kwargs)``:
  A unified dispatch entry that routes to ``default_forward``, ``sft_forward``,
  ``sac_forward``, or other training branches based on ``forward_type``. For a
  custom VLA model, it is recommended to keep this dispatch layer so rollout,
  actor, SFT, and other callers all go through the same entry point.
- ``default_forward(forward_inputs, **kwargs)``:
  Used during training to recompute logprob / value / entropy from the
  intermediate results cached during rollout. Following OpenPI,
  ``forward_inputs`` usually includes at least:

  - ``chains``: the action chain from diffusion or iterative sampling, used to
    recompute logprob during training.
  - ``denoise_inds``: the denoising step index for each sample, which determines
    which training signal to use.
  - ``tokenized_prompt`` / ``tokenized_prompt_mask``: the text prompt and its
    mask.
  - ``action``: the actual action sent to the environment, usually flattened as
    ``[B, ...]``.
  - ``model_action``: the raw action output of the model, usually preserved
    before output transformation.
  - Observation fields: such as ``observation/image``,
    ``observation/state``, ``observation/wrist_image``, and so on, so the model
    can reconstruct the observation.

  OpenPI's ``default_forward`` returns a ``dict`` whose key fields include:

  - ``logprobs``: recomputed log probabilities of the current action under the
    training branch.
  - ``values``: state values produced by the value head.
  - ``entropy``: policy entropy or an equivalent uncertainty statistic used for
    PPO / RL loss.
- ``predict_action_batch(env_obs, mode="train"|"eval", compute_values=True, **kwargs)``:
  Used during rollout / inference. It takes environment observations as input
  and returns actions that can be executed directly. Following OpenPI,
  ``env_obs`` usually includes:

  - ``main_images``: the main-view image.
  - ``wrist_images``: wrist camera images, optional.
  - ``extra_view_images``: additional view images, optional.
  - ``states``: low-dimensional states.
  - ``task_descriptions``: text task descriptions.

  OpenPI returns ``(actions, result)``:

  - ``actions``: action tensors sent directly to the environment.
  - ``result["prev_logprobs"]``: the old-policy logprob recorded during rollout.
  - ``result["prev_values"]``: the old value recorded during rollout.
  - ``result["forward_inputs"]``: the cached fields that will be fed into
    ``default_forward`` again during training.

When implementing a custom model, the field names do not need to be exactly the
same as OpenPI, but it is recommended to guarantee two things: first,
``predict_action_batch`` should return enough ``forward_inputs`` for
``default_forward`` to reconstruct all information required for training;
second, ``default_forward`` should output at least the fields actually required
by the training algorithm, such as ``logprobs``, ``values``, and ``entropy``.

3. Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create `examples/embodiment/config/your_config.yaml`.
It should include fields such as `model_type`, `action_token_len`, and
`precision`. This template exposes your model hyperparameters so experiments are
easy to configure.

.. code-block:: yaml

  model:
    model_type: "your_model_type"
    action_token_len: 7
    action_chunks_len: 1
    unnorm_key: your_action_key
    precision: "bf16"
    vocab_size: 32000
    hidden_size: 4096
    image_size: [224, 224]
    is_lora: False
    low_cpu_mem_usage: True
    trust_remote_code: True

If your custom model needs explicit wrapping rules under FSDP, you can also add
``fsdp_config.wrap_policy`` to the config. When this field contains custom
rules, RLinf will prioritize them. Otherwise, it will continue to follow the
model's own ``_no_split_modules`` / ``_no_split_names`` together with the
built-in logic.

Common fields include:

- ``transformer_layer_cls_to_wrap``: specify modules by class name that should
  be wrapped as transformer blocks. This is suitable for the main structure,
  such as decoder layers or attention blocks.
- ``module_classes_to_wrap``: specify extra modules by class name that should be
  wrapped, such as vision encoders, projectors, or adapters that are not
  standard transformer blocks.
- ``no_split_names``: perform finer-grained wrapping based on a module's
  ``_fsdp_wrap_name``. This is useful when class names are not distinctive
  enough, but the module role is fixed.

.. code-block:: yaml

  fsdp_config:
    wrap_policy:
      transformer_layer_cls_to_wrap:
        - "MyDecoderLayer"
      module_classes_to_wrap:
        - "MyVisionTower"
        - "MyProjector"
      no_split_names:
        - "value_head"

If your custom module is essentially the main transformer block, configuring
``transformer_layer_cls_to_wrap`` is usually enough. ``module_classes_to_wrap``
is more suitable for modules outside the main block that should still
participate in FSDP wrapping separately.

4. Model Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After calling ``register_model("your_model_type", build_my_model, category="embodied")``,
you can directly use ``model_type: your_model_type`` in YAML configs. If your
model also depends on a corresponding ``processor`` class, it is recommended to
load it inside ``build_my_model`` before returning the model instance.

.. code-block:: python

  from omegaconf import OmegaConf
  from rlinf.models import get_model

  cfg = OmegaConf.create(
      {
          "model_type": "your_model_type",
          "precision": "fp32",
          "is_lora": False,
      }
  )
  model = get_model(cfg)

5. Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- RLinf performs extra environment-side processing on actions produced by
  different models. For example, ``prepare_actions(...)`` in
  ``rlinf/envs/action_utils.py`` applies different post-processing logic based
  on ``env_type`` and ``model_type``. If you are using a brand new custom
  ``model_type``, it is recommended to complete this extra action
  post-processing directly inside ``predict_action_batch(...)``, so the
  ``actions`` returned to the environment are already in their final executable
  format. This avoids adding extra environment-side branching logic for every
  new model type.
- When registering a custom model with ``register_model(...)``, you can pass
  ``force=True`` to override an existing implementation. This is suitable when
  you want to replace an RLinf built-in model while keeping the original
  ``model_type``. In that case, you can continue to reuse RLinf's existing
  action processing, training branches, and other compatibility logic for that
  built-in ``model_type``, reducing the amount of additional adaptation code.
  The prerequisite is that your model interface remains fully aligned with the
  built-in one.
- Different VLA models use different wrap policies under FSDP in RLinf. You can
  refer to ``get_fsdp_wrap_policy(...)`` in
  ``rlinf/hybrid_engines/fsdp/utils.py`` for the current logic. If you are
  integrating a custom model, it is recommended to explicitly add the
  corresponding wrap policy config so that key modules such as transformer
  layers, vision encoders, projectors, and value heads participate in FSDP
  wrapping as expected. Otherwise, you may encounter degraded training
  performance, unreasonable memory usage, or modules that are not sharded
  correctly.
