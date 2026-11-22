Add an FSDP Model
=================

Integrate a PyTorch or Hugging Face Transformers model into RLinf's FSDP (Fully Sharded Data Parallel) backend for RL training and rollout. The walkthrough below follows the OpenVLA pattern for embodied policies.

At a Glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Item
     - Details
   * - Files to create/edit
     - ``rlinf/models/embodiment/<name>/`` (or an external package) and ``examples/embodiment/config/<name>.yaml``
   * - Extension point
     - ``register_model`` and ``RLINF_EXT_MODULE`` (see :ref:`out-of-tree-extensions`)
   * - Required methods
     - ``BasePolicy`` subclass implementing ``default_forward`` and ``predict_action_batch``
   * - Quick verification
     - Load your registered model on CPU with ``get_model(cfg)`` and add unit tests under ``tests/unit_tests/``

Step-by-Step Implementation
---------------------------

1. Model Registration
~~~~~~~~~~~~~~~~~~~~~

Register a model builder with ``register_model`` in ``rlinf/models/__init__.py`` (in-tree) or in an external extension module.

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
           action_dim=cfg.action_dim,
           attn_implementation=cfg.attn_implementation,
           low_cpu_mem_usage=cfg.low_cpu_mem_usage,
           trust_remote_code=cfg.trust_remote_code,
       )


   register_model("your_model_type", build_my_model, category="embodied")

This registration does two things:

- Registers ``model_type`` in ``SupportedModel`` so Hydra config validation accepts it.
- Makes ``rlinf.models.get_model(cfg)`` build your model instance.

Distributed Registration for Out-of-Tree Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ray workers run in separate Python processes, so calling ``register_model`` only in your driver script does not reach rollout or actor workers. Put your registration calls inside a ``register()`` hook and export ``RLINF_EXT_MODULE`` before launching (see :ref:`out-of-tree-extensions` for the module layout and ``ClusterEnvVar.EXT_MODULE`` in ``rlinf/scheduler/cluster/cluster.py``):

.. code-block:: bash

   export RLINF_EXT_MODULE=my_pkg.rlinf_ext

.. note::

   Standalone checkpoint utilities such as ``python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf`` also load ``RLINF_EXT_MODULE`` before calling ``get_model``. Export this variable before running them when your ``model_type`` lives in an external package.

2. Model Implementation
~~~~~~~~~~~~~~~~~~~~~~~

Create your policy class under ``rlinf/models/embodiment/<name>/`` (or in your external package), inheriting from ``rlinf.models.embodiment.base_policy.BasePolicy`` and your Hugging Face base class. Implement ``default_forward`` and ``predict_action_batch``; ``BasePolicy.forward`` dispatches ``ForwardType.DEFAULT`` to ``default_forward`` and ``ForwardType.REFL`` to ``refl_forward``. ``predict_action_batch`` encapsulates generation, decoding, and optional value computation so RL-specific logic stays inside the model.

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

For a complete reference implementation, see ``rlinf/models/embodiment/openpi/tasks/rl.py``. Each method plays a distinct role in the rollout-training loop:

- ``forward(forward_type=ForwardType.DEFAULT, **kwargs)``:
  Unified dispatch entry point. ``BasePolicy.forward`` routes ``ForwardType.DEFAULT`` to ``default_forward`` and ``ForwardType.REFL`` to ``refl_forward``. Override ``forward`` if your policy also supports ``ForwardType.SFT`` or ``ForwardType.SAC`` branches.
- ``default_forward(forward_inputs, **kwargs)``:
  Recomputes log-probabilities, values, and entropy during training from tensors cached at rollout time. Following OpenPI, ``forward_inputs`` typically includes:

  - ``chains``: action chain from diffusion or iterative sampling, used to recompute log-probabilities during training.
  - ``denoise_inds``: denoising step index for each sample.
  - ``tokenized_prompt`` / ``tokenized_prompt_mask``: text prompt tokens and mask.
  - ``action``: executed environment action, typically flattened as ``[B, ...]``.
  - ``model_action``: raw model action before output transformation.
  - Observation fields such as ``observation/image``, ``observation/state``, and ``observation/wrist_image`` needed to reconstruct observations.

  ``default_forward`` returns a dict containing the tensors required by your loss:

  - ``logprobs``: recomputed log-probabilities of the sampled actions.
  - ``values``: state values from the value head.
  - ``entropy``: policy entropy for PPO / RL regularization.
- ``predict_action_batch(env_obs, mode="train"|"eval", compute_values=True, **kwargs)``:
  Runs during rollout and evaluation, taking environment observations and returning executable actions. Following OpenPI, ``env_obs`` typically contains:

  - ``main_images``: primary camera image.
  - ``wrist_images``: wrist camera image (optional).
  - ``extra_view_images``: additional view images (optional).
  - ``states``: low-dimensional robot state.
  - ``task_descriptions``: text task instructions.

  ``predict_action_batch`` returns ``(actions, result)``:

  - ``actions``: action tensor sent to the environment.
  - ``result["prev_logprobs"]``: behavior-policy log-probabilities recorded during rollout.
  - ``result["prev_values"]``: behavior-policy value estimates recorded during rollout.
  - ``result["forward_inputs"]``: cached tensors passed back to ``default_forward`` during training.

Your custom model can use its own keys inside ``forward_inputs`` as long as ``predict_action_batch`` caches everything ``default_forward`` needs and ``default_forward`` returns the fields required by your chosen loss (``logprobs``, ``values``, and ``entropy``).

.. _refl-model-interface:

Train from Differentiable Rewards
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ReFL connects a model that generates differentiable samples to a scorer that rewards each sample. Reward gradients pass through the samples to the model's trainable parameters.

Implement ``refl_forward(forward_inputs, xt_source="rollout", **kwargs)`` and return a dict containing ``samples``, a tensor with the batch dimension first. The model handles state sampling, time units, noising, and decoding; its sample layout must match the scorer. ``BasePolicy.forward`` calls this method for ``ForwardType.REFL``. A custom ``forward`` must support the same dispatch.

The actor reads ``xt_source`` from ``algorithm.refl_xt_source``. Return ``forward_inputs`` alongside ``samples`` to set the scorer context; the default is the original input. ``scorer(samples, forward_inputs)`` returns rewards of shape ``[B]``, with higher values preferred. Freeze the scorer's parameters and preserve gradients through its input samples.

SD3 takes ``forward_inputs={"prompts": prompts}``, where ``prompts`` is a list of strings. It samples with its current weights and passes the raw prompts to the ImageReward scorer for tokenization. To supply an existing state, pass ``x_t`` and ``timesteps`` in ``refl_inputs``, and ``prompt_embeds`` and ``pooled_prompt_embeds`` in ``forward_inputs``. SD3 uses times in ``[0, 1]``, with 0 for data and 1 for noise.

Register a scorer builder with ``register_refl_scorer`` and select it through ``actor.refl_reward_model.type``. The builder receives the full ``actor.refl_reward_model`` config and the actor device, and returns a scorer. Save this extension module as ``my_refl_ext.py`` to register ImageReward:

.. code-block:: python

   # my_refl_ext.py
   from rlinf.algorithms.refl_scorers import register_refl_scorer


   def build_image_scorer(cfg, device):
       from rlinf.envs.sim.diffusion.rewards.visual.image_reward import ImageRewardScorer

       return ImageRewardScorer.from_config(cfg, device)


   def register():
       register_refl_scorer("custom_image_reward", build_image_scorer)

Prepare the models, data, and launch-argument array from the :doc:`SD3 recipe </rst_source/examples/embodied/refl>`. Replace ``<ext_dir>`` with the directory containing ``my_refl_ext.py`` and launch training:

.. code-block:: bash

   export PYTHONPATH="<ext_dir>:$PYTHONPATH"
   export RLINF_EXT_MODULE=my_refl_ext
   bash examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
     "${REFL_ARGS[@]}" \
     actor.refl_reward_model.type=custom_image_reward

Each worker imports ``RLINF_EXT_MODULE`` and calls ``register()``. To use your own scorer, change how the builder constructs it.

Add tests for reward shape, frozen parameters, and gradients back to the samples to ``tests/unit_tests/test_diffusion.py``:

.. code-block:: bash

   pytest tests/unit_tests/test_diffusion.py -k scorer

3. Configuration File
~~~~~~~~~~~~~~~~~~~~~

Create a Hydra config under ``examples/embodiment/config/<name>.yaml`` with your model hyperparameters:

.. code-block:: yaml

   model:
     model_type: "your_model_type"
     action_dim: 7
     num_action_chunks: 1
     unnorm_key: your_action_key
     precision: "bf16"
     vocab_size: 32000
     hidden_size: 4096
     image_size: [224, 224]
     is_lora: False
     low_cpu_mem_usage: True
     trust_remote_code: True

If your model needs explicit FSDP wrapping rules, add ``fsdp_config.wrap_policy``. When custom rules are present, RLinf prioritizes them; otherwise it falls back to the model's ``_no_split_modules`` / ``_no_split_names`` and built-in rules:

- ``transformer_layer_cls_to_wrap``: class names of main transformer blocks (such as decoder layers or attention blocks).
- ``module_classes_to_wrap``: class names of additional modules outside the main backbone (such as vision encoders, projectors, or adapters).
- ``no_split_names``: fine-grained wrapping matched against each module's ``_fsdp_wrap_name`` attribute.

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

4. Model Usage and Quick Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After calling ``register_model("your_model_type", build_my_model, category="embodied")``, select ``model_type: your_model_type`` in YAML configs. If your model requires a ``processor`` object, load and attach it inside ``build_my_model``.

Verify locally on CPU that ``get_model`` resolves and instantiates your builder:

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
~~~~~~~~

- **Action post-processing**: ``prepare_actions(...)`` in ``rlinf/envs/action_utils.py`` applies model- and environment-specific action transforms. For a custom ``model_type``, perform model-specific post-processing inside ``predict_action_batch(...)`` so returned ``actions`` are directly executable by the environment.
- **Overriding a built-in model**: Pass ``force=True`` to ``register_model(...)`` to replace an in-tree implementation while keeping its ``model_type``. This reuses existing action processing and training branches as long as your model matches the built-in interface.
- **FSDP wrap policy**: See ``get_fsdp_wrap_policy(...)`` in ``rlinf/hybrid_engines/fsdp/utils.py`` for built-in wrap policies. Configure ``fsdp_config.wrap_policy`` explicitly for custom architectures so transformer layers, vision encoders, projectors, and value heads shard as intended.

Per-Wrapper Mixed Precision (FSDP1)
-----------------------------------

Use ``mixed_precision`` as the global default MP policy. Select wrappers in ``wrappers``, define their MP policies in ``mixed_precision_policies``, and bind the names in ``mixed_precision_rules`` to override that default. This Pi05 example keeps sensitive modules in FP32 while using BF16 elsewhere.

For OpenPI, set ``actor.model.precision: fp32`` for original parameters and ``actor.model.openpi.embed_dtype: bf16`` for backbone activations and residuals.

.. code-block:: yaml

   fsdp_config:
     strategy: fsdp
     sharding_strategy: no_shard
     use_orig_params: true
     mixed_precision:
       param_dtype: bf16
       reduce_dtype: fp32
       buffer_dtype: fp32
       cast_forward_inputs: false
       cast_root_forward_inputs: false
     wrappers:
       projections:
         wrap_names: [action_in_proj, action_out_proj, time_mlp_in, time_mlp_out]
       norms:
         module_classes: [RMSNorm]
       vision_embedding:
         module_names: [img, img.stem]
       vision_mlps:
         module_names: ["img.encoder.layers.*.mlp"]
         wrap_mode: subtree
     mixed_precision_policies:
       fp32:
         param_dtype: fp32
         reduce_dtype: fp32
         buffer_dtype: fp32
         cast_forward_inputs: true
       fp32_preserve_inputs:
         param_dtype: fp32
         reduce_dtype: fp32
         buffer_dtype: fp32
         cast_forward_inputs: false
     mixed_precision_rules:
       projections: fp32
       norms: fp32_preserve_inputs
       vision_embedding: fp32_preserve_inputs
       vision_mlps: fp32
     amp_autocast:
       enabled: false

Choose one selector per wrapper: ``module_names`` matches exact module paths or globs such as ``img.encoder.layers.*.mlp``; ``module_classes`` accepts short or qualified Python class names, using the same resolver as ``wrap_policy.module_classes_to_wrap``; ``wrap_names`` matches existing ``_fsdp_wrap_name`` tags. Pi05 assigns these tags to its projection modules during model initialization.

Set ``wrap_mode`` to control the boundaries:

- ``individual`` (default): wrap each match and retain automatic child wrapping.
- ``subtree``: wrap each match's complete subtree, with no inner wrappers. The example creates one FP32 wrapper for each layer's MLP.
- ``combined``: wrap all matches under their existing common parent, with no inner wrappers. Include all parameters and buffers under that parent.

Each wrapper's managed parameters must share their original dtype. Mixed ``requires_grad`` requires ``use_orig_params: true``. Subtree and combined boundaries must not overlap other explicitly selected wrappers.

Specify all three dtype fields in each named policy. Wrappers without a binding use the global policy; nested wrappers retain their own policies. Both global and named policies default to ``cast_forward_inputs: true``. Set it to ``false`` when the module handles input casting itself, as OpenPI RMSNorm does. When inputs must retain FP32 precision, as with Pi05 actions and timesteps, set global ``cast_root_forward_inputs: false`` to avoid rounding them to BF16 before they reach that wrapper.

``buffer_dtype`` controls floating-point buffers during computation. FSDP saves buffers in their original dtype in full and sharded model checkpoints; ``LOCAL_STATE_DICT`` does not restore their dtype. ``summon_full_params()`` restores parameter precision and does not restore buffer precision.

Retaining Low-Precision Gradients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``keep_low_precision_grads`` defaults to ``false`` in both the global policy and named policies. FSDP converts gradients back to the parameters' original dtype for the optimizer. With ``true``, sharded strategies and FSDP CPU parameter offload prepare gradients in the compute dtype (``param_dtype``, or the original dtype when unset); ``no_shard`` without CPU offload retains the effective reduction dtype. Retaining a smaller dtype reduces gradient storage.

Use ``FP32MasterAdamW`` with FSDP1 and ``no_shard`` to update parameters from retained gradients of a different dtype. Select it for the whole actor:

.. code-block:: yaml

   actor:
     model:
       precision: fp32
     optim:
       use_fp32_master_params: true
     fsdp_config:
       strategy: fsdp
       sharding_strategy: no_shard
       mixed_precision:
         param_dtype: bf16
         reduce_dtype: bf16
         buffer_dtype: fp32
         keep_low_precision_grads: true

Set ``keep_low_precision_grads`` in ``mixed_precision_policies`` to retain gradients for specific wrappers. The optimizer preserves parameter groups and optimizer settings, converts gradients to FP32 during updates, and keeps FP32 optimizer states. BF16/FP16 original parameters also have an FP32 master copy. Use the default ``keep_low_precision_grads: false`` with native AdamW.
