Embodied RL Model (FSDP)
========================

Integrate a PyTorch or Hugging Face Transformers policy into RLinf's FSDP (Fully Sharded Data Parallel) backend for RL rollout and training. By the end, you have a ``BasePolicy`` subclass, a registered model builder, and Hydra and FSDP wrapping configuration.

The work follows three checkpoints. First, implement the policy contract and verify a rollout-to-training round-trip locally on CPU. Next, register the builder and check that ``get_model`` resolves it. Finally, add the Hydra model and FSDP configuration.

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
   * - Verification
     - Direct local CPU round-trip and ``get_model`` check

1. Implement the Policy Contract Locally
----------------------------------------

Create your policy under ``rlinf/models/embodiment/<name>/`` (or in an external package), subclassing ``rlinf.models.embodiment.base_policy.BasePolicy`` and ``torch.nn.Module`` (or a Hugging Face model class). Put ``BasePolicy`` before ``nn.Module`` in the base list if you rely on ``BasePolicy.forward`` directly, because ``nn.Module`` defines its own stub ``forward``.

The minimal Gaussian policy below implements both required methods without external weights:

.. code-block:: python

   from typing import Any

   import torch
   import torch.nn as nn
   from torch.distributions import Normal

   from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


   class TinyGaussianPolicy(BasePolicy, nn.Module):
       def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
           super().__init__()
           self.backbone = nn.Sequential(
               nn.Linear(obs_dim, hidden_dim),
               nn.Tanh(),
               nn.Linear(hidden_dim, hidden_dim),
               nn.Tanh(),
           )
           self.actor_mean = nn.Linear(hidden_dim, action_dim)
           self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
           self.value_head = nn.Linear(hidden_dim, 1)

       def _distribution(self, states: torch.Tensor) -> tuple[Normal, torch.Tensor]:
           feat = self.backbone(states)
           mean = self.actor_mean(feat)
           std = torch.exp(self.actor_logstd).expand_as(mean)
           return Normal(mean, std), self.value_head(feat)

       def default_forward(
           self,
           forward_inputs: dict[str, Any],
           compute_logprobs: bool = True,
           compute_entropy: bool = True,
           compute_values: bool = True,
           **kwargs,
       ) -> dict[str, torch.Tensor]:
           dist, values = self._distribution(forward_inputs["states"])
           outputs: dict[str, torch.Tensor] = {}
           if compute_logprobs:
               outputs["logprobs"] = dist.log_prob(forward_inputs["action"])
           if compute_entropy:
               outputs["entropy"] = dist.entropy()
           if compute_values:
               outputs["values"] = values
           return outputs

       @torch.inference_mode()
       def predict_action_batch(
           self,
           env_obs: dict[str, Any],
           mode: str = "train",
           calculate_values: bool = True,
           **kwargs,
       ) -> tuple[torch.Tensor, dict[str, Any]]:
           states = env_obs["states"]
           dist, values = self._distribution(states)
           action = dist.sample() if mode == "train" else dist.mean
           prev_logprobs = dist.log_prob(action)
           prev_values = values if calculate_values else torch.zeros_like(values)
           result = {
               "prev_logprobs": prev_logprobs,
               "prev_values": prev_values,
               "forward_inputs": {"states": states, "action": action},
           }
           return action, result

Each public method enters the rollout-training loop in this order:

- ``predict_action_batch(env_obs, mode="train"|"eval", calculate_values=True, **kwargs)`` runs during rollout and evaluation. It receives ``env_obs`` (which for vision-language-action policies such as ``rlinf/models/embodiment/openpi/tasks/rl.py`` also carries ``main_images``, ``wrist_images``, ``extra_view_images``, and ``task_descriptions``) and returns ``(actions, result)``. ``actions`` goes to the environment, while ``result`` carries ``prev_logprobs``, ``prev_values``, and ``forward_inputs`` cached for training.
- ``forward(forward_type=ForwardType.DEFAULT, **kwargs)`` is the dispatch entry point called by training workers. ``BasePolicy.forward`` routes ``ForwardType.DEFAULT`` to ``default_forward`` and ``ForwardType.REFL`` to ``refl_forward``. Override ``forward`` if your policy also handles ``ForwardType.SFT`` or ``ForwardType.SAC``.
- ``default_forward(forward_inputs, **kwargs)`` runs under autograd during PPO or GRPO updates. It reads the tensors cached in ``forward_inputs`` (for iterative or diffusion policies, keys such as ``chains``, ``denoise_inds``, ``tokenized_prompt``, and ``model_action``) and returns ``logprobs``, ``values``, and ``entropy`` for the configured loss.

You can pick any keys inside ``forward_inputs`` as long as ``predict_action_batch`` stores everything ``default_forward`` needs. Perform model-specific action post-processing inside ``predict_action_batch`` so returned ``actions`` are directly executable by the environment instead of adding a new branch to ``prepare_actions`` in ``rlinf/envs/action_utils.py``.

Before registering the model, run a local CPU round-trip check that verifies output shapes, log-probability parity between rollout and training, and gradient flow:

.. code-block:: python

   torch.manual_seed(0)
   policy = TinyGaussianPolicy(obs_dim=8, action_dim=4)
   actions, result = policy.predict_action_batch({"states": torch.randn(2, 8)})
   out = policy(forward_type=ForwardType.DEFAULT, forward_inputs=result["forward_inputs"])

   assert actions.shape == (2, 4)
   assert out["logprobs"].shape == (2, 4)
   assert out["values"].shape == (2, 1)
   assert out["entropy"].shape == (2, 4)
   assert torch.allclose(out["logprobs"], result["prev_logprobs"], atol=1e-6)

   loss = -(out["logprobs"].mean() + 0.01 * out["entropy"].mean()) + out["values"].pow(2).mean()
   loss.backward()
   assert policy.actor_mean.weight.grad is not None
   print("Local check passed:", {k: tuple(v.shape) for k, v in out.items()})

Expected output:

.. code-block:: text

   Local check passed: {'logprobs': (2, 4), 'entropy': (2, 4), 'values': (2, 1)}

.. _out-of-tree-extensions:

2. Register the Model Builder
-----------------------------

Register a builder with ``register_model``. The builder takes ``(cfg, torch_dtype)`` and returns the instantiated model; for Hugging Face checkpoints, call ``from_pretrained(cfg.model_path, torch_dtype=torch_dtype, ...)`` and attach any ``processor`` inside the builder.

Calling ``register_model`` registers the string in ``SupportedModel`` so Hydra validation accepts it, and maps ``model_type`` to your builder in ``get_model``. Pass ``force=True`` to replace a built-in implementation while keeping its ``model_type``. Choose how you want to host your code:

.. tab-set::

   .. tab-item:: In-Tree (Inside RLinf)
      :sync: ext-in-tree

      Register the builder inside ``rlinf/models/__init__.py`` (so every worker imports it automatically), then check locally on CPU with ``load_to_device: False``:

      .. code-block:: python

         from omegaconf import OmegaConf

         from rlinf.config import SupportedModel
         from rlinf.models import get_model, register_model


         def build_my_model(cfg, torch_dtype):
             return TinyGaussianPolicy(
                 obs_dim=cfg.obs_dim,
                 action_dim=cfg.action_dim,
                 hidden_dim=cfg.get("hidden_dim", 64),
             ).to(dtype=torch_dtype)


         register_model("tiny_gaussian", build_my_model, category="embodied")

         assert SupportedModel("tiny_gaussian").value == "tiny_gaussian"
         cfg = OmegaConf.create(
             {
                 "model_type": "tiny_gaussian",
                 "obs_dim": 8,
                 "action_dim": 4,
                 "hidden_dim": 64,
                 "precision": "fp32",
                 "is_lora": False,
                 "load_to_device": False,
             }
         )
         model = get_model(cfg)
         assert isinstance(model, TinyGaussianPolicy)

   .. tab-item:: Out-of-Tree (External Repo)
      :sync: ext-out-of-tree

      You do not need to fork RLinf. Because Ray launches each rollout and actor worker in its own Python process, calling ``register_model`` only in your driver script does not reach workers. Put your registration inside a ``register()`` hook in your own package and export ``RLINF_EXT_MODULE`` before launching (see ``ClusterEnvVar.EXT_MODULE`` in ``rlinf/scheduler/cluster/cluster.py``):

      .. code-block:: python

         # my_pkg/rlinf_ext.py
         from rlinf.models import register_model


         def build_my_model(cfg, torch_dtype):
             from my_pkg.policy import TinyGaussianPolicy

             return TinyGaussianPolicy(
                 obs_dim=cfg.obs_dim,
                 action_dim=cfg.action_dim,
                 hidden_dim=cfg.get("hidden_dim", 64),
             ).to(dtype=torch_dtype)


         def register():
             register_model("tiny_gaussian", build_my_model, category="embodied")

      .. code-block:: bash

         export RLINF_EXT_MODULE=my_pkg.rlinf_ext

      RLinf imports this module and calls ``register()`` automatically in every worker process before loading configs or models.

      .. note::

         Standalone checkpoint tools such as ``python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf`` also load ``RLINF_EXT_MODULE`` before calling ``get_model``. Export it before running them when your ``model_type`` lives in an external package.

3. Add Hydra Model and FSDP Configuration
-----------------------------------------

Create a Hydra config under ``examples/embodiment/config/<name>.yaml`` (or a reusable model fragment under ``examples/embodiment/config/model/``) that selects your registered ``model_type`` and declares its hyperparameters:

.. code-block:: yaml

   actor:
     model:
       model_type: "tiny_gaussian"
       obs_dim: 8
       action_dim: 4
       hidden_dim: 64
       precision: "bf16"
       is_lora: False

When your architecture needs explicit FSDP sharding boundaries, set ``fsdp_config.wrap_policy``. RLinf applies custom rules first and otherwise falls back to ``_no_split_modules``, ``_no_split_names``, and the built-in rules in ``get_fsdp_wrap_policy`` (``rlinf/hybrid_engines/fsdp/utils.py``):

.. code-block:: yaml

   actor:
     fsdp_config:
       wrap_policy:
         transformer_layer_cls_to_wrap:
           - "MyDecoderLayer"
         module_classes_to_wrap:
           - "MyVisionTower"
           - "MyProjector"
         no_split_names:
           - "value_head"

- ``transformer_layer_cls_to_wrap``: class names of main transformer blocks such as decoder or attention layers.
- ``module_classes_to_wrap``: class names of extra modules outside the main backbone such as vision encoders, projectors, or adapters.
- ``no_split_names``: fine-grained wrapping matched against each module's ``_fsdp_wrap_name`` attribute.

.. _refl-model-interface:

Train from Differentiable Rewards
---------------------------------

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
