Model Adaptation
================

Use the existing model factory and worker contracts to add a model to RLinf.
This page maps the registered model families to their implementations, then
defines the integration pattern used by the LingBot-VLA 2.0 example. Choose the
model's role first: an action policy, a language model, a diffusion model and a
reward model enter different worker paths.

Existing Implementations
------------------------

The source audit on 2026-09-22 found 39 distinct names in ``SupportedModel``,
including the new ``lingbotvlav2`` entry. Registration means that a name can be
recognized; the builder, algorithm and environment must also support the chosen
workflow. In particular, MolmoAct2 currently supports evaluation rollouts, and
``cma`` is a legacy name without a built-in factory binding.

Most VLA integrations combine an RLinf adapter with an installed model package
or source fork. Some also carry selected model or transform code inside RLinf.
The common contract is the factory and policy interface, not the location of
every backbone implementation.

.. list-table:: Model families and source boundaries
   :header-rows: 1
   :widths: 32 36 32

   * - Registered names
     - RLinf implementation
     - Model dependency / role
   * - ``openvla``, ``openvla_oft``
     - ``embodiment/openvla``, ``embodiment/openvla_oft``, ``embodiment/prismatic``
     - External ``prismatic`` plus local RL/model adaptations.
   * - ``openpi``, ``openpi_rlinf``
     - ``embodiment/openpi``, ``embodiment/openpi_rlinf``
     - External ``openpi``; the latter includes more local model modules.
   * - ``pi0_fast``
     - ``embodiment/pi0_fast``
     - External LeRobot ``PI0FastPolicy`` with RLinf rollout/replay adaptation.
   * - ``gr00t``, ``gr00t_n1d6``, ``gr00t_n1d7``
     - ``embodiment/gr00t``
     - Version-specific ``gr00t`` source and adapters.
   * - ``dexbotic_pi``, ``dexbotic_dm0``
     - ``embodiment/dexbotic_pi``, ``embodiment/dexbotic_dm0``
     - External ``dexbotic`` plus RL policy subclasses.
   * - ``lingbotvla``, ``lingbotvlav2``
     - ``embodiment/lingbotvla``, ``embodiment/lingbotvlav2``
     - External ``lingbotvla``; isolate V1 and V2 environments.
   * - ``starvla``, ``abot_m0``, ``evo1``
     - Matching directories under ``embodiment/``
     - External starVLA, ABot and Evo-1 source adapters.
   * - ``dreamzero``, ``fastwam``, ``cosmos3``
     - Matching directories under ``embodiment/``
     - External ``groot``, ``fastwam`` and ``cosmos_framework``.
   * - ``molmoact2``
     - ``embodiment/molmoact2``
     - RLinf LeRobot fork; evaluation-only policy contract.
   * - ``mlp_policy``, ``rlt_mlp_policy``, ``rlt_td3_mlp_policy``
     - ``embodiment/mlp_policy``
     - RLinf-owned models with algorithm-specific configuration.
   * - ``cnn_policy``, ``flow_policy``
     - ``embodiment/cnn_policy``, ``embodiment/flow_policy``
     - RLinf-owned policies using standard PyTorch components.
   * - ``cfg_model``, ``recap_value_model``, ``steam_value_model``
     - ``embodiment/openpi_cfg``, ``embodiment/value_model``
     - OpenPI-derived policies/value models with distinct training roles.
   * - ``sd3``, ``wan22_ti2v_5b``
     - ``diffusion/sd3``, ``diffusion/wan``
     - Diffusers pipelines and RLinf diffusion wrappers.
   * - ``qwen2.5``, ``qwen2.5_vl``, ``qwen3``, ``qwen3_vl``
     - FSDP model manager and configured language-model backends
     - Transformers / backend implementations; language, multimodal or SFT role.
   * - ``qwen3_moe``, ``qwen3_vl_moe``, ``deepseek_v3``, ``glm4_moe_lite``
     - Configured language-model backend
     - Backend-specific model support; not embodied ``BasePolicy`` builders.
   * - ``resnet``
     - ``embodiment/reward``
     - Separate reward-model registry, using torchvision.
   * - ``cma``
     - Legacy ``SupportedModel`` entry
     - No current built-in ``rlinf.models.get_model`` binding; do not select it as a working policy recipe.

Implementation paths in the table are relative to ``rlinf/models/``. The source
of truth is ``rlinf/config.py``, ``rlinf/models/__init__.py`` and the corresponding
model installer in ``requirements/install.sh``. These entries describe source
integration, not a claim that every model/environment combination has passed a
GPU test in this change.

Action Policy Contract
----------------------

Build an action policy through ``rlinf.models.get_model(cfg)`` and its registered
``get_model(cfg, torch_dtype)`` builder. Keep optional model imports inside that
builder. The returned ``BasePolicy`` exposes ``predict_action_batch`` for rollout
and ``default_forward`` via ``ForwardType`` for actor scoring. Unsupported modes
must fail explicitly; action prediction alone does not imply PPO support.

Keep the backbone, configuration, model transforms and kernels in the model
package. The RLinf adapter owns environment I/O mapping, RL sampling, likelihood
replay, trainable scope and value heads. RLinf workers own FSDP, optimizers,
checkpoint resume, placement and weight sync.

Installation and Validation
---------------------------

Use ``requirements/install.sh`` and its shared dependency helpers. Pin new model
dependencies to a package version or source revision. If required hooks are
unpublished, version the compatibility patch and hashes, retain source/license
notices, and reject conflicting user edits. V2 uses
``requirements/embodied/models/lingbotvlav2/source.json`` and ``ppo-compat.patch``;
Docker and CI use the same installer. Existing models' less strict source pins
need separate compatibility checks before changing them.

Verify checkpoint loading, preprocessing, rollout/actor likelihood agreement,
finite gradients, intended trainable parameters, optimizer updates, weight sync
and resume. Standalone evaluation needs a complete ``rollout.model``; upstream
deployment export needs its own reload test. Keep EN/ZH instructions and
Docker/CI aligned and report which checks ran.

See :doc:`new_model_fsdp` for worker integration and
:doc:`../examples/embodied/lingbotvlav2` for V2 installation, PPO and evaluation.
