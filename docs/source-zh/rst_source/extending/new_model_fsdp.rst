具身 RL 模型（FSDP）
====================

将 PyTorch 或 Hugging Face Transformers 模型接入 RLinf 的 FSDP（Fully Sharded Data Parallel，全分片数据并行）后端，用于强化学习 rollout 与训练。完成后，你会得到一个 ``BasePolicy`` 子类、注册好的模型构建函数，以及 Hydra 与 FSDP 包装配置。

接入过程分为三个检查点：先在本地实现 policy 接口并跑通 rollout 到训练的 CPU 闭环；接着注册构建函数，确认 ``get_model`` 能解析模型；最后补充 Hydra 模型与 FSDP 配置。

接入速览
--------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 项目
     - 说明
   * - 新增或修改的文件
     - ``rlinf/models/embodiment/<name>/``\ （或外部包）与 ``examples/embodiment/config/<name>.yaml``
   * - 扩展点
     - ``register_model`` 与 ``RLINF_EXT_MODULE``\ （见 :ref:`out-of-tree-extensions`\ ）
   * - 必选方法
     - 继承 ``BasePolicy``，实现 ``default_forward`` 与 ``predict_action_batch``
   * - 验证方式
     - 在本地 CPU 上直接跑通前向闭环与 ``get_model`` 解析检查

1. 在本地实现 Policy 接口
-------------------------

在 ``rlinf/models/embodiment/<name>/``\ （或外部包）中新建模型类，继承 ``rlinf.models.embodiment.base_policy.BasePolicy`` 和 ``torch.nn.Module``\ （或对应的 Hugging Face 模型基类）。如果直接复用 ``BasePolicy.forward`` 分发逻辑，请把 ``BasePolicy`` 写在 ``nn.Module`` 前面，避免被 ``nn.Module`` 默认的 ``forward`` 占位实现覆盖。

下面的最小高斯 policy 实现不依赖外部权重，可以直接在 CPU 上运行：

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

三个公开方法在 rollout 与训练闭环中的调用顺序如下：

- ``predict_action_batch(env_obs, mode="train"|"eval", calculate_values=True, **kwargs)``：rollout 与评估阶段调用。输入环境观测 ``env_obs``\ （对于视觉语言动作模型，如 ``rlinf/models/embodiment/openpi/tasks/rl.py``，通常还会包含 ``main_images``、``wrist_images``、``extra_view_images`` 和 ``task_descriptions``\ ），返回 ``(actions, result)``。``actions`` 直接送入环境执行，``result`` 包含旧对数概率 ``prev_logprobs``、旧价值估计 ``prev_values``，以及供训练复用的 ``forward_inputs``。
- ``forward(forward_type=ForwardType.DEFAULT, **kwargs)``：训练 worker 调用的统一入口。``BasePolicy.forward`` 将 ``ForwardType.DEFAULT`` 分发到 ``default_forward``，将 ``ForwardType.REFL`` 分发到 ``refl_forward``。如果 policy 还需要支持 ``ForwardType.SFT`` 或 ``ForwardType.SAC``，可重写 ``forward``。
- ``default_forward(forward_inputs, **kwargs)``：PPO 或 GRPO 更新时在梯度模式下调用。它读取 ``forward_inputs`` 中缓存的张量（扩散或迭代采样模型常见 ``chains``、``denoise_inds``、``tokenized_prompt`` 与 ``model_action`` 等键），返回所选损失函数所需的 ``logprobs``、``values`` 和 ``entropy``。

自定义模型可以自行约定 ``forward_inputs`` 中的键名，只要 ``predict_action_batch`` 缓存的内容足以让 ``default_forward`` 重建训练输入即可。模型专属的动作后处理放在 ``predict_action_batch`` 内完成，让 ``actions`` 直接满足环境格式，无需在 ``rlinf/envs/action_utils.py`` 的 ``prepare_actions`` 中增加新分支。

注册模型前，先在本地 CPU 上验证输出 shape、rollout 与训练重算对数概率的一致性以及反向传播梯度：

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

预期输出：

.. code-block:: text

   Local check passed: {'logprobs': (2, 4), 'entropy': (2, 4), 'values': (2, 1)}

.. _out-of-tree-extensions:

2. 注册模型构建函数
-------------------

使用 ``register_model`` 注册模型构建函数。构建函数接收 ``(cfg, torch_dtype)`` 并返回模型实例；加载 Hugging Face 权重时，在构建函数内调用 ``from_pretrained(cfg.model_path, torch_dtype=torch_dtype, ...)`` 并挂载所需的 ``processor``。

调用 ``register_model`` 会同步将模型名加入 ``SupportedModel``\ （使 Hydra 校验通过），并在 ``get_model`` 中绑定构建函数；传入 ``force=True`` 可覆盖主仓内置实现并沿用原有 ``model_type``。按代码存放位置选择扩展方式：

.. tab-set::

   .. tab-item:: 主仓库内注册（In-Tree）
      :sync: ext-in-tree

      在 ``rlinf/models/__init__.py`` 中注册构建函数（所有 worker 导入时自动生效），并在本地 CPU 上配合 ``load_to_device: False`` 验证解析：

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

   .. tab-item:: 外部仓库注册（Out-of-Tree）
      :sync: ext-out-of-tree

      无需 fork 或修改 RLinf 源码。由于 Ray 会在独立 Python 进程中拉起各个 rollout 和 actor worker，仅在主进程入口脚本里调用 ``register_model`` 不会同步到 worker 进程。把注册逻辑放进外部包的 ``register()`` 函数，并在启动前导出 ``RLINF_EXT_MODULE``\ （环境变量定义见 ``rlinf/scheduler/cluster/cluster.py`` 中的 ``ClusterEnvVar.EXT_MODULE``\ ）：

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

      RLinf 会在每个 worker 进程初始化时自动导入该模块并调用 ``register()``，在读取配置和构建模型之前完成注册。

      .. note::

         独立运行的 checkpoint 转换工具（例如 ``python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf``\ ）在调用 ``get_model`` 前同样会加载 ``RLINF_EXT_MODULE``。如果 ``model_type`` 仅在外部包中注册，运行转换工具前也需要设置这个环境变量。

3. 添加 Hydra 模型与 FSDP 配置
------------------------------

在 ``examples/embodiment/config/<name>.yaml``\ （或 ``examples/embodiment/config/model/`` 下的模型片段）中声明注册好的 ``model_type`` 与超参数：

.. code-block:: yaml

   actor:
     model:
       model_type: "tiny_gaussian"
       obs_dim: 8
       action_dim: 4
       hidden_dim: 64
       precision: "bf16"
       is_lora: False

模型在 FSDP 下需要显式指定切分边界时，补充 ``fsdp_config.wrap_policy``。配置了自定义规则时 RLinf 会优先使用，否则回退到模型自身的 ``_no_split_modules``、``_no_split_names`` 以及 ``rlinf/hybrid_engines/fsdp/utils.py`` 中 ``get_fsdp_wrap_policy`` 的内置规则：

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

- ``transformer_layer_cls_to_wrap``：按类名指定主干 transformer block（如 decoder layer、attention block）。
- ``module_classes_to_wrap``：按类名指定主干之外需要单独包装的模块（如 vision encoder、projector、adapter）。
- ``no_split_names``：按模块上的 ``_fsdp_wrap_name`` 属性做细粒度包装。

.. _refl-model-interface:

通过可微奖励训练
----------------

ReFL 接入分为两部分：模型生成可微样本，scorer 为每个样本打分。奖励梯度经过样本回传到模型的可训练参数。

模型实现 ``refl_forward(forward_inputs, xt_source="rollout", **kwargs)``，返回包含 ``samples`` 的 dict，样本 tensor 的首维为 batch。模型负责状态采样、时间单位、加噪和解码，样本布局与 scorer 保持一致。``BasePolicy.forward`` 根据 ``ForwardType.REFL`` 调用这个方法。重写 ``forward`` 时保留这一分支。

actor 从 ``algorithm.refl_xt_source`` 读取 ``xt_source``。返回值中的 ``forward_inputs`` 用作 scorer 上下文，默认沿用原始输入。``scorer(samples, forward_inputs)`` 返回 shape 为 ``[B]`` 的奖励，数值越大越好。冻结 scorer 的参数，并保留输入样本的梯度。

SD3 的输入为 ``forward_inputs={"prompts": prompts}``，其中 ``prompts`` 是字符串列表。模型用当前权重采样，并将原始提示词传给 ImageReward scorer 分词。传入已有状态时，``refl_inputs`` 提供 ``x_t`` 和 ``timesteps``，``forward_inputs`` 提供 ``prompt_embeds`` 与 ``pooled_prompt_embeds``。SD3 的时间范围为 ``[0, 1]``，0 对应数据，1 对应噪声。

使用 ``register_refl_scorer`` 注册 scorer 的 builder，再通过 ``actor.refl_reward_model.type`` 选择。builder 接收完整的 ``actor.refl_reward_model`` 配置和 actor device，返回 scorer。将下面的扩展模块保存为 ``my_refl_ext.py``，注册 ImageReward scorer：

.. code-block:: python

   # my_refl_ext.py
   from rlinf.algorithms.refl_scorers import register_refl_scorer


   def build_image_scorer(cfg, device):
       from rlinf.envs.sim.diffusion.rewards.visual.image_reward import ImageRewardScorer

       return ImageRewardScorer.from_config(cfg, device)


   def register():
       register_refl_scorer("custom_image_reward", build_image_scorer)

按 :doc:`SD3 示例 </rst_source/examples/embodied/refl>` 准备模型、数据和启动参数数组。将 ``<ext_dir>`` 替换为 ``my_refl_ext.py`` 所在目录，再启动训练：

.. code-block:: bash

   export PYTHONPATH="<ext_dir>:$PYTHONPATH"
   export RLINF_EXT_MODULE=my_refl_ext
   bash examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
     "${REFL_ARGS[@]}" \
     actor.refl_reward_model.type=custom_image_reward

每个 worker 导入 ``RLINF_EXT_MODULE`` 并调用 ``register()``。更换打分器时，修改 builder 中的 scorer 构造逻辑。

按 Wrapper 设置混合精度（FSDP1）
--------------------------------

``mixed_precision`` 是全局默认 MP 策略。在 ``wrappers`` 中选择模块，在 ``mixed_precision_policies`` 中定义 MP 策略，用 ``mixed_precision_rules`` 将两者的名称配对，覆盖对应 wrapper 的全局默认策略。下面的 Pi05 示例让精度敏感模块使用 FP32，其他部分使用 BF16。

OpenPI 设置 ``actor.model.precision: fp32``，让原始参数使用 FP32；设置 ``actor.model.openpi.embed_dtype: bf16``，让主干激活和残差使用 BF16。

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

每个 wrapper 选择一种匹配方式：``module_names`` 支持完整模块路径和 ``img.encoder.layers.*.mlp`` 等 glob；``module_classes`` 支持短类名和完整限定名，与 ``wrap_policy.module_classes_to_wrap`` 共用解析规则；``wrap_names`` 匹配已有的 ``_fsdp_wrap_name`` 标签。Pi05 在模型初始化时为 projection 模块设置这些标签。

通过 ``wrap_mode`` 控制包装边界：

- ``individual``\ （默认）：每个命中模块分别包装，保留自动子包装。
- ``subtree``：每个命中模块的完整子树分别包装，内部不再创建 wrapper。本例为每层 MLP 创建一个 FP32 wrapper。
- ``combined``：将所有命中模块合到现有共同父模块，内部不再创建 wrapper；需要选中该父模块下的所有参数和 buffer。

每个 wrapper 管理的参数必须具有一致的原始 dtype，混合 ``requires_grad`` 需要设置 ``use_orig_params: true``。subtree 和 combined 边界不能与其他显式选择的 wrapper 重叠。

命名策略必须填写三个 dtype 字段。未绑定规则的 wrapper 使用全局策略，嵌套 wrapper 保留各自策略。全局和命名策略的 ``cast_forward_inputs`` 默认均为 ``true``。模块自行处理输入转换时（例如 OpenPI RMSNorm），将其设为 ``false``。输入需要保留 FP32 精度时（例如 Pi05 的 action 和 timestep），将全局 ``cast_root_forward_inputs`` 设为 ``false``，避免输入到达该 wrapper 前被舍入到 BF16。

``buffer_dtype`` 控制浮点 buffer 的计算 dtype。FSDP 在完整和分片模型 checkpoint 中按原始 dtype 保存 buffer；``LOCAL_STATE_DICT`` 不做该 dtype 恢复。``summon_full_params()`` 恢复参数精度，不恢复 buffer 精度。

保留低精度梯度
~~~~~~~~~~~~~~

全局策略和命名策略中的 ``keep_low_precision_grads`` 默认均为 ``false``。FSDP 将梯度转回参数的原始 dtype，供优化器更新。设为 ``true`` 时，分片策略和 FSDP CPU 参数 offload 使用计算 dtype 准备梯度（``param_dtype``，未设置时为原始 dtype）；不使用 CPU offload 的 ``no_shard`` 保留实际归约 dtype。保留较小的 dtype 可以减少梯度存储。

使用 FSDP1 的 ``no_shard`` 和 ``FP32MasterAdamW``，可以用保留的不同 dtype 梯度更新参数。为整个 actor 选择该优化器：

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

在 ``mixed_precision_policies`` 中设置 ``keep_low_precision_grads``，可以为指定 wrapper 保留梯度。优化器沿用参数分组和优化器配置，在更新时将梯度转为 FP32，并保持 FP32 优化器状态。原始 BF16/FP16 参数还会有一份 FP32 master weights。使用原生 AdamW 时，保持默认的 ``keep_low_precision_grads: false``。
