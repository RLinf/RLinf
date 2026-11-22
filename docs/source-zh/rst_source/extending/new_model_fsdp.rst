接入 FSDP 模型
==============

将 PyTorch 或 Hugging Face Transformers 模型接入 RLinf 的 FSDP（Fully Sharded Data Parallel，全分片数据并行）后端，用于强化学习训练与 rollout。下面按 OpenVLA 的具身策略模式演示完整步骤。

接入速览
--------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 项目
     - 说明
   * - 新增或修改的文件
     - ``rlinf/models/embodiment/<name>/``\ （或外部包）以及 ``examples/embodiment/config/<name>.yaml``
   * - 扩展点
     - ``register_model`` 与 ``RLINF_EXT_MODULE``\ （见 :ref:`out-of-tree-extensions`\ ）
   * - 必选方法
     - 继承 ``BasePolicy``，实现 ``default_forward`` 与 ``predict_action_batch``
   * - 快速验证
     - 在 CPU 上调用 ``get_model(cfg)`` 实例化模型，并在 ``tests/unit_tests/`` 中补充单元测试

逐步实现
--------

1. 模型注册
~~~~~~~~~~~

在 ``rlinf/models/__init__.py``\ （主仓内）或外部扩展模块中调用 ``register_model`` 注册模型构建函数。

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

这次注册会完成两件事：

- 将 ``model_type`` 写入 ``SupportedModel``，使 Hydra 配置校验通过。
- 让 ``rlinf.models.get_model(cfg)`` 可以构建你的模型实例。

外部包的分布式注册
^^^^^^^^^^^^^^^^^^

Ray worker 运行在独立的 Python 进程中，仅在主进程入口脚本调用 ``register_model`` 无法同步到 rollout 或 actor worker。请把注册逻辑放进外部模块的 ``register()`` 函数，并在启动前导出 ``RLINF_EXT_MODULE``\ （模块结构见 :ref:`out-of-tree-extensions`，环境变量格式见 ``rlinf/scheduler/cluster/cluster.py`` 中的 ``ClusterEnvVar.EXT_MODULE``\ ）：

.. code-block:: bash

   export RLINF_EXT_MODULE=my_pkg.rlinf_ext

.. note::

   独立运行的 checkpoint 转换工具（例如 ``python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf``\ ）在调用 ``get_model`` 前同样会加载 ``RLINF_EXT_MODULE``。如果 ``model_type`` 仅在外部包中注册，运行转换工具前也需要设置该环境变量。

2. 模型实现
~~~~~~~~~~~

在 ``rlinf/models/embodiment/<name>/``\ （或外部包）中创建模型类，继承 ``rlinf.models.embodiment.base_policy.BasePolicy`` 及对应的 Hugging Face 基类。实现 ``default_forward`` 和 ``predict_action_batch``；``BasePolicy.forward`` 会自动将 ``ForwardType.DEFAULT`` 分发到 ``default_forward``，将 ``ForwardType.REFL`` 分发到 ``refl_forward``。``predict_action_batch`` 封装动作生成、解码和可选的价值估计，将 RL 相关逻辑收敛在模型内部。

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

完整参考实现见 ``rlinf/models/embodiment/openpi/tasks/rl.py``。三个接口在 rollout 与训练闭环中的分工如下：

- ``forward(forward_type=ForwardType.DEFAULT, **kwargs)``：统一分发入口。``BasePolicy.forward`` 将 ``ForwardType.DEFAULT`` 路由到 ``default_forward``，将 ``ForwardType.REFL`` 路由到 ``refl_forward``。如果策略还需要支持 ``ForwardType.SFT`` 或 ``ForwardType.SAC`` 分支，可重写 ``forward``。
- ``default_forward(forward_inputs, **kwargs)``：训练阶段根据 rollout 缓存的张量重新计算 logprob、value 和 entropy。参考 OpenPI，``forward_inputs`` 通常包含：

  - ``chains``：扩散或迭代采样产生的动作链，供训练阶段重算对数概率。
  - ``denoise_inds``：每个样本对应的去噪步索引。
  - ``tokenized_prompt`` / ``tokenized_prompt_mask``：文本 prompt token 及其 mask。
  - ``action``：实际交给环境执行的动作，通常展平为 ``[B, ...]``。
  - ``model_action``：输出变换前的模型原始动作。
  - 观测字段：如 ``observation/image``、``observation/state``、``observation/wrist_image`` 等，用于重建模型输入。

  ``default_forward`` 返回包含损失函数所需字段的字典：

  - ``logprobs``：当前策略下采样动作的重算对数概率。
  - ``values``：value head 输出的状态价值。
  - ``entropy``：用于 PPO 或 RL 正则项的策略熵。
- ``predict_action_batch(env_obs, mode="train"|"eval", compute_values=True, **kwargs)``：rollout 与评估阶段调用，输入环境观测并输出可执行动作。参考 OpenPI，``env_obs`` 通常包含：

  - ``main_images``：主视角图像。
  - ``wrist_images``：腕部相机图像（可选）。
  - ``extra_view_images``：额外视角图像（可选）。
  - ``states``：低维机器人状态。
  - ``task_descriptions``：文本任务描述。

  ``predict_action_batch`` 返回 ``(actions, result)``：

  - ``actions``：直接交给环境执行的动作张量。
  - ``result["prev_logprobs"]``：rollout 阶段记录的旧策略对数概率。
  - ``result["prev_values"]``：rollout 阶段记录的旧价值估计。
  - ``result["forward_inputs"]``：训练阶段传回 ``default_forward`` 的缓存字段。

自定义模型无需照搬 OpenPI 的键名，只要满足两点即可：``predict_action_batch`` 缓存的 ``forward_inputs`` 足以让 ``default_forward`` 重建训练所需输入，且 ``default_forward`` 返回所选损失函数依赖的字段（如 ``logprobs``、``values`` 和 ``entropy``）。

.. _refl-model-interface:

通过可微奖励训练
^^^^^^^^^^^^^^^^

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

在 ``tests/unit_tests/test_diffusion.py`` 中测试奖励 shape、参数冻结和从奖励到样本的梯度：

.. code-block:: bash

   pytest tests/unit_tests/test_diffusion.py -k scorer

3. 配置文件
~~~~~~~~~~~

在 ``examples/embodiment/config/<name>.yaml`` 下创建 Hydra 配置文件，声明模型超参数：

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

如果模型在 FSDP 下需要显式指定 wrapping 规则，在配置中补充 ``fsdp_config.wrap_policy``。配置了自定义规则时 RLinf 会优先使用；否则回退到模型自身的 ``_no_split_modules`` / ``_no_split_names`` 及内置规则：

- ``transformer_layer_cls_to_wrap``：按类名指定主干 transformer block（如 decoder layer、attention block）。
- ``module_classes_to_wrap``：按类名指定主干之外需要单独包装的模块（如 vision encoder、projector、adapter）。
- ``no_split_names``：按模块上的 ``_fsdp_wrap_name`` 属性做细粒度包装。

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

4. 模型使用与快速验证
~~~~~~~~~~~~~~~~~~~~~

调用 ``register_model("your_model_type", build_my_model, category="embodied")`` 后，即可在 YAML 配置中填写 ``model_type: your_model_type``。如果模型依赖 ``processor``，建议在 ``build_my_model`` 中加载并挂载到模型实例上。

在本地 CPU 上直接验证 ``get_model`` 能否解析并实例化你的构建函数：

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

5. 注意事项
~~~~~~~~~~~

- **动作后处理**：``rlinf/envs/action_utils.py`` 中的 ``prepare_actions(...)`` 会按 ``env_type`` 和 ``model_type`` 转换动作。接入全新的自定义 ``model_type`` 时，建议在 ``predict_action_batch(...)`` 中直接输出环境可执行的动作格式，避免在环境侧增加新的模型分支。
- **覆盖内置模型**：向 ``register_model(...)`` 传入 ``force=True`` 可覆盖内置模型实现并沿用原有 ``model_type``。只要接口与内置实现一致，就能直接复用现有的动作处理和训练分支。
- **FSDP wrap policy**：内置策略见 ``rlinf/hybrid_engines/fsdp/utils.py`` 中的 ``get_fsdp_wrap_policy(...)``。接入自定义结构时建议显式配置 ``fsdp_config.wrap_policy``，确保 transformer layer、vision encoder、projector 和 value head 按预期切分。

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
