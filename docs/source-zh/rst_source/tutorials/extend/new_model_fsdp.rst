使用 FSDP+HuggingFace 添加新模型
========================================

本文档重点介绍如何使用 HuggingFace Transformers 库与 PyTorch FSDP（Fully Sharded Data Parallel，全分片数据并行）  
来训练和生成模型。它支持 HuggingFace 中实现的任意模型，只要兼容 PyTorch 即可。  
作为示例，本节将提供一个逐步的操作流程，展示如何按照 OpenVLA 模式将一个新的 HuggingFace 模型集成到 RLinf 中。  

前置条件
-------------

* 熟悉 **HuggingFace Transformers 库**  
* 理解 **RLinf** 框架架构  
* 掌握 **PyTorch** 与分布式训练知识  

逐步实现
---------------------------

1. 模型注册
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

现在推荐通过注册模型 builder 的方式接入 RLinf。对于“其他仓库依赖 RLinf”这种场景，这也是最推荐的接入方式。

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

这一次注册会同时完成两件事：

- 让 ``model_type`` 通过 RLinf 的配置校验
- 让 ``rlinf.models.get_model(cfg)`` 能正确构建你的模型

作为依赖库使用时的分布式注册（``RLINF_EXT_MODULE``）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

当 RLinf 以第三方依赖形式集成到你的工程里、并由 **Ray** 拉起分布式 Worker 时，
仅在主进程（driver）里执行 ``register_model(...)`` 等注册代码，**不会** 自动同步到
各个 Worker 进程：Worker 是独立的 Python 进程，不会重复执行你在入口脚本里的注册逻辑。
因此，在 rollout、训练等实际跑在 Worker 上的路径里，模型可能仍未注册，导致构建或
查找失败。

为此，请通过环境变量 ``RLINF_EXT_MODULE`` 指定一个 **扩展模块** 的 import 路径。
RLinf 在 **每个 Worker 初始化时** 会自动 ``import`` 该模块，并在存在时调用其
``register()``，从而在 Worker 侧完成与主进程一致的模型（或其它扩展）注册，无需改
RLinf 源码。

扩展模块需实现 ``register()``，在其中调用 ``register_model`` 等逻辑。环境变量写法与
说明见 ``rlinf/scheduler/cluster/cluster.py`` 中 ``ClusterEnvVar.EXT_MODULE`` 的文档字符串，
例如：

.. code-block:: bash

   export RLINF_EXT_MODULE=rlinf_ext
   # 或使用完整包路径，例如：
   export RLINF_EXT_MODULE=workflows.scripts.rlinf_ext

独立运行的检查点转换等工具（例如
``python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf``）在调用
``get_model`` 之前也会加载 ``RLINF_EXT_MODULE``，与 Worker 侧行为一致；因此若
``model_type`` 仅通过扩展模块注册，转换前同样需要设置该环境变量。

2. 模型实现
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 `rlinf/models/embodiment/your_model_action_model.py` 中创建你的类，并继承自 HuggingFace 基类。  
对于接入 RLinf 的自定义 VLA 模型，需要遵循 `rlinf.models.embodiment.base_policy.BasePolicy`
的接口约定，至少提供 `forward` 和 `predict_action_batch` 接口；如果直接继承
`BasePolicy`，通常需要实现 `default_forward` 和 `predict_action_batch`，其中 `forward`
会由基类统一分发。`predict_action_batch` 用于封装生成、解码和可选的数值计算，将
RL 逻辑保持在模型内部。  

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

参考 `rlinf/models/embodiment/openpi/openpi_action_model.py`，这三个接口的输入输出建议按下面理解：

- ``forward(forward_type=..., **kwargs)``:
  统一分发入口，根据 ``forward_type`` 路由到 ``default_forward``、``sft_forward``、
  ``sac_forward`` 或其他训练分支。对于自定义 VLA 模型，推荐保留这一层分发，
  让 rollout、actor、sft 等调用方都走统一入口。
- ``default_forward(forward_inputs, **kwargs)``:
  用于训练阶段根据 rollout 缓存的中间结果重新计算 logprob / value / entropy。
  参考 OpenPI，``forward_inputs`` 通常至少包含：

  - ``chains``: 扩散或迭代采样过程中的动作链，供训练时重算 logprob 使用。
  - ``denoise_inds``: 每个样本对应的去噪 step 索引，决定取哪一步的训练信号。
  - ``tokenized_prompt`` / ``tokenized_prompt_mask``: 文本 prompt 及其 mask。
  - ``action``: 实际交给环境执行的动作，通常展平成 ``[B, ...]``。
  - ``model_action``: 模型原始输出动作，通常在输出变换前保留。
  - 观测字段: 如 ``observation/image``、``observation/state``、
    ``observation/wrist_image`` 等，供模型重新构造 observation。

  OpenPI 的 ``default_forward`` 返回一个 ``dict``，核心字段包括：

  - ``logprobs``: 当前动作在训练分支下重算得到的对数概率。
  - ``values``: value head 输出的状态值。
  - ``entropy``: 策略熵或等价的不确定性统计，用于 PPO / RL loss。
- ``predict_action_batch(env_obs, mode="train"|"eval", compute_values=True, **kwargs)``:
  用于 rollout / 推理阶段，输入环境观测并输出可直接执行的动作。参考 OpenPI，
  ``env_obs`` 通常包含：

  - ``main_images``: 主视角图像。
  - ``wrist_images``: wrist camera 图像，可选。
  - ``extra_view_images``: 额外视角图像，可选。
  - ``states``: 低维状态。
  - ``task_descriptions``: 文本任务描述。

  OpenPI 的返回值是 ``(actions, result)``：

  - ``actions``: 直接发送给环境执行的动作张量。
  - ``result["prev_logprobs"]``: rollout 时记录下来的旧策略 logprob。
  - ``result["prev_values"]``: rollout 时记录下来的旧 value。
  - ``result["forward_inputs"]``: 训练阶段会再次送入 ``default_forward`` 的缓存字段集合。

实现自定义模型时，不要求字段名与 OpenPI 完全一致，但建议保证两点：一是
``predict_action_batch`` 返回的 ``forward_inputs`` 足以让 ``default_forward``
重建训练所需的全部信息；二是 ``default_forward`` 输出至少覆盖训练算法实际依赖的
``logprobs``、``values``、``entropy`` 等字段。

3. 配置文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 `examples/embodiment/config/your_config.yaml` 中创建配置文件，  
包含 `model_type`、`action_token_len` 和 `precision` 等字段。  
该模板会暴露你模型的超参数，方便实验设置。  

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

如果你的自定义模型在 FSDP 下需要显式指定 wrapping 规则，也可以在配置中补充
``fsdp_config.wrap_policy``。当该字段中出现自定义规则时，RLinf 会优先使用这里的
配置；否则仍按模型自身的 ``_no_split_modules`` / ``_no_split_names`` 以及内置逻辑
处理。

常用字段包括：

- ``transformer_layer_cls_to_wrap``: 按类名指定需要按 transformer block 方式 wrapping
  的模块，适合 decoder layer、attention block 等主体结构。
- ``module_classes_to_wrap``: 按类名指定额外需要 wrapping 的模块，适合 vision encoder、
  projector、adapter 等非标准 transformer block。
- ``no_split_names``: 按模块上的 ``_fsdp_wrap_name`` 做更细粒度的 wrapping，适合类名
  不方便区分、但模块角色固定的场景。

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

如果你的自定义模块本质上就是主干 transformer block，通常只配置
``transformer_layer_cls_to_wrap`` 就够了；``module_classes_to_wrap`` 更适合补充那些
不属于主干 block、但仍希望单独参与 FSDP wrapping 的模块。

4. 模型使用
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

完成 ``register_model("your_model_type", build_my_model, category="embodied")`` 后，
即可在 YAML 配置中直接使用 ``model_type: your_model_type``。如果你的模型还依赖
对应的 ``processor`` 类，建议在 ``build_my_model`` 中统一完成加载后再返回模型实例。

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- RLinf 会在环境侧对不同模型输出的 action 做额外处理，例如
  `rlinf/envs/action_utils.py` 中的 `prepare_actions(...)` 会根据 `env_type`
  和 `model_type` 执行不同的后处理逻辑。如果你使用的是全新的自定义 `model_type`，
  建议直接在 `predict_action_batch(...)` 中完成这部分额外的 action 后处理，
  让返回给环境的 `actions` 已经是最终可执行格式，这样可以避免再为新模型类型补充
  额外的环境侧分支逻辑。
- 使用 `register_model(...)` 注册自定义模型时，可以通过 `force=True` 覆盖已有模型实现。
  这种方式适合“替换 RLinf 内置模型实现但沿用原有 model type”的场景。这样你可以继续复用
  RLinf 针对该内置 `model_type` 已有的 action 处理、训练分支和其他兼容逻辑，从而减少
  额外适配代码；前提是你的模型接口需要与 RLinf 内置模型保持完全对齐。
- 对于不同 VLA 模型，RLinf 在 FSDP 训练下使用了不同的 wrap policy，相关逻辑可参考
  `rlinf/hybrid_engines/fsdp/utils.py` 中的 `get_fsdp_wrap_policy(...)`。如果你接入的是
  自定义模型，建议在模型配置中显式补充对应的 wrap policy 配置，确保 transformer layer、
  vision encoder、projector、value head 等关键模块按预期参与 FSDP wrapping；否则可能
  出现训练性能下降、显存不合理或某些模块未被正确切分的问题。

