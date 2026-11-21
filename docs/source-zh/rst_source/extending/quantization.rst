为模型接入量化训练
==================

可以扩展量化格式，也可以将新模型接到共享 QAT 运行时。量化后端实现数值操作，模型适配器描述投影目标和采样约束；OpenPI π0.5 的实现可作为模型参考。

先保持原模型接口
----------------

先按 :doc:`FSDP 模型扩展 <new_model_fsdp>` 注册 policy，使原浮点训练能运行。量化适配器复用这个 model_type，不改变 worker、runner 或原 checkpoint 格式。普通 Linear 可以复用 FakeQuantLinear 和 QuantizedLinear；函数式投影由模型适配器显式处理。actor 和 rollout 投影接收 QuantizationBackend 对象。工厂只在初始化时选择配置指定的格式，模型前向直接调用后端方法，不再判断格式。示例配置仍默认选择 NVFP4。

实现数值后端
------------

继承 ``rlinf/quantization/backends/base.py`` 中的 QuantizationBackend，实现 quantize、dequantize、gemm 和 probe；probe 负责认证实际硬件与数值路径。在 create_quantization_backend 中加入格式，并补充配置支持的配方，模型适配器无需新增格式白名单或分支。

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 操作
     - 契约
   * - ``quantize(values, role="weight", native=False)``
     - 返回 QuantizedTensor，包含 data、scales、logical_shape、format、storage，以及可选的 global_scale/padded_shape。role 区分激活与权重。
   * - ``dequantize(value)``
     - 返回去除 padding 的逻辑 FP32 张量。logical/native 存储标记明确区分逐元素编码与原生打包布局。
   * - ``gemm(activation, weight, bias=None, output_dtype=...)``
     - 使用相容的原生低比特矩阵，在支持的硬件上计算，保持配方要求的输出舍入与 bias 顺序。
   * - ``autotune()``
     - 返回 eager 调优上下文，默认不做额外操作。NVFP4 用 FlashInfer 选择 CUTLASS 策略。图捕获前需退出此上下文。
   * - ``fake_quantize(values, role="weight")``
     - 完成 QDQ 和 STE；格式需要特殊 STE 前向或融合路径时覆盖此方法。
   * - ``fake_linear(input, weight, bias=None, weight_dtype=None)``
     - 执行 actor 伪量化投影。weight_dtype 保留模型可选的浮点参数类型对齐，固定精度配方自行决定计算边界。

公共 quantize 和 dequantize 方法只分派 logical/native 布局，各布局由独立私有方法处理并复用数值公式。两种布局统一使用 QuantizedTensor；NVFP4 native 反量化先独立解包 nibble、还原 scale 布局，再调用公共反量化公式。真实 GEMM 直接消费同一个容器。

kernels 目录仅保留 actor 或 rollout 实际使用的执行路径。logical 编码在 CPU/CUDA 上复用 PyTorch 参考实现，不保留单独输出 codes 的 kernel。真实 GEMM 仅用于推理，actor 梯度通过伪量化路径计算。

NVFP4 覆盖两个 fake 方法，保留融合 BF16 QDQ、Passthrough STE，以及 BF16 GEMM 后加 FP32 bias、再恢复输入 dtype 的顺序。gate/up 权重继续分别计算全局 scale。把数值方法与模型参数的持有关系分开后，新增格式可以复用以下四个模块：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 模块
     - 职责
   * - QuantizationBackend
     - 实现量化、反量化、真实 GEMM 和伪量化前向；持有数值配方。
   * - QuantizedWeight
     - 持有派生的低比特 data、scales、global_scale 和形状；不持有浮点 Parameter，也不执行前向计算。
   * - QuantizedLinear
     - 保留原浮点 Parameter 和 checkpoint 键，持有 QuantizedWeight，并组织激活动态量化、真实 GEMM 和输出整形。
   * - FakeQuantLinear
     - 保留原浮点 Parameter，每次前向调用 backend.fake_linear，以 STE 将梯度传给浮点权重。

QuantizedTensor 是一次数值操作返回的值；QuantizedWeight 是长期持有这些派生 buffers 的 Module。调用 QuantizedWeight.load_weights(master_weight) 从当前浮点权重刷新 buffers；value 每次从当前 buffers 构造 QuantizedTensor 视图。clear() 使值不可用，同时保留分配供后续刷新复用。设备/类型变换与 checkpoint 加载也会使值不可用，下一次同步重新生成量化权重。

普通 Linear 使用 QuantizedLinear.load_weights() 刷新派生权重，再通过 forward 计算。函数式 FFN、QKV 或 patch 适配器直接持有 QuantizedWeight，并复用 ``rlinf/quantization/layers/linear.py`` 中的 quantized_linear 组织计算。

Pi0 选择 eager 采样或图回放，用 capture_cuda_graph() 和 release_cuda_graph() 管理图的生命周期。仅在 needs_capture() 判定需要新图时，Pi0 才清理旧图、准备 policy 并调用 capture()。对于量化 policy，这一步准备是在 runtime.backend.autotune() 内，用实际观测和显式噪声执行 _sample_actions_eager()。每次符合条件的采样调用 replay()。EvalCudaGraph 管理静态输入、捕获和回放，量化准备由 Pi0 负责。

迁移设备、转换 dtype 或用 load_state_dict(assign=True) 替换参数前，先调用 release_cuda_graph()，变更后调用 capture_cuda_graph()。rollout worker 的卸载和重载流程遵循这个顺序。

FakeQuantLinear 和 QuantizedLinear 共用 BaseQuantizedLinear 的初始化与 from_linear 转换逻辑，复用原 Parameter、checkpoint 键、训练状态和 FSDP 包装标记。两种前向计算分别由对应子类实现。

``kernels/int8.py`` 和 ``kernels/nvfp4.py`` 提供 GPU 调用入口；``kernels/int8_triton.py`` 和 ``kernels/nvfp4_triton.py`` 实现对应的 Triton kernel。入口按需导入 Triton 实现，配置验证阶段保持轻量。

目录按职责组织：

.. code-block:: text

   quantization/
     backends/       # 抽象接口、格式实现、兼容性检查
     kernels/        # CUDA/FlashInfer 包装与 Triton kernel
     layers/         # 通用真实/伪量化 Linear 与派生权重 buffers
     algorithms/     # 采样事件、冻结锚点与 PPO 修正
     models/openpi/  # 模型目标、参数视图与投影替换

一次注册两类职责
----------------

配置验证在 CPU 驱动进程执行，验证回调与注册入口不加载模型或推理后端；RLinf 基础包本身依赖 PyTorch。真正的模块替换在 worker 内通过延迟工厂创建。将下面代码放入扩展模块，例如 my_policy_qat。驱动入口须在调用 validate_cfg 前显式调用该模块的 register()；设置 ``RLINF_EXT_MODULE=my_policy_qat`` 让各 worker 分别加载同一注册。现有标准入口不会在配置验证前自动加载这个环境变量。

.. code-block:: python

   from functools import cache

   from rlinf.quantization.registry import register_quantization_model
   from my_policy_qat.config import validate_model_config

   def build_adapter():
       from rlinf.quantization.registry import QuantizationAdapter
       from my_policy_qat.adapter import (
           discover_targets, apply_actor, apply_rollout, load_rollout_weights,
       )
       return QuantizationAdapter(
           discover=discover_targets,
           apply_actor=apply_actor,
           apply_rollout=apply_rollout,
           load_rollout_weights=load_rollout_weights,
       )

   @cache
   def register():
       register_quantization_model("my_policy", validate_model_config, build_adapter)

register() 使用标准库 cache 保证同一进程只注册一次，避免多次初始化 worker 时重复注册。在自定义训练入口中，先调用注册函数，再调用原入口。

.. code-block:: python

   from my_policy_qat import register
   from examples.embodiment.train_embodied_agent import main

   if __name__ == "__main__":
       register()
       main()

两类回调作为同一个条目注册，重复名称或缺少回调都会报错。配置验证器接收 model mapping、CorrectionSpec 和 actor/rollout scope，检查模型支持范围，并返回可比较的采样与预处理配置。actor 和 rollout 的有效配置必须相同；这一步不实例化执行适配器。

实现模型操作
------------

执行工厂返回 QuantizationAdapter 的四个回调。共享运行时只发现一次目标，解释 qat.targets 中的通配符后，将已选中的 TargetManifestEntry 序列传给两个安装回调。适配器直接使用这份清单，不再遍历模型或匹配 selectors。

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 回调
     - 契约
   * - ``discover(model)``
     - 返回按路径排序的 TargetManifestEntry 序列。每项包含目标路径、原 Parameter 的 state_dict 键、实际形状和模型内部 kind。
   * - ``apply_actor(model, targets, *, backend)``
     - 按已解析清单安装 actor 伪量化投影，返回同一份目标清单；原 Parameter 对象、requires_grad 和 checkpoint 键不变。
   * - ``apply_rollout(model, targets, *, backend)``
     - 安装 rollout 真实低比特投影及其 QuantizedWeight，派生 buffers 不进入 checkpoint；保持同样的参数契约。
   * - ``load_rollout_weights(model)``
     - 逐层调用 load_weights()，从当前浮点 master 参数生成量化权重；模型版本由共享 runtime 统一发布。

运行时根据已验证的 block_size 和 quantization_format 构造后端，将 backend 传给两个安装回调。validate_qat_config 检查完整配方和 actor/rollout 配置一致性，后端创建时检查 block_size 不变量。目标发现和投影工厂检查模型结构，安装阶段同步使用已解析清单。适配器完成替换后刷新模型的 FSDP 包装标记。

QuantizedPolicyRuntime 持有模型版本和 profile hash。每次 load_weights(version) 先撤销已发布版本，使所有 QuantizedWeight 的派生值不可用，再调用模型权重加载器。所有权重加载完成且后端配方一致后，运行时才发布新版本。每个 QuantizedWeight 在加载完成后公开派生值。

公开数值入口检查实际张量的设备、形状与布局，私有 kernel 启动函数复用这些已验证的数据。NVFP4 GEMM 负责输入与输出参数检查，私有 epilogue 复用这项契约。后端初始化检查 GPU 架构和依赖版本；量化、打包和 GEMM 的数值对照放在测试中。

OpenPI 的 actor 伪量化位于 ``models/openpi/actor.py``，rollout 真实低比特投影位于 ``models/openpi/rollout.py``。适配器使用 apply_pi05_actor_quantization、apply_pi05_rollout_quantization 和 load_pi05_rollout_weights 三个入口。所有格式共用模型投影和 ``rlinf/quantization/layers/weights.py``；格式数值位于 ``rlinf/quantization/backends/``，原生 GPU 调用位于 ``rlinf/quantization/kernels/``。

重放已采样的选择
----------------

保存采样输入和原始 rollout log-prob，让 actor 在第一次 PPO 更新前对同一批选择重新评分。RLinf 每批只冻结一次 actor 分数，保存在 proximal_logprobs 中；prev_logprobs 保留原始行为分数。OpenPI 使用已有动作链和选定的转移索引重放。

为 actor 和 rollout 使用相同的采样与预处理设置，共享配置校验会通过模型配置验证器比较这些设置。接入其它分布时，为已保存的选择实现对应的 log-prob 计算。

初始化工厂选择数值后端，不支持的配方在配置入口直接报错。

验证接入
--------

先验证关闭 QAT 后保留原模型，再验证参数 identity、state_dict、冻结范围、梯度、派生权重可用性和模型发布版本。随后对固定输入与噪声检查真实数值/后端调用，最后运行该模型的训练与任务评估；单纯注册成功不代表完整 RL 已支持。

现有测试覆盖一个独立的小模型只注册一次即可到达配置验证和 QAT 执行入口，并在新进程初始化 RLinf 后禁止额外 Torch/模型导入，检查配置验证路径。新增适配器仍需自己的真实模型、采样重放及训练验证。运行方式见 :doc:`量化训练指南 <../guides/quantization>`。
