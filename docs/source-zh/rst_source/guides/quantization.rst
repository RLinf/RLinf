具身 Policy 量化训练
====================

使用 CALVIN 示例默认的 NVFP4 W4A4 配方，完成 OpenPI π0.5 的 QAT、训练内验证和断点续训；也可显式切换 INT8 或关闭量化。actor 保留浮点主参数，rollout 使用真实低比特矩阵乘法；当前工作流使用同步 FSDP1 与 HF rollout。

actor 每次前向都根据当前权重重新做伪量化，包括冻结的 VLM 权重。rollout 将同步后的权重打包为低比特缓存，在下一次权重更新前复用；激活则在每次前向动态量化。

准备环境与资产
--------------

以下命令在包含 QAT 配置的 RLinf 仓库根目录执行。先按 :doc:`CALVIN 资产与模型下载 <../examples/embodied/calvin>` 准备模拟器和 SFT checkpoint。可使用公开镜像，或从源码建立环境；两条路径都需要安装当前检出的 RLinf 本体，不能依赖旧环境里的 editable 安装。

使用预构建环境
~~~~~~~~~~~~~~

已有 NVIDIA 驱动和 Docker 时，挂载当前代码与模型目录，使用公开 CALVIN 镜像。下面先只暴露一张 GPU，运行短验证。参考环境是 SM120、72 GB GPU，实际显存需求随批量和拓扑变化。短验证的固定预算支持 1、2、4 或 8 张可见 GPU；八卡机器可改用 --gpus all，其它拓扑需调整环境数、批量和 placement，以满足整除约束。

.. code-block:: bash

   docker run -it --name rlinf-qat --gpus device=0 --cpus=64 --shm-size=32g \
     --ulimit memlock=-1 --ulimit stack=67108864 \
     -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
     -v "$PWD:/workspace/RLinf" -v "/path/to/model:/workspace/model" \
     -w /workspace/RLinf \
     infinigence-ai-registry.cn-beijing.cr.aliyuncs.com/rlinf/rlinf:agentic-rlinf0.4-calvin

   source switch_env openpi
   # 国内网络可为后续 uv 命令选择公开索引。
   export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple
   export HF_ENDPOINT=https://hf-mirror.com
   uv pip install --no-deps -e .
   export QAT_MODEL_PATH=/workspace/model

本次验证的镜像索引 digest 是 ``sha256:4640269ba481cad1e6f11a9db566d07eece209465685027f5ecc3d2588d1a1d7``；需要固定环境时，可在镜像名后增加 @ 和这个 digest。它提供 Torch 2.11.0+cu128、Triton 3.6.0 和 OpenPI/CALVIN；后续只需添加所选量化后端的依赖。

从源码建立环境
~~~~~~~~~~~~~~

没有预构建环境时，在带 CUDA 12.8 工具链的 NVIDIA Linux 环境使用已有安装器。当前 π0.5 实现使用 Torch 算子，短验证不需要编译 flash-attn；安装器的 --install-rlinf 选项同时安装检出的代码。

.. code-block:: bash

   # 国内网络可再加 --use-mirror。
   UV_TORCH_BACKEND=cu128 bash requirements/install.sh --torch 2.11.0 \
     --no-flash-attn --install-rlinf embodied --model openpi --env calvin
   source .venv/bin/activate
   export QAT_MODEL_PATH=/path/to/RLinf-Pi05-CALVIN-ABC-D-SFT

确认输入与量化依赖
~~~~~~~~~~~~~~~~~~

两条环境路径准备好后，默认 NVFP4 配方需要额外安装 FlashInfer；显式选择 INT8 时使用已有 Torch/Triton。先确认权重与 normalization 文件存在，再检查解释器和实际导入的源码路径。

.. code-block:: bash

   # 国内网络可先设置以下公开索引。
   export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple
   export HF_ENDPOINT=https://hf-mirror.com
   uv pip install 'flashinfer-python==0.7.1'  # 仅 NVFP4 需要。
   test -s "$QAT_MODEL_PATH/model.safetensors"
   test -s "$QAT_MODEL_PATH/InternRobotics/InternData-Calvin_ABC/norm_stats.json"
   python -c 'import rlinf, torch; print(rlinf.__file__, torch.__version__, torch.version.cuda)'
   export EMBODIED_PATH="$PWD/examples/embodiment"

尚无模型时，可在激活的环境中使用 ``hf download RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT --revision ffe18938a37d3855c9ff50d6220d39bcd6434f15 --local-dir "$QAT_MODEL_PATH"`` 获取本次参考版本，再执行文件检查。环境安装、模型下载和首次 JIT 编译应分别计时，不能把预下载资产算作从零下载成功。

INT8 需要支持 BF16 的 SM80 或更新设备；NVFP4 GEMM 需要 SM100/103/120/121，首次调用会编译原生后端。后端初始化检查 GPU 架构和依赖版本，数值正确性由单元测试覆盖。记录 GPU、驱动、依赖和代码版本后，再启动训练。

选择配置组
----------

量化规则和概率修正放在 ``examples/embodiment/config/qat/``，只修改顶层 ``qat`` 节点。主示例默认选择 nvfp4，短验证和恢复也使用同一配置；使用 ``qat=int8`` 或 ``qat=off`` 切换。PPO、FSDP、worker 和训练预算由主实验配置独立管理，切换量化组会保留这些设置。

.. list-table::
   :header-rows: 1
   :widths: 24 28 24 24

   * - qat 选项
     - 权重
     - 激活
     - actor 前向
   * - off
     - 原模型
     - 原模型
     - 原模型
   * - int8
     - 输入维 block128
     - 输入维 block128
     - fake quant
   * - nvfp4
     - 输入维 block16
     - 输入维 block16
     - fake quant

使用 ``qat=off`` 关闭量化，得到相同实验设置下的浮点对照。需要官方 BF16 的完整设置时，选择原有 calvin_d_d_ppo_openpi_pi05 配置。当前 QAT 实验明确使用同步 FSDP1/HF，关闭 compile、CUDA Graph、AMP 和 offload；这些是本实现的运行条件，不能当成 QAT 的普遍要求。

两种配方的权重和激活均沿输入维度分块，使用动态 scale 和 nearest_even 舍入，中点取相邻偶数编码。INT8 的每块 scale 在 FP32 中按 max(amax × (1/127), 1e-12) 计算。actor 始终使用伪量化前向，rollout 根据格式选择真实低比特 GEMM；内部数值配方标识和版本由配置推导并记入 hash，用于同步核验。

选择量化目标
------------

``qat.targets`` 是投影路径的包含列表，多项取并集。下面是 OpenPI π0.5 的可直接填写示例；其它模型使用各自适配器提供的目标路径。

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - targets 值
     - 量化范围
   * - ["*"]
     - 全部受支持投影：视觉、两个 Gemma expert，以及 action/time 入口出口投影。
   * - ["img.*"]
     - 视觉投影，包括 patch 卷积、视觉注意力和 FFN 投影。
   * - ["llm.*"]
     - Gemma 的语言 expert 和 action expert 投影。
   * - ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]
     - 仅独立的 action/time 入口出口投影；不包含 Gemma action expert。

匹配使用区分大小写的 fnmatch 模式，``*`` 可以跨越路径中的点号；``**`` 没有额外的递归含义。精确路径也可直接填写，每个模式都必须命中至少一个候选目标。例如，仅量化视觉时，在下文命令中增加 ``qat.targets='["img.*"]'``。候选范围不包含 embedding、norm、softmax、注意力分数矩阵乘法或 value head，因此默认星号也不会量化这些运算。

先检查再运行
------------

先展开实际配置，核对两个模型路径、量化格式和训练预算。这条命令只输出配置，不启动模型。

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
     --config-path="$EMBODIED_PATH/config" \
     --config-name=calvin_d_d_ppo_openpi_pi05_qat \
     qat=nvfp4 \
     actor.model.model_path="$QAT_MODEL_PATH" \
     rollout.model.model_path="$QAT_MODEL_PATH" \
     --cfg job --resolve

确认后去掉 ``--cfg job --resolve`` 启动训练。示例沿用官方 D→D 设置：64 个环境、8 次 rollout、4 遍 PPO；默认冻结 VLM、训练 action/value，但量化覆盖视觉、语言和 action 投影。embedding、norm、softmax 和注意力分数矩阵乘法保留原实现。NVFP4 的 global scale 跨整个逻辑投影矩阵计算，batch 内其它样本可能影响量化，因此对照实验应保持批量与 placement 相同。

先验证一轮数据通路
------------------

下面复用主配置，通过命令行显式缩小训练预算：8 个环境、每次 10 步、1 次 rollout、1 遍 PPO，保存一次 checkpoint 并触发一次短验证。它只检查采样、反向、同步、保存与验证是否可运行，不能用其任务成功率评价 policy。显存不足时先调整 placement 和批量，并保存最终展开配置。

.. code-block:: bash

   export QAT_OUTPUT="$PWD/results/qat-smoke"
   python examples/embodiment/train_embodied_agent.py \
     --config-path="$EMBODIED_PATH/config" \
     --config-name=calvin_d_d_ppo_openpi_pi05_qat \
     actor.model.model_path="$QAT_MODEL_PATH" rollout.model.model_path="$QAT_MODEL_PATH" \
     runner.logger.log_path="$QAT_OUTPUT" runner.logger.experiment_name=pi05_qat_smoke \
     runner.max_epochs=1 runner.save_interval=1 runner.val_check_interval=1 \
     env.train.total_num_envs=8 env.train.rollout_epoch=1 \
     env.train.max_episode_steps=10 env.train.max_steps_per_rollout_epoch=10 \
     env.eval.total_num_envs=8 env.eval.rollout_epoch=1 \
     env.eval.max_episode_steps=10 env.eval.max_steps_per_rollout_epoch=10 \
     actor.micro_batch_size=1 actor.global_batch_size=8 algorithm.update_epoch=1

checkpoint 位于 ``$QAT_OUTPUT/pi05_qat_smoke/checkpoints/global_step_1/actor``；默认 TensorBoard 指标和展开配置 config.yaml 位于 ``$QAT_OUTPUT/tensorboard/``。有效反向后 action/value 参数应有有限梯度，冻结 VLM 不应改变；真实 rollout 的缓存会随同步权重重建。验证这些结果后再恢复正常训练预算。

恢复与比较
----------

保持上述短验证命令的配置、拓扑和批量，将 runner.max_epochs 从 1 改成 2，并增加下列覆盖，就会从第 1 轮继续。无需手动打包量化 checkpoint。

.. code-block:: text

   runner.resume_dir="$QAT_OUTPUT/pi05_qat_smoke/checkpoints/global_step_1"

标准 checkpoint 保存 actor 主参数、优化器与调度状态；量化缓存属于派生数据，恢复后重新生成。环境与 rollout 进程会重建，不能把恢复训练解释为逐位连续的模拟轨迹。

示例使用启用 init_sync 的 patch weight syncer，启动时先同步全部 actor 参数，随后发送增量更新。如果选择 bucket syncer，actor 和 rollout 的冻结参数必须在启动时一致，因为它只同步可训练参数和持久 buffers。恢复的 checkpoint 若与 rollout checkpoint 的冻结 VLM 不同，应使用 patch init_sync，或为 rollout 加载相同的冻结权重。

比较 BF16/PTQ/QAT 时，需要保持 SFT 起点、normalization、任务序列、初态、噪声、批量、步数和训练预算一致，并额外保留“普通 RL 后再 PTQ”的控制。训练内 validation 使用当前 policy；当前 QAT 预设不支持独立 only_eval。历史固定 1000 条长程评测与 200 轮结果依赖原评测 harness，不应当成这个新入口已经重现的成绩。先用短 smoke 验证工程链路，再运行事先固定的正式协议。

配置概率修正
------------

rollout 保留原行为 log-prob；actor 在每批 PPO 更新前评分一次并冻结旧锚点。``qat.correction.mode=importance_sampling`` 用 exp(旧 actor log-prob − rollout log-prob) 加权 PPO 贡献，``qat.correction.max_weight`` 默认为 2；设为 ``qat.correction.mode=off`` 时，每条样本的行为权重为 1，actor 和 rollout 仍使用所选量化配方。

主实验中的 algorithm.recompute_logprobs=false 关闭的是旧的“重算后覆盖原行为分数”路径。QAT 仍会重算 actor 分数，只是单独保存在 proximal_logprobs，供 PPO 锚点和概率修正使用。

OpenPI 计算的是选定高斯转移的密度，不能称为完整最终动作 policy 的密度。

INT8 使用 [-128,127]、中点取偶数舍入和 1e-12 scale 下限；NVFP4 使用 E2M1、E4M3FN 块 scale 与 FP32 全局 scale。fake quant 与真实 GEMM 的累加可能不同，概率修正不保证二者逐元素相等。

扩展模型与限制
--------------

其它模型通过 :doc:`量化模型扩展 <../extending/quantization>` 的单次注册接入。当前已适配 OpenPI π0.5；自定义模型还需实现可重放的采样评分事件。

当前门槛拒绝 FSDP 命名精度规则、独立 only_eval、异步/RTC、LoRA、compile、CUDA Graph 和可训练 sigma。QAT 也不支持 ``openpi.short_memory=true`` 和 ``openpi.pytorch_numerics=true``；两项默认均为 false，不影响默认 CALVIN 配置。RLinf 提供通用 CUDA Graph 管理器；当前主线 OpenPI policy 的采样器尚未实现该接口。实验分支已验证的 eval 采样器 Graph 接线需单独集成，也不能直接当成训练采样和反向已支持。预设描述 RLinf 数值规则，不代表 ApxInf 或 ModelOpt checkpoint 兼容。记录展开配置、Git SHA、依赖、模型/normalization 指纹与原始结果，才能区分环境、训练与量化导致的差异。
