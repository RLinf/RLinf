版本说明
==========

RLinf v0.4 版本发布
-------------------

🚀 RLinf v0.4 正式发布。让强化学习真正走出仿真，敏捷操控物理世界。

如果说 v0.3 打通了具身智能从仿真到真机的基础链路，那么 RLinf v0.4 则是一次面向真实物理世界规模化落地的全面跃迁。本版本围绕「乐高式真机系统 + 高效率真机算法 + 全模态模型与国产算力底座」三个核心维度，重塑真机强化学习的工程与算法范式：

- 🧩 **乐高式真机工程体系**\（``rlinf.robotics``）：将机械臂、灵巧手、相机、移动底盘与遥操作外设抽象为即插即用的组件，硬件驱动与任务逻辑彻底解耦。更换本体、调整构型或跨节点部署时，上层任务与策略代码无需改动。
- ⚡ **真机 SOTA 后训练算法矩阵**：推出表征解耦在线强化学习（RLT）、零标注自监督离线强化学习（STEAM）、30 步成功率跃升至 96%+ 的在线策略蒸馏（OPD），以及将推理等待压缩至约 2 ms 的异步实时控制（RTC），正面解决真机样本贵、标注难、推理卡顿三个痛点。
- 🌐 **全模态模型重构与扩展**：从底层重写与官方 JAX 逐算子数值对齐的 PyTorch 原生 OpenPI（π₀ / π₀.₅），接入 NVIDIA Cosmos3、FastWAM、π₀-FAST、Evo-1、MolmoAct2 等世界动作模型与 VLA，并将强化学习首次拓展至 SD3 / Wan2.2 视觉生成模型。
- 🤝 **八大异构算力与开源生态**：新增昆仑芯、壁仞、沐曦、Intel XPU 支持，在华为昇腾与摩尔线程上实现最高 2.78× 吞吐提升，并与 A100 收敛对齐。RLinf 正式入选 PyTorch 官方生态，成为 NVIDIA Isaac Lab v3.0 官方集成的强化学习训练引擎，并与 SGLang、AMD 发布多项社区实践。

本版本对已支持的训练与评测配方做了严格的正确性与可复现性验证（见文末测试结果）。

具身
^^^^

1. 真机
"""""""

统一模块化机器人接口把硬件与任务解耦，支持机械臂、末端、相机与底盘灵活组合，并新增 3 款真机平台与双臂 / 单臂采集、训练、部署工具链。

**统一机器人架构：**

- **模块化真机接口**\（``rlinf.robotics``）：硬件驱动、任务定义与环境 wrapper 解耦，可按统一层级路径组合机械臂、灵巧手 / 夹爪、多视角相机与移动底盘。更换底层驱动或调整跨节点部署时，无需修改任务与策略代码。链接：:doc:`机器人接口 <../concepts/robotics>` ｜ :doc:`机器人架构 <../concepts/robotics_architecture>` ｜ :doc:`接入新机器人 <../extending/new_robot>`

**真机平台与训练链路：**

- **双臂 Franka 全流程**：新增 PICO VR 双臂遥操作采集、按臂独立接管的双臂 HG-DAgger 在线训练，以及基于原生 PyTorch OpenPI π₀.₅ 的双臂 SFT 与真机部署。链接：:doc:`双臂 Franka PICO DAgger <../examples/embodied/dual_franka_pico_dagger>` ｜ :doc:`双臂 Franka OpenPI <../examples/embodied/dual_franka_openpi_pytorch>`
- **单臂 Franka**：新增 Qwen-VL 真机视觉语言奖励模型、深度相机采集、平滑人工接管（Smooth Intervention）、DAgger 在线写入 LeRobot 数据集，以及 LeRobot 数据集分片合并与过滤。链接：:doc:`真机奖励模型 <../examples/embodied/franka_reward_model>`
- **新平台**：新增双臂 YAM（PICO VR 遥操作与 LeRobot 数据采集）、AgileX Piper（松灵）与 SO-101。链接：:doc:`AgileX Piper <../examples/embodied/piper>` ｜ :doc:`SO-101 <../examples/embodied/so101>`

2. 算法
"""""""

在真机在线强化学习、无标注离线策略优化、在线蒸馏与高频实时控制四个方向新增多项 SOTA 算法，降低真机样本开销，并提升任务成功率与部署流畅度。

**真机与仿真强化学习：**

- **RLT（RL Token）**：两阶段在线强化学习。高维视觉表征与在线控制解耦：第一阶段联合微调 VLA 与 RLT token 特征提取器，第二阶段在冻结的紧凑表征上训练轻量 Off-Policy Actor-Critic（含 TD3 变体）。在 ManiSkill 与 Franka 真机精密插孔任务上高效收敛。链接：:doc:`RLT <../examples/embodied/rlt>`
- **STEAM**：自监督离线优势建模。无需人工标注、手工奖励或在线环境交互，仅利用示教视频帧对的时序偏序训练深度集成进度评论器（worst-of-N），自动提取帧级优势并驱动无分类器引导（CFG）策略优化。在双臂叠毛巾、商品结账、可乐补货等真机任务上显著超越 RECAP。链接：:doc:`STEAM <../examples/embodied/steam>`

**在线策略蒸馏与实时控制：**

- **OPD（On-Policy Distillation）**：Student 自采样，Teacher 对 action token 做密集打分并即时卸载。LIBERO-Spatial 上 30 步训练即可将 OpenVLA-OFT Student 成功率从 67% 提升至 96%+。链接：:doc:`OPD <../examples/embodied/opd>`
- **RTC（Real-Time Control）**：下一动作 chunk 的推理与当前 chunk 的执行异步重叠，并用软 / 硬掩码平滑动作边界。Rollout 推理等待从数秒压缩至约 2 ms，单回合耗时最高缩短 7.4 秒，支持 LIBERO 仿真与 Franka 真机高频控制。链接：:doc:`RTC <../guides/rtc>`

3. 模型
"""""""

继续扩展具身模型生态，新增 5 款世界动作模型与 VLA，并完成 OpenPI 原生 PyTorch 重构，覆盖 SFT、在线 RL 与高吞吐闭环评测。

- **OpenPI（π₀ / π₀.₅）PyTorch 原生重实现**：与官方 JAX 参考数值严格对齐，解决原社区 PyTorch 版本的数值偏差与收敛问题。打通 LIBERO、ManiSkill、BEHAVIOR-1K、RoboTwin 及单 / 双臂 Franka 真机的全量 SFT、LoRA 与 RL。链接：:doc:`OpenPI SFT <../examples/embodied/sft_openpi>` ｜ :doc:`π₀ / π₀.₅ RL <../examples/embodied/pi0>`
- **NVIDIA Cosmos3**：16B MoT 全模态世界动作模型在 LIBERO 上的 FSDP2 监督微调，收敛精度与官方对齐；结合 SGLang 多 batch 与流水线并行评测，吞吐相对直接集成提升 3.33×。链接：:doc:`Cosmos3 SFT <../examples/embodied/sft_cosmos3>` ｜ :doc:`Cosmos3 SGLang 评测 <../evaluations/guides/cosmos3_sglang>`
- **FastWAM**：基于视频扩散的世界动作模型，支持 LIBERO 四大套件上的 FSDP 监督微调与批量闭环评测。链接：:doc:`FastWAM <../examples/embodied/sft_fastwam>`
- **π₀-FAST**：接入 LeRobot 自回归动作 token 策略，支持 LIBERO-Long 确定性评测与 token-level GRPO。链接：:doc:`π₀-FAST <../examples/embodied/pi0_fast>`
- **Evo-1**：约 1B 的 flow-matching VLA（InternVL3-1B + DiT 动作头），支持 LIBERO 全参数 SFT 与 GRPO。链接：:doc:`Evo-1 <../examples/embodied/evo1>`
- **MolmoAct2**：AllenAI 开源 VLA，支持 LIBERO Spatial / Object / Goal / Long 并行闭环评测。链接：:doc:`MolmoAct2 <../examples/embodied/molmoact2>`

4. 仿真器与评测
"""""""""""""""

拓展大规模家居操作基准与高保真真机链路仿真，并构建面向世界动作模型与 VLA 的高吞吐评测体系。

- **RoboCasa365**：大规模家居操作环境，提供 OpenPI PPO 在线强化学习示例。链接：:doc:`RoboCasa365 <../examples/embodied/robocasa365>`
- **真机链路仿真器**：在仿真训练中注入真实传感器观测延迟，以及跨集群 / 边云网络带宽与延迟，使真机算法在进入物理实机前完成系统回归。链接：:doc:`真机仿真器 <../examples/embodied/real_simulator>`
- **SGLang 与 ApxInf 评测后端**：完善 Cosmos3、DreamZero 等世界动作模型的 SGLang 评测路径；新增基于 Rust/CUDA 的 ApxInf 进程内推理后端，支持 π₀.₅ 在 LIBERO 上的低延迟评测。链接：:doc:`SGLang 具身模型评测 <../extending/sglang_embodied_model>` ｜ :doc:`DreamZero SGLang 评测 <../evaluations/guides/dreamzero_sglang>` ｜ :doc:`ApxInf <../guides/apxinf>`
- **现有仿真器**：LIBERO 新增相机标定、任意分辨率渲染与跳过中间帧渲染，并对齐 LIBERO-Pro 初始扰动；RoboTwin 新增新版 DAgger 与内存回收优化；EmbodiChain 升级至 0.3.0。

5. 系统与异构硬件
"""""""""""""""""

分布式图像与张量传输、权重重分片和训练可观测性全面升级。新增 4 款国产 / 异构加速卡，并在华为昇腾与摩尔线程上取得显著性能突破。

**性能与通信：**

- **无损观测图像压缩**：对 Env → Rollout 的图像观测做 XOR-delta + LZ4/Zstd 无损压缩，逐比特还原的同时将图像网络负载降至原始大小的 36%–41%。链接：:doc:`观测压缩 <../guides/obs_compression>`
- **通信与权重同步**：新增集合通信张量压缩、灵活轨迹传输和重构后的权重重分片，并支持异步 Runner 下 Rollout 与权重同步的安全重叠。
- **训练栈与可观测性**：默认运行时升级至 PyTorch 2.11（配套新版 SGLang / vLLM），支持 FSDP 模块级混合精度、SGLang VLM 趋势奖励（QwenTrend）以及全链路分布式 Tracing / Profiling。链接：:doc:`引擎版本 <../guides/version>`

**国产卡与跨硬件：**

- **新增四款加速卡**：昆仑芯（Kunlunxin XPU）、壁仞（Biren SUPA）、沐曦（MetaX MACA）与 Intel XPU。同一套训练栈可运行于 NVIDIA、AMD 及 6 类主流国产 / 异构加速卡。
- **华为昇腾（Ascend CANN）**：新增 GR00T（N1.5 / N1.6 / N1.7）、StarVLA、DreamZero、Wan 世界模型训练。通过 Attention 元数据缓存、NPU 融合算子、设备端批量预处理与 FSDP 前后向预取重叠，GR00T N1.5/N1.6 动作推理吞吐提升 195%–278%，四卡训练吞吐提升 37%–42%，StarVLA 训练吞吐提升 186%–240%。10 款主流具身模型在 Ascend 910B3 上达到 A100 75% 以上的端到端计算效率。
- **摩尔线程（MUSA）**：完善安装脚本、Docker 镜像与端到端测试。在 S5000 上完成 WoVR 世界模型强化学习闭环复现，191 步成功率曲线与 A100 的相关系数为 0.976（LIBERO 评测 74.6% vs A100 73.4%）。双机混合分片使整步耗时降低 15.3%，权重同步耗时降低 66%。

生成模型
^^^^^^^^

强化学习首次从具身策略扩展到独立的图像与视频扩散生成流水线。

- **Diffusion-NFT**：将扩散生成过程抽象为强化学习环境，支持 SD3 与 Wan2.2（含 TI2V-5B）在线 RL，内置图像 OCR 与视频帧级 Video OCR 奖励，并可扩展自定义媒体评分模型。链接：:doc:`视频生成模型的强化学习 <../examples/video_models>`

Agentic AI
^^^^^^^^^^

扩展大规模 MoE 推理训练与多智能体 FSDP 后端。

- **Moonlight-16B-A3B GRPO**：基于 Megatron-Bridge（TP=8、EP=8）与 SGLang 专家并行，支持 MLA + MoE 架构的 Moonlight-16B-A3B，以及 DeepSeek-V3 / Kimi-K2 架构模型的数学推理强化学习。链接：:doc:`Moonlight-16B GRPO <../examples/agentic/math_reasoning/moonlight>`
- **MA-FSDP**：多智能体协同强化学习同时支持 FSDP 与 Megatron。SearchR1 新增 FSDP 训练配置。

生态合作与学术研究
^^^^^^^^^^^^^^^^^^

- **PyTorch Ecosystem**：RLinf 正式成为 PyTorch 生态成员。博客：`PyTorch Ecosystem Landscape Q3 Update <https://pytorch.org/blog/pytorch-ecosystem-landscape-q3-update/>`__
- **NVIDIA Isaac Lab v3.0.0**：Isaac Lab 采用 RLinf 作为 VLA 后训练与强化学习训练基础设施。文档：:doc:`Isaac Lab 中的 RLinf <../examples/embodied/isaaclab>`
- **RLinf × SGLang**：打通 Cosmos3、DreamZero 等世界动作模型的 SFT 与 SGLang 流水线并行评测，评测吞吐提升 3.33×。博客：:doc:`RLinf × SGLang：Cosmos3 <blog/cosmos3_sft_and_sglang_evaluate>`
- **RLinf × AMD**：在 AMD MI300（ROCm）上复现 WoVR 静态世界模型强化学习，并用冻结的 Qwen3-VL（TOPReward）替代任务专用奖励模型。LIBERO-Spatial 成功率从 44.8% 提升至 56.0%，与专用 ResNet 奖励处于同一水平。博客：:doc:`RLinf × WorldLoop <blog/worldloop_topreward>`
- **NSDI 2027**：MoE 高性能变换-通信融合库 *FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion* 被 NSDI 2027 接收。文档：:doc:`FUSCO <../examples/system/fusco>` ｜ 论文：`arXiv:2512.22036 <https://arxiv.org/abs/2512.22036>`__
- **STEAM**：发布自监督时序集成优势建模论文 *STEAM: Self-Supervised Temporal Ensemble Advantage Modeling for Real-World Robot Learning*。文档：:doc:`STEAM <publications/steam>` ｜ 论文：`arXiv:2606.29834 <https://arxiv.org/abs/2606.29834>`__

重要修复
^^^^^^^^

v0.4 修复了影响训练数值精度、分布式通信稳定性与断点续训的问题，建议升级到最新版。主要包括：

- **算法与数值**：修复 PPO Actor 指标未按有效 mask 宽度归一、熵正则项随 micro-batch 误缩放、ReinForce++ 基线在单样本组下崩溃、DAgger bootstrap mask 不对齐、Co-training 未应用 SFT loss，以及 Critic 解释方差未全局聚合。
- **模型与训练后端**：修复 IQL Critic 未走 FSDP 根节点、FSDP 梯度范数未跨全部分片归约、FSDP2 局部 shard 恢复类型错误，以及 DCP Checkpoint 未保存单 rank 随机数状态。
- **调度与权重同步**：修复异步训练中 Rollout 与权重同步重叠的竞态、BucketWeightSyncer 参数名未排序导致的集合通信挂起、发送端初始同步缺少屏障，以及 FSDP 集合通信未遵循 ``RLINF_TIMEOUT``。
- **续训与环境**：Auto-resume 自动跳过未写完的 Checkpoint；修复 ManiSkill 卸载时的显存回收、MetaWorld 与多卡调度下的 EGL 设备映射，以及 LIBERO-10 GRPO 按初始状态分组采样。

升级说明
^^^^^^^^

从 v0.3 升级到 v0.4 请注意以下变化：

- **运行环境**：默认升级至 PyTorch 2.11 与新版 SGLang / vLLM。请使用 ``requirements/install.sh`` 重建环境，或拉取 ``rlinf0.4`` 系列 Docker 镜像，不要复用 v0.3 虚拟环境。
- **真机接口**：``rlinf.envs.realworld`` 与 ``rlinf.scheduler.hardware.robots`` 已迁移至 ``rlinf.robotics`` 与 ``rlinf.envs.real``。自定义接入请参考 :doc:`接入新机器人 <../extending/new_robot>`。
- **OpenPI 权重**：``model_type: openpi`` 默认指向与 JAX 对齐的新版 PyTorch 实现。加载旧版 OpenPI-PyTorch Checkpoint 前，请先使用 :doc:`Checkpoint 转换工具 <../guides/convertor>`。

快速开始
^^^^^^^^

- **安装**：请使用本版本的 ``requirements/install.sh`` 重建环境，不要复用 v0.3 虚拟环境。文档：:doc:`安装指南 <../start/installation>`
- **百度智能云**：`https://cloud.baidu.com/doc/AIHC/s/fmrenj9u1 <https://cloud.baidu.com/doc/AIHC/s/fmrenj9u1>`__
- **无问芯穹**：`https://docs.neogpu.com/posts/rlinf-ppo-vla.html <https://docs.neogpu.com/posts/rlinf-ppo-vla.html>`__

RLinf v0.4 测试结果
^^^^^^^^^^^^^^^^^^^

下列为 v0.4 全部测试结果，每条配置附有结果曲线。

.. list-table::
   :header-rows: 1
   :widths: 34 24 42

   * - 配置文件
     - 说明
     - 结果曲线
   * - ``examples/sft/config/behavior_sft_openpi_pi05_rlinf.yaml``
     - OpenPI_RLinf π₀.₅ · BEHAVIOR SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/behavior_sft_openpi_pi05_rlinf.png
          :alt: OpenPI_RLinf π₀.₅ · BEHAVIOR SFT result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/calvin_abcd_d_ppo_openpi_pi05.yaml``
     - OpenPI π₀.₅ · CALVIN ABC-D PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/calvin_abcd_d_ppo_openpi_pi05.png
          :alt: OpenPI π₀.₅ · CALVIN ABC-D PPO result curve
          :width: 95%
          :align: center
   * - ``examples/offline_rl/config/cfg_rl_openpi.yaml``
     - CFG-RL · OpenPI
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/cfg_rl_openpi.png
          :alt: CFG-RL · OpenPI result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/frankasim_ppo_mlp.yaml``
     - MLP · FrankaSim PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/frankasim_ppo_mlp.png
          :alt: MLP · FrankaSim PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/frankasim_sac_cnn_async.yaml``
     - CNN · FrankaSim SAC Async
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/frankasim_sac_cnn_async.png
          :alt: CNN · FrankaSim SAC Async result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/gsenv_ppo_openpi_pi05.yaml``
     - OpenPI π₀.₅ · GSEnv PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/gsenv_ppo_openpi_pi05.png
          :alt: OpenPI π₀.₅ · GSEnv PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/isaaclab_franka_stack_cube_ppo_gr00t.yaml``
     - GR00T · IsaacLab Stack Cube PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/isaaclab_franka_stack_cube_ppo_gr00t.png
          :alt: GR00T · IsaacLab Stack Cube PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_10_grpo_openvlaoft.yaml``
     - OpenVLA-OFT · LIBERO-10 GRPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_10_grpo_openvlaoft.png
          :alt: OpenVLA-OFT · LIBERO-10 GRPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_10_opd_openvlaoft.yaml``
     - OPD · OpenVLA-OFT · LIBERO-10
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_10_opd_openvlaoft.png
          :alt: OPD · OpenVLA-OFT · LIBERO-10 result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_10_ppo_gr00t.yaml``
     - GR00T · LIBERO-10 PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_10_ppo_gr00t.png
          :alt: GR00T · LIBERO-10 PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_goal_ppo_openpi.yaml``
     - OpenPI π₀ · LIBERO-Goal PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_goal_ppo_openpi.png
          :alt: OpenPI π₀ · LIBERO-Goal PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_goal_ppo_openpi_pi05.yaml``
     - OpenPI π₀.₅ · LIBERO-Goal PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_goal_ppo_openpi_pi05.png
          :alt: OpenPI π₀.₅ · LIBERO-Goal PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_object_async_ppo_openpi_pi05.yaml``
     - OpenPI π₀.₅ · LIBERO-Object Async PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_object_async_ppo_openpi_pi05.png
          :alt: OpenPI π₀.₅ · LIBERO-Object Async PPO result curve
          :width: 95%
          :align: center
   * - ``examples/sft/config/libero_sft_cosmos3.yaml``
     - Cosmos3 · LIBERO SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_sft_cosmos3.png
          :alt: Cosmos3 · LIBERO SFT result curve
          :width: 95%
          :align: center
   * - ``examples/sft/config/libero_sft_dreamzero_5b.yaml``
     - DreamZero-5B · LIBERO SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_sft_dreamzero_5b.png
          :alt: DreamZero-5B · LIBERO SFT result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_spatial_async_ppo_openpi.yaml``
     - OpenPI π₀ · LIBERO-Spatial Async PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_spatial_async_ppo_openpi.png
          :alt: OpenPI π₀ · LIBERO-Spatial Async PPO result curve
          :width: 95%
          :align: center
   * - ``evaluations/libero/libero_spatial_cosmos3_eval_sglang.yaml``
     - Cosmos3 · SGLang 评测
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_spatial_cosmos3_eval_sglang.png
          :alt: Cosmos3 · SGLang 评测 result curve
          :width: 95%
          :align: center
   * - ``evaluations/libero/libero_spatial_eval_pi05_RTC.yaml``
     - RTC · π₀.₅ · LIBERO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_spatial_eval_pi05_RTC.png
          :alt: RTC · π₀.₅ · LIBERO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_spatial_grpo_evo1.yaml``
     - Evo-1 · LIBERO-Spatial GRPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_spatial_grpo_evo1.png
          :alt: Evo-1 · LIBERO-Spatial GRPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_spatial_grpo_openpi_pi05.yaml``
     - OpenPI π₀.₅ · LIBERO-Spatial GRPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_spatial_grpo_openpi_pi05.png
          :alt: OpenPI π₀.₅ · LIBERO-Spatial GRPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_spatial_grpo_starvla.yaml``
     - StarVLA · LIBERO-Spatial GRPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_spatial_grpo_starvla.png
          :alt: StarVLA · LIBERO-Spatial GRPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_spatial_opd_openvlaoft.yaml``
     - OPD · OpenVLA-OFT · LIBERO-Spatial
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_spatial_opd_openvlaoft.png
          :alt: OPD · OpenVLA-OFT · LIBERO-Spatial result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_spatial_ppo_gr00t_n1d6.yaml``
     - GR00T-N1.6 · LIBERO-Spatial PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_spatial_ppo_gr00t_n1d6.png
          :alt: GR00T-N1.6 · LIBERO-Spatial PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_spatial_ppo_openpi_rlinf.yaml``
     - OpenPI_RLinf · LIBERO-Spatial PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_spatial_ppo_openpi_rlinf.png
          :alt: OpenPI_RLinf · LIBERO-Spatial PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_async_ppo_openpi.yaml``
     - OpenPI π₀ · ManiSkill Async PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_async_ppo_openpi.png
          :alt: OpenPI π₀ · ManiSkill Async PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_async_ppo_openpi_pi05.yaml``
     - OpenPI π₀.₅ · ManiSkill Async PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_async_ppo_openpi_pi05.png
          :alt: OpenPI π₀.₅ · ManiSkill Async PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_async_ppo_openvla.yaml``
     - OpenVLA · ManiSkill Async PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_async_ppo_openvla.png
          :alt: OpenVLA · ManiSkill Async PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_async_ppo_openvlaoft.yaml``
     - OpenVLA-OFT · ManiSkill Async PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_async_ppo_openvlaoft.png
          :alt: OpenVLA-OFT · ManiSkill Async PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_grpo_openvla.yaml``
     - OpenVLA · ManiSkill GRPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_grpo_openvla.png
          :alt: OpenVLA · ManiSkill GRPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_grpo_openvlaoft.yaml``
     - OpenVLA-OFT · ManiSkill GRPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_grpo_openvlaoft.png
          :alt: OpenVLA-OFT · ManiSkill GRPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_ppo_mlp.yaml``
     - MLP · ManiSkill PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_ppo_mlp.png
          :alt: MLP · ManiSkill PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_ppo_mlp_vlm_trend_reward_sglang.yaml``
     - SGLang VLM Trend reward
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_ppo_mlp_vlm_trend_reward_sglang.png
          :alt: SGLang VLM Trend reward result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_ppo_openpi.yaml``
     - OpenPI π₀ · ManiSkill PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_ppo_openpi.png
          :alt: OpenPI π₀ · ManiSkill PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_ppo_openpi_pi05.yaml``
     - OpenPI π₀.₅ · ManiSkill PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_ppo_openpi_pi05.png
          :alt: OpenPI π₀.₅ · ManiSkill PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_ppo_openvla.yaml``
     - OpenVLA · ManiSkill PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_ppo_openvla.png
          :alt: OpenVLA · ManiSkill PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_ppo_openvlaoft.yaml``
     - OpenVLA-OFT · ManiSkill PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_ppo_openvlaoft.png
          :alt: OpenVLA-OFT · ManiSkill PPO result curve
          :width: 95%
          :align: center
   * - ``examples/sft/config/maniskill_rlt_stage1_sft_openpi_pi05.yaml``
     - RLT Stage 1 · ManiSkill
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_rlt_stage1_sft_openpi_pi05.png
          :alt: RLT Stage 1 · ManiSkill result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml``
     - RLT Stage 2 AC · ManiSkill
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_rlt_stage2_ac_mlp.png
          :alt: RLT Stage 2 AC · ManiSkill result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_rlt_stage2_td3_mlp.yaml``
     - RLT Stage 2 TD3 · ManiSkill
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_rlt_stage2_td3_mlp.png
          :alt: RLT Stage 2 TD3 · ManiSkill result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_sac_mlp.yaml``
     - MLP · ManiSkill SAC
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_sac_mlp.png
          :alt: MLP · ManiSkill SAC result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_sac_mlp_async_decoupled.yaml``
     - MLP · ManiSkill SAC Async Decoupled
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_sac_mlp_async_decoupled.png
          :alt: MLP · ManiSkill SAC Async Decoupled result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_sac_mlp_resnet_reward_async.yaml``
     - ResNet reward · ManiSkill SAC Async
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_sac_mlp_resnet_reward_async.png
          :alt: ResNet reward · ManiSkill SAC Async result curve
          :width: 95%
          :align: center
   * - ``examples/reasoning/config/math/moonlight-16b-grpo-megatron.yaml``
     - Moonlight-16B · GRPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/moonlight-16b-grpo-megatron.png
          :alt: Moonlight-16B · GRPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/opensora_libero_spatial_grpo_openvlaoft.yaml``
     - OpenSora · OpenVLA-OFT · LIBERO-Spatial GRPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/opensora_libero_spatial_grpo_openvlaoft.png
          :alt: OpenSora · OpenVLA-OFT · LIBERO-Spatial GRPO result curve
          :width: 95%
          :align: center
   * - ``examples/agent/agentlightning/calc_x/config/qwen2.5-1.5b-enginehttp-multiturn.yaml``
     - AgentLightning · Qwen2.5-1.5B
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/qwen2.5-1.5b-enginehttp-multiturn.png
          :alt: AgentLightning · Qwen2.5-1.5B result curve
          :width: 95%
          :align: center
   * - ``examples/reasoning/config/math/qwen2.5-1.5b-ppo-megatron.yaml``
     - Qwen2.5-1.5B · PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/qwen2.5-1.5b-ppo-megatron.png
          :alt: Qwen2.5-1.5B · PPO result curve
          :width: 95%
          :align: center
   * - ``examples/sft/config/qwen2_5_vl_megatron_sft_vlm.yaml``
     - Qwen2.5-VL · Megatron SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/qwen2_5_vl_megatron_sft_vlm.png
          :alt: Qwen2.5-VL · Megatron SFT result curve
          :width: 95%
          :align: center
   * - ``examples/sft/config/qwen2_5_vl_sft_vlm.yaml``
     - Qwen2.5-VL · SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/qwen2_5_vl_sft_vlm.png
          :alt: Qwen2.5-VL · SFT result curve
          :width: 95%
          :align: center
   * - ``examples/sft/config/qwen3_vl_sft_vlm.yaml``
     - Qwen3-VL · SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/qwen3_vl_sft_vlm.png
          :alt: Qwen3-VL · SFT result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/realsimulator_robotwin_adjust_bottle_dagger_openpi.yaml``
     - OpenPI · RoboTwin DAgger
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/realsimulator_robotwin_adjust_bottle_dagger_openpi.png
          :alt: OpenPI · RoboTwin DAgger result curve
          :width: 95%
          :align: center
   * - ``examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml``
     - RLT Stage 1 · Franka
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/realworld_rlt_stage1_sft_openpi_pi05.png
          :alt: RLT Stage 1 · Franka result curve
          :width: 95%
          :align: center
   * - ``examples/sft/config/realworld_sft_openpi_dual_franka_tcp_rot6d.yaml``
     - OpenPI PyTorch · 双臂 Franka SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/realworld_sft_openpi_pytorch_dual_franka_tcp_rot6d.png
          :alt: OpenPI PyTorch · 双臂 Franka SFT result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/robocasa365_opendrawer_ppo_openpi.yaml``
     - OpenPI · RoboCasa365 PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/robocasa365_opendrawer_ppo_openpi.png
          :alt: OpenPI · RoboCasa365 PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/robotwin_beat_block_hammer_grpo_openvlaoft.yaml``
     - OpenVLA-OFT · RoboTwin GRPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/robotwin_beat_block_hammer_grpo_openvlaoft.png
          :alt: OpenVLA-OFT · RoboTwin GRPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/robotwin_place_empty_cup_ppo_openvlaoft.yaml``
     - OpenVLA-OFT · RoboTwin PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/robotwin_place_empty_cup_ppo_openvlaoft.png
          :alt: OpenVLA-OFT · RoboTwin PPO result curve
          :width: 95%
          :align: center
   * - ``examples/sft/config/robotwin_sft_openpi_rlinf.yaml``
     - OpenPI_RLinf · RoboTwin SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/robotwin_sft_openpi_rlinf.png
          :alt: OpenPI_RLinf · RoboTwin SFT result curve
          :width: 95%
          :align: center
   * - ``examples/diffusion/config/sd3_nft_ocr.yaml``
     - Diffusion-NFT · SD3 · OCR
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/sd3_nft_ocr.png
          :alt: Diffusion-NFT · SD3 · OCR result curve
          :width: 95%
          :align: center
   * - ``examples/offline_rl/config/steam_compute_advantages_ensemble.yaml``
     - STEAM advantage ensemble
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/steam_compute_advantages_ensemble.png
          :alt: STEAM advantage ensemble result curve
          :width: 95%
          :align: center
   * - ``examples/offline_rl/config/steam_value_model_sft.yaml``
     - STEAM value model SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/steam_value_model_sft.png
          :alt: STEAM value model SFT result curve
          :width: 95%
          :align: center
   * - ``examples/agent/searchr1/config/train_qwen2.5.yaml``
     - SearchR1 · Qwen2.5
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/train_qwen2.5.png
          :alt: SearchR1 · Qwen2.5 result curve
          :width: 95%
          :align: center
   * - ``examples/agent/searchr1/config/train_qwen2.5_fsdp.yaml``
     - SearchR1 · Qwen2.5 FSDP
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/train_qwen2.5_fsdp.png
          :alt: SearchR1 · Qwen2.5 FSDP result curve
          :width: 95%
          :align: center
   * - ``examples/diffusion/config/wan22_ti2v_5b_nft_ocr.yaml``
     - Diffusion-NFT · Wan2.2 · OCR
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/wan22_ti2v_5b_nft_ocr.png
          :alt: Diffusion-NFT · Wan2.2 · OCR result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/wan_libero_spatial_grpo_openvlaoft.yaml``
     - Wan · OpenVLA-OFT · LIBERO-Spatial GRPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/wan_libero_spatial_grpo_openvlaoft.png
          :alt: Wan · OpenVLA-OFT · LIBERO-Spatial GRPO result curve
          :width: 95%
          :align: center
