为什么选择 RLinf
================

RLinf 是一个面向具身智能、智能体与生成式基础模型的开源分布式强化学习框架。本页概括框架的设计理念、实测吞吐收益与工作负载覆盖范围。

RLinf 的核心设计
----------------

- **宏观到微观流程（M2Flow）**：将逻辑 RL 工作流编排与底层物理通信、硬件调度解耦（见 :doc:`publications/rlinf_system`）。
- **灵活执行模式**：无需修改训练循环，直接通过 YAML 在 *共享式*、*分离式* 与 *混合式* 部署间切换（见 :doc:`../concepts/execution_modes`）。
- **自动与动态调度**：支持静态模式自动选型与训练期在线弹性扩缩，在保持 on-policy 语义的前提下提升集群利用率（见 :doc:`publications/dynarl` 与 :doc:`../guides/dynamic_scheduling`）。
- **完整的工作负载覆盖（v0.4+）**：

  - *具身仿真*：覆盖 LIBERO、ManiSkill3、RoboTwin、IsaacLab、BEHAVIOR、CALVIN、MetaWorld 与 Genesis 等模拟器，适配 OpenVLA、OpenVLA-OFT、π₀ / π₀.₅、GR00T、LingBot-VLA 及 MLP/CNN policy（见 :doc:`../examples/simulators_index`、:doc:`../examples/vla_wam_index`）。
  - *真机机器人*：通过统一的 ``rlinf.robotics`` 硬件接口支持 Franka、双臂 Franka、SO-101、Piper、Turtle2、YAM 等平台的在线 RL、DAgger 与遥操作（见 :doc:`../examples/real_world_index`）。
  - *世界模型与生成式 RL*：支持 WoVR / Wan、OpenSora、DreamZero、Cosmos3 等世界模型 RL 与扩散/视频生成后训练（见 :doc:`../examples/world_model_index`）。
  - *推理与智能体*：支持数学推理、VQA 以及 Search-R1、WideSeek-R1、rStar2、在线代码 RL 与 AgentLightning 等多轮工具调用智能体（见 :doc:`../examples/agentic/index`）。
  - *SFT 与离线 RL*：支持有监督微调、IQL、RECAP、STEAM 与仿真实协同训练（见 :doc:`../examples/sft_index`、:doc:`../examples/methods_index`）。

- **多加速器支持**：在同一套 ``Worker`` 抽象下支持 NVIDIA GPU、AMD ROCm、昇腾 Ascend、摩尔线程 MUSA、昆仑芯与壁仞加速器（见 :doc:`../start/installation`）。

实测性能
--------

- **端到端训练加速**：M2Flow 结合混合式部署与细粒度流水线，相比同类 RL 框架获得 **1.07×–2.43×** 端到端吞吐提升——数学推理相对 veRL 与 Slime 最高提升 **1.7×**，具身 VLA 训练最高提升 **2.43×**\ （详见 :doc:`publications/rlinf_system` 与 :doc:`blog/compare_with_verl`）。
- **在线动态扩缩**：DynaRL 在数秒内完成 worker 资源重分配，在保持 on-policy 语义的同时将动态负载效率再提升 **20–40%**\ （详见 :doc:`publications/dynarl`）。

统一后端与算法
--------------

- **可切换训练与推理引擎**：可用 **FSDP + Hugging Face** 快速迭代具身与科研实验，也可用 **Megatron + SGLang / vLLM**\ （配合 :doc:`../extending/mbridge`）支撑大规模 LLM 与 VLM 训练。
- **内置算法**：在同一套优势函数与损失注册表下提供 PPO、Async PPO、GRPO、DAPO、REINFORCE++、SAC、CrossQ、RLPD 与 IQL（见 :doc:`../reference/algorithms/index`）。
- **配置与指标参考**：Hydra YAML 结构与训练指标定义见 :doc:`../reference/training/index`。

可复现的端到端配方
------------------

RLinf 提供可直接运行的配置与启动脚本，用于复现论文级实验结果：

- **具身基准**：提供 **LIBERO**、**ManiSkill**、**RoboTwin** 以及 Franka / Turtle2 真机训练与独立评测配方（见 :doc:`../examples/index`、:doc:`../evaluations/index`）。
- **推理与智能体基准**：提供基于 DeepSeek-R1-Distill-Qwen 系列的 **AIME24 / AIME25 / GPQA-diamond** 数学推理配方，以及单智能体与多智能体检索和代码训练流程（见 :doc:`../examples/agentic/index`）。
