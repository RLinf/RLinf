为什么选择 RLinf
=================================

RLinf 专为大规模通过强化学习对基础模型进行后训练而构建。本页汇总了框架背后的设计理念、性能数据与 SOTA 复现配方——当你想了解“为什么”而非“怎么做”时，请阅读本页。

RLinf 的独特之处
---------------------------------

- **宏观到微观流程（M2Flow）：** 一种新范式，通过微观级的执行流程完成宏观级的逻辑流程，将逻辑工作流构建（可编程）与物理通信调度（高效执行）解耦。

- **灵活的执行模式：**

  - *共享式* — 所有任务共享全部 GPU。
  - *分离式* — 支持细粒度流水线。
  - *混合式* — 可定制地组合共享式与分离式两种模式。

- **自动调度：**

  - *动态调度* — 即时调度资源分配，最大化资源利用率。
  - *静态调度* — 根据训练任务自动选择最合适的执行模式，无需手动分配资源。

- **具身智能支持：**

  - 快速适配主流 VLA 模型：`OpenVLA`_、`OpenVLA-OFT`_、`π₀`_、`GR00T-N1.5`_。
  - 通过标准化 RL 接口支持主流 CPU 与 GPU 模拟器：`ManiSkill3`_、`LIBERO`_、`IsaacLab`_。
  - 支持 π₀ 模型族首次基于 flow-matching 动作专家的强化学习微调。

性能
---------------------------------

- **结合细粒度流水线的混合式** 相比同类框架实现 **120%+** 的吞吐率提升。
- **自动在线扩缩** 动态扩展训练资源，GPU 切换只需数秒，在保持 RL 算法 on-policy 特性的同时进一步提升效率 **20–40%**。

灵活且易用
---------------------------------

- **多后端集成**，在统一接口背后无需修改代码即可切换：

  - *FSDP + Hugging Face* — 快速适配新模型与算法，适合初学者与快速原型开发。
  - *Megatron + SGLang* — 面向大规模训练优化，为高负载场景提供极致效率。

- 通过异步通信通道实现自适应通信。
- 内建多种强化学习方法，包括 `PPO`_、`GRPO`_、`DAPO`_、`Reinforce++`_ 等。

SOTA 强化学习复现
---------------------------------

RLinf 提供开箱即用的端到端配方，可复现或达到 **业界领先（SOTA）的 RL 结果**——直接运行官方配置与脚本即可获得论文级数据，无需额外工程改造。

- **具身智能任务：** RLinf 在 **LIBERO**、**ManiSkill**、**RoboTwin** 等基准上，使用 OpenVLA、OpenVLA-OFT、π₀/π₀.₅、GR00T 等 VLA 模型达到或接近 SOTA 成功率。详见 :doc:`../examples/index` 示例库与 :doc:`../reference/index` 算法说明。
- **智能体任务（含数学推理）：** RLinf 基于 DeepSeek-R1-Distill-Qwen 系列模型，在 **AIME24 / AIME25 / GPQA-diamond** 上达到 SOTA 表现，并支持 Search-R1、在线代码补全等单智能体与多智能体任务。详见 :doc:`../examples/agentic/index`。

.. _PPO: https://arxiv.org/abs/1707.06347
.. _GRPO: https://arxiv.org/abs/2402.03300
.. _DAPO: https://arxiv.org/abs/2503.14476
.. _Reinforce++: https://arxiv.org/abs/2501.03262
.. _OpenVLA: https://github.com/openvla/openvla
.. _OpenVLA-OFT: https://github.com/moojink/openvla-oft
.. _ManiSkill3: https://github.com/haosulab/ManiSkill
.. _LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO
.. _IsaacLab: https://github.com/isaac-sim/IsaacLab
.. _π₀: https://github.com/Physical-Intelligence/openpi
.. _GR00T-N1.5: https://github.com/NVIDIA/Isaac-GR00T.git
