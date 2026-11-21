环境要求
========

按任务规模选择对应硬件。你可以从单张 24 GB GPU 起步，无需修改代码就能将同一套配置扩展到多机集群。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/hardware_tiers_zh.svg
   :align: center
   :width: 100%

   RLinf 从单卡到多机集群的硬件档位。

按场景划分的 GPU 档位
---------------------

.. list-table::
   :header-rows: 1
   :widths: 22 44 34

   * - 档位
     - 适用任务
     - 推荐入口
   * - 单卡（24 GB 起）
     - MLP / CNN 仿真 RL、VLA 评测、LoRA SFT、真机数据采集与部署
     - :doc:`评测快速体验 <../evaluations/get_started/quick_tour>`、:doc:`MLP 强化学习 <../examples/embodied/mlp>`
   * - 1–2 卡（40–80 GB）
     - 单机 OpenVLA / OpenPI 快速上手、共享式 PPO / GRPO
     - :doc:`VLA 快速上手 <vla>`
   * - 4–8 卡（80 GB）
     - VLA 全参 SFT 与基准复现（LIBERO、ManiSkill、RoboTwin）、7B–14B 推理 RL
     - :doc:`示例库 <../examples/index>`
   * - 多机集群（8×H100 起）
     - 大规模分离式与混合式训练、Megatron + SGLang / vLLM、世界模型
     - :doc:`多节点训练 <../guides/multi_node>`

除 NVIDIA GPU 外，RLinf 还支持 AMD ROCm、华为昇腾 CANN、摩尔线程 MUSA、Intel XPU、昆仑芯 XPU、沐曦 MACA 与壁仞 SUPA。已验证的模型与环境组合见 :ref:`硬件支持矩阵 <model-hardware-support>`。

软件环境
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 组件
     - 支持 / 测试版本
   * - 操作系统
     - Linux（已在 Ubuntu 22.04 上测试）
   * - Python
     - 3.10–3.12（由 ``requirements/install.sh`` 自动管理）
   * - NVIDIA 驱动与 CUDA
     - 驱动 535+（仿真渲染需支持图形与 EGL）；CUDA 12.4+
   * - 容器（可选）
     - Docker 26+ 与 NVIDIA Container Toolkit（预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__）

大规模参考集群配置
------------------

多机大规模训练与完整基准复现通常在以下集群配置上验证：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 组件
     - 参考配置
   * - GPU
     - 每节点 8×H100
   * - CPU
     - 每节点 192 核
   * - 内存
     - 每节点 1.8 TB
   * - 网络
     - NVLink + RoCE / IB（3.2 Tbps）
   * - 存储
     - 单机 1 TB 本地盘 / 多机 10 TB 共享存储（NAS）
