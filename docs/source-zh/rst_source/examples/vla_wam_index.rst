具身模型
========

按模型或 policy 类型查找安装、checkpoint 和强化学习训练流程。若从基准出发选择示例，请参阅 :doc:`simulators_index`。

.. _model-hardware-support:

硬件支持
--------

模型示例默认使用 NVIDIA CUDA。AMD ROCm、华为昇腾 CANN、摩尔线程 MUSA、Intel XPU、昆仑芯 XPU、沐曦 MACA 和壁仞 SUPA 也支持下列模型与环境组合。点击模型或硬件链接查看训练示例及已提供的硬件配置说明。非 CUDA 后端上，ManiSkill 用 CPU 运行 PhysX，渲染后端单独配置。

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - 模型
     - 硬件
     - 环境与范围
   * - :doc:`DreamZero <embodied/sft_dreamzero>`
     - :ref:`华为昇腾 CANN <sft-dreamzero-hardware>`
     - SFT（LeRobot 数据集）
   * - :doc:`GR00T N1.5 / N1.7 <embodied/gr00t>`
     - :ref:`AMD ROCm · 华为昇腾 CANN · 摩尔线程 MUSA <gr00t-hardware>` · Intel XPU
     - N1.5：LIBERO · ManiSkill（全部四种后端）· N1.7：LIBERO · ManiSkill（Intel XPU）；ManiSkill 需要带有 ``maniskill_widowx`` head 的 checkpoint
   * - :doc:`LingbotVLA 1.0 <embodied/lingbotvla>`
     - :ref:`AMD ROCm <robotwin-hardware>`
     - :ref:`RoboTwin <robotwin-hardware>`
   * - :doc:`OpenVLA <embodied/maniskill>`
     - AMD ROCm · 华为昇腾 CANN · 摩尔线程 MUSA · 沐曦 MACA
     - ManiSkill
   * - :doc:`OpenVLA-OFT <embodied/openvla_oft>`
     - :ref:`AMD ROCm · 华为昇腾 CANN · 摩尔线程 MUSA <openvla-oft-hardware>` · 昆仑芯 XPU
     - LIBERO · ManiSkill（AMD、昇腾、MUSA）· :ref:`RoboTwin（ROCm） <robotwin-hardware>` · :ref:`Wan 世界模型 <wan-hardware>`\ （昇腾、MUSA、ROCm、昆仑芯）
   * - :doc:`π₀ / π₀.₅ (OpenPI) <embodied/pi0>`
     - :ref:`AMD ROCm · 华为昇腾 CANN · 摩尔线程 MUSA <pi0-hardware>` · Intel XPU · 昆仑芯 XPU · 沐曦 MACA · 壁仞 SUPA
     - LIBERO（AMD、昇腾、MUSA、Intel、昆仑芯、沐曦）· ManiSkill（AMD、昇腾、MUSA、Intel）· :ref:`RoboTwin（ROCm） <robotwin-hardware>` · :doc:`SFT <embodied/sft_openpi>`\ （壁仞 SUPA）
   * - :doc:`StarVLA (QwenOFT) <embodied/starvla>`
     - :ref:`华为昇腾 CANN <starvla-hardware>`
     - LIBERO

模型示例
--------

选择模型页面，查看完整流程及其支持的环境。

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: ABot-M0
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/ABot-M0.png
      :img-alt: ABot-M0 模型
      :link: embodied/abot_m0
      :link-type: doc

      在标准 LIBERO 和 LIBERO-Plus 上用 PPO 微调 ABot-M0。

   .. grid-item-card:: Dexbotic
      :img-top: https://raw.githubusercontent.com/dexmal/dexbotic/main/resources/intro.png
      :img-alt: Dexbotic 模型
      :link: embodied/dexbotic
      :link-type: doc

      在 LIBERO 上用 PPO 训练基于 π₀.₅ 的 Dexbotic policy。

   .. grid-item-card:: Evo-1
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/evo1.png
      :img-alt: Evo-1 模型
      :link: embodied/evo1
      :link-type: doc

      用具身强化学习训练 Evo-1 视觉-语言-动作模型。

   .. grid-item-card:: GR00T
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/gr00t.png
      :img-alt: NVIDIA GR00T 模型
      :link: embodied/gr00t
      :link-type: doc

      在 LIBERO、ManiSkill 和 IsaacLab 上用 PPO 微调 GR00T N1.5、N1.6 和 N1.7。

   .. grid-item-card:: LingbotVLA
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/lingbotvla.png
      :img-alt: LingbotVLA 模型
      :link: embodied/lingbotvla_index
      :link-type: doc

      在 RoboTwin 操作任务上训练 LingbotVLA 1.0（GRPO）与 2.0（PPO）。

   .. grid-item-card:: MLP Policy
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/3_layer_mlp.jpg
      :img-alt: MLP policy 网络
      :link: embodied/mlp
      :link-type: doc

      在多种仿真环境中用 PPO、SAC 或 GRPO 训练轻量级 MLP policy。

   .. grid-item-card:: MolmoAct2（仅评测）
      :img-top: https://raw.githubusercontent.com/allenai/molmoact2/main/assets/MolmoAct2.svg
      :img-alt: MolmoAct2 标识
      :link: embodied/molmoact2
      :link-type: doc

      在 LIBERO 套件上评测官方 MolmoAct2-LIBERO checkpoint。

   .. grid-item-card:: OpenVLA-OFT
      :img-top: https://openvla-oft.github.io/static/images/libero_task_performance_results.png
      :img-alt: OpenVLA-OFT 结果
      :link: embodied/openvla_oft
      :link-type: doc

      在 NVIDIA CUDA、AMD ROCm、华为昇腾 CANN 和摩尔线程 MUSA 上训练 OpenVLA-OFT。

   .. grid-item-card:: π₀ / π₀.₅（OpenPI）
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
      :img-alt: OpenPI π₀ 与 π₀.₅ 模型
      :link: embodied/pi0
      :link-type: doc

      在 LIBERO、ManiSkill、RoboTwin、CALVIN 和 MetaWorld 上用 Flow-SDE 或 Flow-Noise PPO 微调 π₀ 和 π₀.₅。

   .. grid-item-card:: π₀-FAST
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
      :img-alt: π₀-FAST 模型
      :link: embodied/pi0_fast
      :link-type: doc

      在 LIBERO-Long 上运行 LeRobot π₀-FAST 确定性评测与 token-level GRPO 微调。

   .. grid-item-card:: Psi0
      :img-top: https://raw.githubusercontent.com/physical-superintelligence-lab/Psi0/a32e57a3fabb8590c80677f9cd3d1fc3db60eb06/assets/media/arch.png
      :img-alt: Psi0 模型架构
      :link: embodied/psi0
      :link-type: doc

      在 SIMPLE 人形机器人操作任务上用 PPO 训练 Psi0。

   .. grid-item-card:: StarVLA
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/starvla.png
      :img-alt: StarVLA 模型
      :link: embodied/starvla
      :link-type: doc

      在 LIBERO 基准上用 GRPO 微调 StarVLA 模型。

.. toctree::
   :hidden:
   :maxdepth: 2

   ABot-M0 <embodied/abot_m0>
   Dexbotic <embodied/dexbotic>
   Evo-1 <embodied/evo1>
   GR00T <embodied/gr00t>
   LingbotVLA <embodied/lingbotvla_index>
   MLP <embodied/mlp>
   MolmoAct2（仅评测） <embodied/molmoact2>
   OpenVLA-OFT <embodied/openvla_oft>
   π₀ / π₀.₅ <embodied/pi0>
   π₀-FAST <embodied/pi0_fast>
   Psi0 <embodied/psi0>
   StarVLA <embodied/starvla>
