SFT 与模仿学习
==============

需要用监督微调或在线模仿学习为 VLA、WAM 或 VLM 冷启动时，从本节进入。完成 SFT 后，可继续查阅 :doc:`vla_wam_index` 或 :doc:`simulators_index` 用强化学习微调得到的 checkpoint。

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Cosmos3 监督微调
      :img-top: https://raw.githubusercontent.com/NVIDIA/cosmos/main/cookbooks/cosmos3/cosmos3-model-architecture.png
      :img-alt: Cosmos3 架构
      :link: embodied/sft_cosmos3
      :link-type: doc

      在 NVIDIA Cosmos3-Nano（OmniMoT）上训练 LIBERO 动作 policy SFT。

   .. grid-item-card:: DAgger
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/dagger.jpg
      :img-alt: DAgger 模仿学习
      :link: embodied/dagger
      :link-type: doc

      结合专家重标注与 replay buffer 更新完成在线模仿学习。

   .. grid-item-card:: DreamZero 监督微调
      :img-top: https://dreamzero0.github.io/images/project_overview.png
      :img-alt: DreamZero 概览
      :link: embodied/sft_dreamzero
      :link-type: doc

      基于 WAN2.1 和 WAN2.2 主干的 DreamZero 全量与混合数据集 SFT。

   .. grid-item-card:: FastWAM
      :img-top: https://yuantianyuan01.github.io/FastWAM/static/images/teaser_main.png
      :img-alt: FastWAM 示意
      :link: embodied/sft_fastwam
      :link-type: doc

      在 LIBERO 上评测 FastWAM 并运行全参数 FSDP SFT。

   .. grid-item-card:: OpenPI 监督微调
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
      :img-alt: OpenPI 监督微调
      :link: embodied/sft_openpi
      :link-type: doc

      在 LIBERO、Franka、BEHAVIOR-1K、RoboTwin 和自定义 LeRobot 数据集上对 π₀ 与 π₀.₅ 做全量或 LoRA SFT。

   .. grid-item-card:: Qwen-VL 监督微调
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/release_0.2/qwen2_5_sft_vlm.png
      :img-alt: Qwen-VL 监督微调
      :link: embodied/sft_vlm
      :link-type: doc

      对 Qwen2.5-VL、Qwen3-VL 和 Qwen3-VL-MoE 做全量 SFT 与评估。

.. toctree::
   :hidden:
   :maxdepth: 2

   Cosmos3 <embodied/sft_cosmos3>
   DAgger <embodied/dagger>
   DreamZero <embodied/sft_dreamzero>
   FastWAM <embodied/sft_fastwam>
   OpenPI <embodied/sft_openpi>
   Qwen-VL <embodied/sft_vlm>
