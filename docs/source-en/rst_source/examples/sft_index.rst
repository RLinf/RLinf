SFT & Imitation
===============

Pick this section to cold-start a VLA, WAM, or VLM checkpoint with supervised fine-tuning or online imitation learning. After SFT, continue to :doc:`vla_wam_index` or :doc:`simulators_index` for RL fine-tuning.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Cosmos3 SFT
      :img-top: https://raw.githubusercontent.com/NVIDIA/cosmos/main/cookbooks/cosmos3/cosmos3-model-architecture.png
      :img-alt: Cosmos3 architecture
      :link: embodied/sft_cosmos3
      :link-type: doc

      Run LIBERO action-policy SFT on NVIDIA Cosmos3-Nano (OmniMoT).

   .. grid-item-card:: DAgger
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/dagger.jpg
      :img-alt: DAgger imitation learning
      :link: embodied/dagger
      :link-type: doc

      Guide online imitation learning with expert relabeling and replay-buffer updates.

   .. grid-item-card:: DreamZero SFT
      :img-top: https://dreamzero0.github.io/images/project_overview.png
      :img-alt: DreamZero overview
      :link: embodied/sft_dreamzero
      :link-type: doc

      Full-parameter and mixture SFT for DreamZero with WAN2.1 and WAN2.2 backbones.

   .. grid-item-card:: FastWAM
      :img-top: https://yuantianyuan01.github.io/FastWAM/static/images/teaser_main.png
      :img-alt: FastWAM teaser
      :link: embodied/sft_fastwam
      :link-type: doc

      Evaluate FastWAM on LIBERO and run full-parameter FSDP SFT.

   .. grid-item-card:: OpenPI SFT
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
      :img-alt: OpenPI supervised fine-tuning
      :link: embodied/sft_openpi
      :link-type: doc

      Run full-parameter or LoRA SFT on π₀ and π₀.₅ across LIBERO, Franka, BEHAVIOR-1K, RoboTwin, and custom LeRobot datasets.

   .. grid-item-card:: Qwen-VL SFT
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/release_0.2/qwen2_5_sft_vlm.png
      :img-alt: Qwen-VL SFT
      :link: embodied/sft_vlm
      :link-type: doc

      Run full-parameter SFT and evaluation for Qwen2.5-VL, Qwen3-VL, and Qwen3-VL-MoE.

.. toctree::
   :hidden:
   :maxdepth: 2

   Cosmos3 <embodied/sft_cosmos3>
   DAgger <embodied/dagger>
   DreamZero <embodied/sft_dreamzero>
   FastWAM <embodied/sft_fastwam>
   OpenPI <embodied/sft_openpi>
   Qwen-VL <embodied/sft_vlm>
