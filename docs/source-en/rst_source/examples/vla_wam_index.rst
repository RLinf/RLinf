Embodied Models
===============

Choose a model or policy class to find its installation, checkpoint, and RL training workflow. For examples organized by benchmark, see :doc:`simulators_index`.

.. _model-hardware-support:

Hardware Support
----------------

Model recipes default to NVIDIA CUDA. AMD ROCm, Huawei Ascend CANN, Moore Threads MUSA, Intel XPU, Kunlunxin XPU, MetaX MACA, and Biren SUPA also support the model and environment combinations below. Follow a model or hardware link for the training recipe and available backend setup instructions. On non-CUDA backends, ManiSkill runs PhysX on CPU and selects the renderer independently.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Model
     - Hardware
     - Environment and Scope
   * - :doc:`DreamZero <embodied/sft_dreamzero>`
     - :ref:`Huawei Ascend CANN <sft-dreamzero-hardware>`
     - SFT (LeRobot datasets)
   * - :doc:`GR00T N1.5 / N1.7 <embodied/gr00t>`
     - :ref:`AMD ROCm · Huawei Ascend CANN · Moore Threads MUSA <gr00t-hardware>` · Intel XPU
     - N1.5 on LIBERO · ManiSkill (all four) · N1.7 on LIBERO · ManiSkill (Intel XPU); ManiSkill requires a ``maniskill_widowx`` checkpoint
   * - :doc:`LingbotVLA 1.0 <embodied/lingbotvla>`
     - :ref:`AMD ROCm <robotwin-hardware>`
     - :ref:`RoboTwin <robotwin-hardware>`
   * - :doc:`OpenVLA <embodied/maniskill>`
     - AMD ROCm · Huawei Ascend CANN · Moore Threads MUSA · MetaX MACA
     - ManiSkill
   * - :doc:`OpenVLA-OFT <embodied/openvla_oft>`
     - :ref:`AMD ROCm · Huawei Ascend CANN · Moore Threads MUSA <openvla-oft-hardware>` · Kunlunxin XPU
     - LIBERO · ManiSkill (AMD, Ascend, MUSA) · :ref:`RoboTwin on ROCm <robotwin-hardware>` · :ref:`Wan world model <wan-hardware>` (Ascend, MUSA, ROCm, Kunlunxin)
   * - :doc:`π₀ / π₀.₅ (OpenPI) <embodied/pi0>`
     - :ref:`AMD ROCm · Huawei Ascend CANN · Moore Threads MUSA <pi0-hardware>` · Intel XPU · Kunlunxin XPU · MetaX MACA · Biren SUPA
     - LIBERO (AMD, Ascend, MUSA, Intel, Kunlunxin, MetaX) · ManiSkill (AMD, Ascend, MUSA, Intel) · :ref:`RoboTwin on ROCm <robotwin-hardware>` · :doc:`SFT <embodied/sft_openpi>` (Biren SUPA)
   * - :doc:`StarVLA (QwenOFT) <embodied/starvla>`
     - :ref:`Huawei Ascend CANN <starvla-hardware>`
     - LIBERO

Model Recipes
-------------

Open a model page for the full workflow and its supported environments.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: ABot-M0
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/ABot-M0.png
      :img-alt: ABot-M0 model
      :link: embodied/abot_m0
      :link-type: doc

      Fine-tune ABot-M0 with PPO on standard LIBERO and LIBERO-Plus.

   .. grid-item-card:: Dexbotic
      :img-top: https://raw.githubusercontent.com/dexmal/dexbotic/main/resources/intro.png
      :img-alt: Dexbotic model
      :link: embodied/dexbotic
      :link-type: doc

      Train the π₀.₅-based Dexbotic policy with PPO on LIBERO.

   .. grid-item-card:: Evo-1
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/evo1.png
      :img-alt: Evo-1 model
      :link: embodied/evo1
      :link-type: doc

      Train the Evo-1 vision-language-action model with embodied RL.

   .. grid-item-card:: GR00T
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/gr00t.png
      :img-alt: NVIDIA GR00T models
      :link: embodied/gr00t
      :link-type: doc

      Fine-tune GR00T N1.5, N1.6, and N1.7 with PPO on LIBERO, ManiSkill, and IsaacLab.

   .. grid-item-card:: LingbotVLA
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/lingbotvla.png
      :img-alt: LingbotVLA models
      :link: embodied/lingbotvla_index
      :link-type: doc

      Train LingbotVLA 1.0 (GRPO) and 2.0 (PPO) on RoboTwin manipulation tasks.

   .. grid-item-card:: MLP Policy
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/3_layer_mlp.jpg
      :img-alt: MLP policy network
      :link: embodied/mlp
      :link-type: doc

      Train a lightweight MLP policy with PPO, SAC, or GRPO across simulation environments.

   .. grid-item-card:: MolmoAct2 (Eval Only)
      :img-top: https://raw.githubusercontent.com/allenai/molmoact2/main/assets/MolmoAct2.svg
      :img-alt: MolmoAct2 logo
      :link: embodied/molmoact2
      :link-type: doc

      Evaluate the official MolmoAct2-LIBERO checkpoint across LIBERO suites.

   .. grid-item-card:: OpenVLA-OFT
      :img-top: https://openvla-oft.github.io/static/images/libero_task_performance_results.png
      :img-alt: OpenVLA-OFT results
      :link: embodied/openvla_oft
      :link-type: doc

      Train OpenVLA-OFT on LIBERO and ManiSkill across NVIDIA CUDA, AMD ROCm, Huawei Ascend CANN, and Moore Threads MUSA.

   .. grid-item-card:: π₀ / π₀.₅ (OpenPI)
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
      :img-alt: OpenPI π₀ and π₀.₅ models
      :link: embodied/pi0
      :link-type: doc

      Fine-tune π₀ and π₀.₅ with Flow-SDE or Flow-Noise PPO across LIBERO, ManiSkill, RoboTwin, CALVIN, and MetaWorld.

   .. grid-item-card:: π₀-FAST
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
      :img-alt: π₀-FAST model
      :link: embodied/pi0_fast
      :link-type: doc

      Run LeRobot π₀-FAST deterministic evaluation and token-level GRPO fine-tuning on LIBERO-Long.

   .. grid-item-card:: Psi0
      :img-top: https://raw.githubusercontent.com/physical-superintelligence-lab/Psi0/a32e57a3fabb8590c80677f9cd3d1fc3db60eb06/assets/media/arch.png
      :img-alt: Psi0 model architecture
      :link: embodied/psi0
      :link-type: doc

      Train Psi0 with PPO on SIMPLE humanoid manipulation tasks.

   .. grid-item-card:: StarVLA
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/starvla.png
      :img-alt: StarVLA models
      :link: embodied/starvla
      :link-type: doc

      Fine-tune StarVLA models with GRPO on LIBERO benchmarks.

.. toctree::
   :hidden:
   :maxdepth: 2

   ABot-M0 <embodied/abot_m0>
   Dexbotic <embodied/dexbotic>
   Evo-1 <embodied/evo1>
   GR00T <embodied/gr00t>
   LingbotVLA <embodied/lingbotvla_index>
   MLP <embodied/mlp>
   MolmoAct2 (Eval Only) <embodied/molmoact2>
   OpenVLA-OFT <embodied/openvla_oft>
   π₀ / π₀.₅ <embodied/pi0>
   π₀-FAST <embodied/pi0_fast>
   Psi0 <embodied/psi0>
   StarVLA <embodied/starvla>
