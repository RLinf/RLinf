模拟器
======

从评测基准或仿真平台切入时，使用本节示例。每篇文档涵盖环境安装、资产路径、观测与动作空间，以及可直接运行的 RL 参考配置。

按模型系列切入（π₀ / π₀.₅、GR00T 等）见 :doc:`vla_wam_index`，真机机器人部署见 :doc:`real_world_index`。使用 AMD ROCm、华为昇腾 CANN 或摩尔线程 MUSA 时，在 :ref:`硬件支持表 <model-hardware-support>` 中选择对应的模型与环境组合。

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: BEHAVIOR-1K
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/behavior.jpg
      :img-alt: BEHAVIOR-1K 基准
      :link: embodied/behavior
      :link-type: doc

      在 OmniGibson 50 个家居任务上用 PPO 微调 OpenVLA-OFT、π₀ 和 π₀.₅。

   .. grid-item-card:: CALVIN
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/calvin.png
      :img-alt: CALVIN 基准
      :link: embodied/calvin
      :link-type: doc

      在 CALVIN 长程场景迁移套件上用 PPO 训练 π₀ 和 π₀.₅。

   .. grid-item-card:: DexJoCo
      :img-top: https://raw.githubusercontent.com/brave-eai/dexjoco/main/docs/pics/water_plant_4.png
      :img-alt: DexJoCo 基准
      :link: embodied/dexjoco
      :link-type: doc

      在灵巧手 WaterPlant 任务上用 SFT 与 PPO 训练 CNN 和 π₀.₅。

   .. grid-item-card:: EmbodiChain
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/embodichain.gif
      :img-alt: EmbodiChain 仿真平台
      :link: embodied/embodichain
      :link-type: doc

      在 EmbodiChain gym 环境中用 PPO 训练 MLP policy。

   .. grid-item-card:: Franka-Sim
      :img-top: https://raw.githubusercontent.com/RLinf/serl/refs/heads/RLinf/franka-sim/franka_sim/franka_sim/envs/xmls/robotiq_2f85/2f85.png
      :img-alt: Franka-Sim 基准
      :link: embodied/frankasim
      :link-type: doc

      在 MuJoCo Franka-Sim 任务上用 PPO 或 SAC 训练 MLP 和 CNN policy。

   .. grid-item-card:: G1 动作跟踪
      :link: embodied/g1_mimic
      :link-type: doc

      MuJoCo Warp 中的 Unitree G1 全身动作跟踪环境。

   .. grid-item-card:: Genesis
      :img-top: https://raw.githubusercontent.com/YilingQiao/Genesis/readme-assets/videos/HeroShot_Final.png
      :img-alt: Genesis 仿真平台
      :link: embodied/genesis
      :link-type: doc

      在 Genesis 物理仿真平台上训练 MLP policy。

   .. grid-item-card:: GenieSim
      :link: embodied/geniesim
      :link-type: doc

      在 GenieSim 双臂操作基准上用 PPO 或 GRPO 训练 π₀.₅。

   .. grid-item-card:: GSEnv
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/gsenv.gif
      :img-alt: GSEnv Real2Sim2Real
      :link: embodied/gsenv
      :link-type: doc

      在高斯泼溅场景中用 PPO 训练 π₀.₅，用于 Real2Sim2Real 迁移。

   .. grid-item-card:: IsaacLab
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/IsaacLab.png
      :img-alt: IsaacLab 基准
      :link: embodied/isaaclab
      :link-type: doc

      在 NVIDIA IsaacLab 仿真环境中用 PPO 训练 GR00T 和 OpenPI policy。

   .. grid-item-card:: LIBERO
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/libero_numbers.jpeg
      :img-alt: LIBERO 基准
      :link: embodied/libero
      :link-type: doc

      在标准 LIBERO（99% 成功率）、LIBERO-Pro、LIBERO-Plus 和 LIBERO-Mem 上训练或评测 OpenVLA-OFT 等 VLA。

   .. grid-item-card:: ManiSkill
      :img-top: https://raw.githubusercontent.com/mani-skill/ManiSkill/main/figures/teaser.jpg
      :img-alt: ManiSkill 基准
      :link: embodied/maniskill
      :link-type: doc

      在 ManiSkill3 桌面操作与 OOD 基准上用 PPO 或 GRPO 训练 OpenVLA 和 OpenVLA-OFT。

   .. grid-item-card:: MetaWorld
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/metaworld.png
      :img-alt: MetaWorld 基准
      :link: embodied/metaworld
      :link-type: doc

      在 MT50 和 ML45 任务上用 PPO 或 GRPO 微调 π₀、π₀.₅ 和 OpenVLA-OFT。

   .. grid-item-card:: MuJoCo Warp
      :img-top: https://raw.githubusercontent.com/google-deepmind/mujoco_warp/main/benchmarks/humanoid/rollout_three_humanoids.webp
      :img-alt: MuJoCo Warp 运动控制
      :link: embodied/mujoco_warp
      :link-type: doc

      在五个 GPU 加速运动控制任务上用 PPO 训练 MLP policy。

   .. grid-item-card:: PolaRiS
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/polaris.png
      :img-alt: PolaRiS 基准
      :link: embodied/polaris
      :link-type: doc

      在 PolaRiS 桌面操作任务上用 PPO 训练 OpenPI policy。

   .. grid-item-card:: 真机仿真器
      :link: embodied/real_simulator
      :link-type: doc

      在仿真训练中注入观测延迟与网络仿真，验证真机部署鲁棒性。

   .. grid-item-card:: RoboCasa
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/robocasa.jpeg
      :img-alt: RoboCasa 基准
      :link: embodied/robocasa
      :link-type: doc

      在 RoboCasa 厨房操作任务上用 GRPO 微调 π₀。

   .. grid-item-card:: RoboCasa365
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/robocasa.jpeg
      :img-alt: RoboCasa365 基准
      :link: embodied/robocasa365
      :link-type: doc

      在 RoboCasa365 基准套件上用 PPO 训练 π₀。

   .. grid-item-card:: RoboDojo（仅评测）
      :img-top: https://media.luminis-sim.com/media/home/teaser.png
      :img-alt: RoboDojo 基准
      :link: embodied/robodojo
      :link-type: doc

      在 RoboDojo 操作基准上评测 π₀.₅ checkpoint。

   .. grid-item-card:: RoboLab（仅评测）
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/robolab.png
      :img-alt: RoboLab 操作基准
      :link: embodied/robolab
      :link-type: doc

      在 RoboLab 的 120 个 DROID 操作任务上评测 π₀.₅。

   .. grid-item-card:: RoboTwin
      :img-top: https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/main/assets/files/50_tasks.gif
      :img-alt: RoboTwin 基准
      :link: embodied/robotwin
      :link-type: doc

      在 RoboTwin 双臂操作任务上用 PPO 或 GRPO 训练 OpenVLA-OFT、π₀ 和 π₀.₅。

   .. grid-item-card:: RoboVerse
      :img-top: https://roboverseorg.github.io/static/images/teaser.jpg
      :img-alt: RoboVerse 基准
      :link: embodied/roboverse
      :link-type: doc

      在 RoboVerse 操作环境中用 PPO 训练 π₀.₅。

.. toctree::
   :hidden:
   :maxdepth: 2

   BEHAVIOR-1K <embodied/behavior>
   CALVIN <embodied/calvin>
   DexJoCo <embodied/dexjoco>
   EmbodiChain <embodied/embodichain>
   Franka-Sim <embodied/frankasim>
   G1 动作跟踪 <embodied/g1_mimic>
   Genesis <embodied/genesis>
   GenieSim <embodied/geniesim>
   GSEnv <embodied/gsenv>
   IsaacLab <embodied/isaaclab>
   LIBERO <embodied/libero>
   ManiSkill <embodied/maniskill>
   MetaWorld <embodied/metaworld>
   MuJoCo Warp <embodied/mujoco_warp>
   PolaRiS <embodied/polaris>
   真机仿真器 <embodied/real_simulator>
   RoboCasa <embodied/robocasa>
   RoboCasa365 <embodied/robocasa365>
   RoboDojo（仅评测） <embodied/robodojo>
   RoboLab（仅评测） <embodied/robolab>
   RoboTwin <embodied/robotwin>
   RoboVerse <embodied/roboverse>
