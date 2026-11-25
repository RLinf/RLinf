Simulators
==========

Pick this section when starting from a benchmark or simulator. Each page covers environment installation, asset paths, observation and action spaces, and a reference RL recipe.

For model-centric workflows (π₀ / π₀.₅, GR00T, …), see :doc:`vla_wam_index`. For real-robot setups, see :doc:`real_world_index`. For AMD ROCm, Huawei Ascend CANN, and Moore Threads MUSA, choose a supported combination in :ref:`the hardware support table <model-hardware-support>`.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: BEHAVIOR-1K
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/behavior.jpg
      :img-alt: BEHAVIOR-1K benchmark
      :link: embodied/behavior
      :link-type: doc

      Fine-tune OpenVLA-OFT, π₀, and π₀.₅ with PPO on 50 OmniGibson household tasks.

   .. grid-item-card:: CALVIN
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/calvin.png
      :img-alt: CALVIN benchmark
      :link: embodied/calvin
      :link-type: doc

      Train π₀ and π₀.₅ with PPO on long-horizon CALVIN scene-transfer suites.

   .. grid-item-card:: DexJoCo
      :img-top: https://raw.githubusercontent.com/brave-eai/dexjoco/main/docs/pics/water_plant_4.png
      :img-alt: DexJoCo benchmark
      :link: embodied/dexjoco
      :link-type: doc

      Train CNN and π₀.₅ policies with SFT and PPO on dexterous WaterPlant manipulation.

   .. grid-item-card:: EmbodiChain
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/embodichain.gif
      :img-alt: EmbodiChain simulator
      :link: embodied/embodichain
      :link-type: doc

      Train MLP policies with PPO on EmbodiChain gym environments.

   .. grid-item-card:: Franka-Sim
      :img-top: https://raw.githubusercontent.com/RLinf/serl/refs/heads/RLinf/franka-sim/franka_sim/franka_sim/envs/xmls/robotiq_2f85/2f85.png
      :img-alt: Franka-Sim benchmark
      :link: embodied/frankasim
      :link-type: doc

      Train MLP and CNN policies with PPO or SAC on MuJoCo Franka-Sim tasks.

   .. grid-item-card:: Genesis
      :img-top: https://raw.githubusercontent.com/YilingQiao/Genesis/readme-assets/videos/HeroShot_Final.png
      :img-alt: Genesis simulator
      :link: embodied/genesis
      :link-type: doc

      Train MLP policies on the Genesis physics simulation platform.

   .. grid-item-card:: GenieSim
      :link: embodied/geniesim
      :link-type: doc

      Train π₀.₅ with PPO or GRPO on dual-arm GenieSim manipulation benchmarks.

   .. grid-item-card:: Go2 ABC
      :link: embodied/go2_abc
      :link-type: doc

      Train a Unitree Go2 policy with PPO on flat ground, rough terrain, and stairs.

   .. grid-item-card:: GSEnv
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/gsenv.gif
      :img-alt: GSEnv Real2Sim2Real
      :link: embodied/gsenv
      :link-type: doc

      Train π₀.₅ with PPO in Gaussian-splatting scenes for Real2Sim2Real transfer.

   .. grid-item-card:: IsaacLab
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/IsaacLab.png
      :img-alt: IsaacLab benchmark
      :link: embodied/isaaclab
      :link-type: doc

      Train GR00T and OpenPI policies with PPO on NVIDIA IsaacLab environments.

   .. grid-item-card:: LIBERO
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/libero_numbers.jpeg
      :img-alt: LIBERO benchmark
      :link: embodied/libero
      :link-type: doc

      Run OpenVLA-OFT and other VLAs on standard LIBERO (99% success), LIBERO-Pro, LIBERO-Plus, and LIBERO-Mem.

   .. grid-item-card:: ManiSkill
      :img-top: https://raw.githubusercontent.com/mani-skill/ManiSkill/main/figures/teaser.jpg
      :img-alt: ManiSkill benchmark
      :link: embodied/maniskill
      :link-type: doc

      Train OpenVLA and OpenVLA-OFT with PPO or GRPO on ManiSkill3 tabletop manipulation and OOD benchmarks.

   .. grid-item-card:: MetaWorld
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/metaworld.png
      :img-alt: MetaWorld benchmark
      :link: embodied/metaworld
      :link-type: doc

      Fine-tune π₀, π₀.₅, and OpenVLA-OFT with PPO or GRPO across MT50 and ML45 tasks.

   .. grid-item-card:: MuJoCo Warp
      :img-top: https://raw.githubusercontent.com/google-deepmind/mujoco_warp/main/benchmarks/humanoid/rollout_three_humanoids.webp
      :img-alt: MuJoCo Warp locomotion
      :link: embodied/mujoco_warp
      :link-type: doc

      Train MLP policies with PPO on five GPU-accelerated locomotion tasks.

   .. grid-item-card:: PolaRiS
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/polaris.png
      :img-alt: PolaRiS benchmark
      :link: embodied/polaris
      :link-type: doc

      Train OpenPI policies with PPO on PolaRiS tabletop manipulation tasks.

   .. grid-item-card:: Real-World Simulator
      :link: embodied/real_simulator
      :link-type: doc

      Inject observation delay and network emulation into simulated training to test deployment robustness.

   .. grid-item-card:: RoboCasa
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/robocasa.jpeg
      :img-alt: RoboCasa benchmark
      :link: embodied/robocasa
      :link-type: doc

      Fine-tune π₀ with GRPO on RoboCasa kitchen manipulation tasks.

   .. grid-item-card:: RoboCasa365
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/robocasa.jpeg
      :img-alt: RoboCasa365 benchmark
      :link: embodied/robocasa365
      :link-type: doc

      Train π₀ with PPO on the RoboCasa365 benchmark suite.

   .. grid-item-card:: RoboDojo (Eval Only)
      :img-top: https://media.luminis-sim.com/media/home/teaser.png
      :img-alt: RoboDojo benchmark
      :link: embodied/robodojo
      :link-type: doc

      Evaluate π₀.₅ checkpoints on the RoboDojo manipulation benchmark.

   .. grid-item-card:: RoboLab (Eval Only)
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/robolab.png
      :img-alt: RoboLab manipulation benchmark
      :link: embodied/robolab
      :link-type: doc

      Evaluate π₀.₅ across 120 DROID manipulation tasks on RoboLab.

   .. grid-item-card:: RoboTwin
      :img-top: https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/main/assets/files/50_tasks.gif
      :img-alt: RoboTwin benchmark
      :link: embodied/robotwin
      :link-type: doc

      Train OpenVLA-OFT, π₀, and π₀.₅ with PPO or GRPO on bimanual RoboTwin manipulation tasks.

   .. grid-item-card:: RoboVerse
      :img-top: https://roboverseorg.github.io/static/images/teaser.jpg
      :img-alt: RoboVerse benchmark
      :link: embodied/roboverse
      :link-type: doc

      Train π₀.₅ with PPO on RoboVerse manipulation environments.

.. toctree::
   :hidden:
   :maxdepth: 2

   BEHAVIOR-1K <embodied/behavior>
   CALVIN <embodied/calvin>
   DexJoCo <embodied/dexjoco>
   EmbodiChain <embodied/embodichain>
   Franka-Sim <embodied/frankasim>
   Genesis <embodied/genesis>
   GenieSim <embodied/geniesim>
   Go2 ABC <embodied/go2_abc>
   GSEnv <embodied/gsenv>
   IsaacLab <embodied/isaaclab>
   LIBERO <embodied/libero>
   ManiSkill <embodied/maniskill>
   MetaWorld <embodied/metaworld>
   MuJoCo Warp <embodied/mujoco_warp>
   PolaRiS <embodied/polaris>
   Real-World Simulator <embodied/real_simulator>
   RoboCasa <embodied/robocasa>
   RoboCasa365 <embodied/robocasa365>
   RoboDojo (Eval Only) <embodied/robodojo>
   RoboLab (Eval Only) <embodied/robolab>
   RoboTwin <embodied/robotwin>
   RoboVerse <embodied/roboverse>
