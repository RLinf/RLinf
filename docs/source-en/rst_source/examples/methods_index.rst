RL Recipes
==========

Pick this section when choosing a training method rather than a benchmark or model family. Recipes cover off-policy and offline RL, online distillation, sim-real co-training, and noise-space policy steering.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Sim-Real Co-Training
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/rlinf-co/overview.png
      :img-alt: Sim-Real Co-Training overview
      :link: embodied/co_training
      :link-type: doc

      Combine PPO in simulation with SFT on real-world data for sim-to-real transfer.

   .. grid-item-card:: DSRL
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/dsrl.png
      :img-alt: DSRL for Pi0
      :link: embodied/dsrl
      :link-type: doc

      Steer a frozen π₀ diffusion policy with lightweight SAC in noise space.

   .. grid-item-card:: FlexRLT
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/flexrlt.png
      :img-alt: FlexRLT frozen VLA and trainable actor-critic
      :link: embodied/flexrlt
      :link-type: doc

      Train a compact actor-critic online from frozen VLA prefix features with STEAM routing.

   .. grid-item-card:: IQL (D4RL)
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/d4rl.png
      :img-alt: IQL on D4RL
      :link: embodied/iql_d4rl
      :link-type: doc

      Train offline Implicit Q-Learning policies on D4RL benchmarks.

   .. grid-item-card:: OPD
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/opd.png
      :img-alt: Online policy distillation
      :link: embodied/opd
      :link-type: doc

      Distill OpenVLA-OFT on LIBERO using teacher log-probabilities over on-policy student action tokens.

   .. grid-item-card:: RECAP
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/recap.png
      :img-alt: RECAP overview
      :link: embodied/recap
      :link-type: doc

      Run offline policy optimization with advantage-guided classifier-free guidance.

   .. grid-item-card:: RL Token
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/RLT.png
      :img-alt: RL Token
      :link: embodied/rlt
      :link-type: doc

      Bootstrap online RL with frozen VLA prefix features and a compact actor-critic.

   .. grid-item-card:: SAC-Flow
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/sac-flow-overview.png
      :img-alt: SAC-Flow overview
      :link: embodied/sac_flow
      :link-type: doc

      Train a Flow Matching policy with SAC in ManiSkill simulation or on a real Franka arm.

   .. grid-item-card:: STEAM
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/steam.png
      :img-alt: STEAM ensemble advantage modeling
      :link: embodied/steam
      :link-type: doc

      Evaluate offline samples conservatively with an ensemble advantage predictor.

   .. grid-item-card:: SuperMimic
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/supermimic.jpg
      :img-alt: SuperMimic overview
      :link: embodied/supermimic
      :link-type: doc

      Train a G1 whole-body tracking policy through a differentiable world model in MuJoCo Warp.

.. toctree::
   :hidden:
   :maxdepth: 2

   Co-Training <embodied/co_training>
   DSRL <embodied/dsrl>
   FlexRLT <embodied/flexrlt>
   IQL (D4RL) <embodied/iql_d4rl>
   OPD <embodied/opd>
   RECAP <embodied/recap>
   RL Token <embodied/rlt>
   SAC-Flow <embodied/sac_flow>
   STEAM <embodied/steam>
   SuperMimic <embodied/supermimic>
