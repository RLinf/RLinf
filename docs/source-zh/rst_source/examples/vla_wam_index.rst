具身模型
========

本类示例以 **模型或策略类** 为主线，展示如何在 RLinf 中接入特定模型家族 —— 包括 checkpoint 加载、processor / config 接线、动作头实现、轻量级 MLP 策略，以及不依赖具体基准的一份强化学习微调参考配方。

如果你的出发点是 "我想训练或微调模型 *X*"，这里是合适的入口。若以基准为主线请参考 :doc:`simulators_index`\ 。

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/mlp.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/3_layer_mlp.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/mlp.html" style="text-decoration: underline; color: blue;">
           <b>MLP 策略强化学习</b>
         </a><br>
         使用 PPO、SAC 或 GRPO 在多种仿真环境中训练轻量级 MLP 策略
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/pi0.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/pi0.html" style="text-decoration: underline; color: blue;">
           <b>π₀和π₀.₅模型强化学习训练</b>
         </a><br>
         在π₀和π₀.₅上实现强化学习的效果跃升
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/gr00t.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/gr00t.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/gr00t.html" style="text-decoration: underline; color: blue;">
           <b>GR00T模型强化学习训练</b>
         </a><br>
         支持GR00T-N1.5，N1.6与N1.7强化学习微调
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/dexbotic.html" style="display: block;"><img src="https://raw.githubusercontent.com/dexmal/dexbotic/main/resources/intro.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/dexbotic.html" style="text-decoration: underline; color: blue;">
           <b>基于 Dexbotic 模型的强化学习训练</b>
         </a><br>
         Dexbotic（基于 π₀.₅）+ LIBERO + PPO 训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/starvla.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/starvla.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/starvla.html" style="text-decoration: underline; color: blue;">
           <b>StarVLA 模型强化学习训练</b>
         </a><br>
         StarVLA + LIBERO + GRPO 具身强化学习训练
       </p>
     </div>

     <!-- TODO: swap for a 3:2 pic/molmoact2.png in RLinf/misc once available. -->
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/molmoact2.html" style="display: block;"><img src="https://raw.githubusercontent.com/allenai/molmoact2/main/assets/MolmoAct2.svg"
            style="width: 100%; height: 200px; object-fit: contain; background: #ffffff; padding: 24px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/molmoact2.html" style="text-decoration: underline; color: blue;">
           <b>MolmoAct2 模型评测</b>
         </a><br>
         在 LIBERO 上评测官方 MolmoAct2-LIBERO checkpoint
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/lingbotvla.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/lingbotvla.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/lingbotvla.html" style="text-decoration: underline; color: blue;">
           <b>基于 Lingbot-VLA 模型的强化学习</b>
         </a><br>
         支持 Lingbot-VLA + RoboTwin + GRPO 训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/evo1.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/evo1.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/evo1.html" style="text-decoration: underline; color: blue;">
           <b>Evo-1 模型强化学习训练</b>
         </a><br>
         使用 Evo-1 视觉语言动作模型进行具身强化学习训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/abot_m0.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/ABot-M0.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/abot_m0.html" style="text-decoration: underline; color: blue;">
           <b>ABot-M0 模型强化学习训练</b>
         </a><br>
         ABot-M0 原生集成与 LIBERO-plus PPO 训练
       </p>
     </div>

   </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   MLP <embodied/mlp>
   π₀ / π₀.₅ <embodied/pi0>
   GR00T <embodied/gr00t>
   Dexbotic <embodied/dexbotic>
   StarVLA <embodied/starvla>
   MolmoAct2 <embodied/molmoact2>
   Lingbot-VLA <embodied/lingbotvla>
   Evo-1 <embodied/evo1>
   ABot-M0 <embodied/abot_m0>
