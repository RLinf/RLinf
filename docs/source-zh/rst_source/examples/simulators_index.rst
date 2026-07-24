基于模拟器的具身强化学习
========================================

本类示例以 **模拟器（基准）** 为主线，展示如何在某个仿真平台上运行 RLinf —— 包括环境安装、资产路径、观测/动作空间，以及一个参考 RL 训练配方（通常为 PPO 或 GRPO + VLA 策略）。

如果你的出发点是 "我想在基准 *X* 上训练"，那这里就是合适的入口。若以模型为主线（pi₀、GR00T 等）请参考 :doc:`vla_wam_index`\ ；包括 Franka 在内的真机部署请参考 :doc:`real_world_index`\ 。如需在 AMD ROCm 或 Ascend CANN 加速器上运行 LIBERO，请参阅 :doc:`支持的加速器 <../guides/index>` 教程章节。

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/maniskill.html" style="display: block;"><video controls autoplay loop muted playsinline preload="metadata" style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
         <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/embody.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/maniskill.html" style="text-decoration: underline; color: blue;">
           <b>基于ManiSkill的强化学习</b>
         </a><br>
         ManiSkill+OpenVLA+PPO/GRPO达到SOTA训练效果
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/libero.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/libero_numbers.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/libero.html" style="text-decoration: underline; color: blue;">
           <b>基于 LIBERO 的强化学习</b>
         </a><br>
         OpenVLA-OFT + PPO/GRPO 在 LIBERO 上成功率 99%，并支持更具挑战的 LIBERO-Pro / LIBERO-Plus 套件
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/behavior.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/behavior.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/behavior.html" style="text-decoration: underline; color: blue;">
           <b>基于Behavior的强化学习</b>
         </a><br>
         支持 BEHAVIOR + OpenVLA-OFT / π₀ / π₀.₅ + PPO 训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/metaworld.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/metaworld.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/metaworld.html" style="text-decoration: underline; color: blue;">
           <b>基于MetaWorld的强化学习</b>
         </a><br>
         支持MetaWorld+π₀/π₀.₅+PPO/GRPO训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/isaaclab.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/IsaacLab.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/isaaclab.html" style="text-decoration: underline; color: blue;">
           <b>基于IsaacLab的强化学习</b>
         </a><br>
         支持IsaacLab+gr00t+PPO训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/calvin.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/calvin.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage"></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/calvin.html" style="text-decoration: underline; color: blue;">
           <b>基于CALVIN的强化学习</b>
         </a><br>
         支持CALVIN+π₀/π₀.₅+PPO/GRPO训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/robocasa.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/robocasa.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/robocasa.html" style="text-decoration: underline; color: blue;">
           <b>基于RoboCasa的强化学习</b>
         </a><br>
         支持RoboCasa+π₀+GRPO训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/robocasa365.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/robocasa.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/robocasa365.html" style="text-decoration: underline; color: blue;">
           <b>基于RoboCasa365的强化学习</b>
         </a><br>
         支持RoboCasa365 + π₀ + PPO训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/robotwin.html" style="display: block;"><img src="https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/main/assets/files/50_tasks.gif"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage"></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/robotwin.html" style="text-decoration: underline; color: blue;">
           <b>基于RoboTwin的强化学习</b>
         </a><br>
         支持RoboTwin + OpenVLA-OFT/π₀/π₀.₅ + PPO/GRPO训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/roboverse.html" style="display: block;"><img src="https://roboverseorg.github.io/static/images/teaser.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage"></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/roboverse.html" style="text-decoration: underline; color: blue;">
           <b>基于RoboVerse的强化学习</b>
         </a><br>
         支持RoboVerse + π₀.₅ + PPO训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/frankasim.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/serl/refs/heads/RLinf/franka-sim/franka_sim/franka_sim/envs/xmls/robotiq_2f85/2f85.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage"></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/frankasim.html" style="text-decoration: underline; color: blue;">
           <b>基于Franka-Sim的强化学习</b>
         </a><br>
         支持Franka-Sim+MLP/CNN+PPO/SAC训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/embodichain.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/embodichain.gif"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/embodichain.html" style="text-decoration: underline; color: blue;">
           <b>基于 EmbodiChain 的强化学习</b>
         </a><br>
         使用 EmbodiChain gym 任务进行 MLP + PPO 训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/polaris.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/polaris.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/polaris.html" style="text-decoration: underline; color: blue;">
           <b>基于 PolaRiS 仿真平台的强化学习</b>
         </a><br>
         PolaRiS + OpenPI + PPO 训练桌面操作任务
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/gsenv.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/gsenv.gif"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/gsenv.html" style="text-decoration: underline; color: blue;">
           <b>基于 GSEnv 的 Real2Sim2Real 强化学习</b>
         </a><br>
         支持 GSEnv + π₀.₅ + PPO 训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/genesis.html" style="display: block;"><img src="https://raw.githubusercontent.com/YilingQiao/Genesis/readme-assets/videos/HeroShot_Final.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/genesis.html" style="text-decoration: underline; color: blue;">
           <b>基于 Genesis 的强化学习</b>
         </a><br>
         在 Genesis 仿真平台上进行 MLP 策略训练
       </p>
     </div>

   </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   ManiSkill <embodied/maniskill>
   LIBERO <embodied/libero>
   Behavior <embodied/behavior>
   MetaWorld <embodied/metaworld>
   IsaacLab <embodied/isaaclab>
   CALVIN <embodied/calvin>
   RoboCasa <embodied/robocasa>
   RoboCasa365 <embodied/robocasa365>
   RoboTwin <embodied/robotwin>
   RoboVerse <embodied/roboverse>
   Franka-Sim <embodied/frankasim>
   EmbodiChain <embodied/embodichain>
   PolaRiS <embodied/polaris>
   GSEnv <embodied/gsenv>
   Genesis <embodied/genesis>
   真机仿真器 <embodied/real_simulator>
