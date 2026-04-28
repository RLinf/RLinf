具身智能场景
============

具身智能场景包含SOTA模型（如pi0、pi0.5、OpenVLA-OFT）和不同模拟器（如LIBERO、ManiSkill、RoboTwin、MetaWorld）的训练示例，以及真机强化学习训练示例等。

.. raw:: html

  <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <video controls autoplay loop muted playsinline preload="metadata" style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
         <source src="https://github.com/RLinf/misc/raw/main/pic/embody.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="maniskill.html" style="text-decoration: underline; color: blue;">
           <b>基于ManiSkill的强化学习</b>
         </a><br>
         ManiSkill+OpenVLA+PPO/GRPO达到SOTA训练效果
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/libero_numbers.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="libero.html" style="text-decoration: underline; color: blue;">
           <b>基于LIBERO的强化学习</b>
         </a><br>
         LIBERO+OpenVLA-OFT+GRPO成功率达99%
       </p>
     </div>

   <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/378920588652fff0a2a0b163b392c94694993345/pic/libero-plus.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="liberoplus_pro.html" style="text-decoration: underline; color: blue;">
           <b>基于 LIBERO-Pro 与 LIBERO-Plus 的强化学习</b>
         </a><br>
         支持 LIBERO-Pro / LIBERO-Plus + OpenVLA-OFT / π₀ / π₀.₅ + PPO/GRPO 训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/behavior.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="behavior.html" style="text-decoration: underline; color: blue;">
           <b>基于Behavior的强化学习</b>
         </a><br>
         支持Behavior+OpenVLA-OFT+PPO/GRPO训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/metaworld.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="metaworld.html" style="text-decoration: underline; color: blue;">
           <b>基于MetaWorld的强化学习</b>
         </a><br>
         支持MetaWorld+π₀/π₀.₅+PPO/GRPO训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/IsaacLab.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="isaaclab.html" style="text-decoration: underline; color: blue;">
           <b>基于IsaacLab的强化学习</b>
         </a><br>
         支持IsaacLab+gr00t+PPO训练
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/calvin.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="calvin.html" style="text-decoration: underline; color: blue;">
           <b>基于CALVIN的强化学习</b>
         </a><br>
         支持CALVIN+π₀/π₀.₅+PPO/GRPO训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/robocasa.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="robocasa.html" style="text-decoration: underline; color: blue;">
           <b>基于RoboCasa的强化学习</b>
         </a><br>
         支持RoboCasa+π₀+GRPO训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/main/assets/files/50_tasks.gif"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="robotwin.html" style="text-decoration: underline; color: blue;">
           <b>基于RoboTwin的强化学习</b>
         </a><br>
         支持RoboTwin + OpenVLA-OFT/π₀/π₀.₅ + PPO/GRPO训练
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/serl/refs/heads/RLinf/franka-sim/franka_sim/franka_sim/envs/xmls/robotiq_2f85/2f85.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="frankasim.html" style="text-decoration: underline; color: blue;">
           <b>基于Franka-Sim的强化学习</b>
         </a><br>
         支持Franka-Sim+MLP/CNN+PPO/SAC训练
       </p>
     </div>

    <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
      <img src="https://github.com/RLinf/misc/raw/main/pic/embodichain.gif"
           style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
      <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="embodichain.html" style="text-decoration: underline; color: blue;">
          <b>基于 EmbodiChain 的强化学习</b>
        </a><br>
        使用 EmbodiChain gym 任务进行 MLP + PPO 训练
      </p>
    </div>
    <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/hpcaitech/Open-Sora-Demo/raw/main/readme/icon.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="opensora.html" style="text-decoration: underline; color: blue;">
           <b>基于 OpenSora 世界模型的强化学习</b>
         </a><br>
         支持 OpenSora 世界模型 + OpenVLA-OFT + GRPO 训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/wan.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="wan.html" style="text-decoration: underline; color: blue;">
           <b>基于 Wan 世界模型的强化学习</b>
         </a><br>
         支持 Wan 世界模型 + OpenVLA-OFT + GRPO 训练
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/gsenv.gif"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="gsenv.html" style="text-decoration: underline; color: blue;">
           <b>基于 GSEnv 的 Real2Sim2Real 强化学习</b>
         </a><br>
         支持 GSEnv + π₀.₅ + PPO 训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
      <img src="https://github.com/RLinf/misc/raw/main/pic/d4rl.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="iql_d4rl.html" style="text-decoration: underline; color: blue;">
          <b>基于 D4RL 基准的离线强化学习</b>
         </a><br>
         支持 D4RL 场景的 IQL 离线训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/pi0_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="pi0.html" style="text-decoration: underline; color: blue;">
           <b>π₀和π₀.₅模型强化学习训练</b>
         </a><br>
         在π₀和π₀.₅上实现强化学习的效果跃升
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/gr00t.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="gr00t.html" style="text-decoration: underline; color: blue;">
           <b>GR00T-N1.5模型强化学习训练</b>
         </a><br>
         支持GR00T-N1.5强化学习微调
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/lingbotvla.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="lingbotvla.html" style="text-decoration: underline; color: blue;">
           <b>基于 Lingbot-VLA 模型的强化学习</b>
         </a><br>
         支持 Lingbot-VLA + RoboTwin + GRPO 训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/dexmal/dexbotic/main/resources/intro.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="dexbotic.html" style="text-decoration: underline; color: blue;">
           <b>基于 Dexbotic 模型的强化学习训练</b>
         </a><br>
         Dexbotic（基于 π₀.₅）+ LIBERO + PPO 训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/starvla.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="starvla.html" style="text-decoration: underline; color: blue;">
          <b>StarVLA 模型强化学习训练</b>
        </a><br>
        StarVLA + LIBERO + GRPO 具身强化学习训练
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RoboVerseOrg/RoboVerse/main/docs/source/metasim/images/tea.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="roboverse.html" style="text-decoration: underline; color: blue;">
           <b>基于RoboVerse的强化学习</b>
         </a><br>
         支持RoboVerse + π₀.₅ + PPO训练
       </p>
     </div>

   </div>

   <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start; flex-wrap: wrap;">
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/3_layer_mlp.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="mlp.html" style="text-decoration: underline; color: blue;">
           <b>基于MLP的强化学习</b>
         </a><br>
         使用 PPO/SAC/GRPO 训练 PPO 策略
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/sac-flow-overview.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="sac_flow.html" style="text-decoration: underline; color: blue;">
           <b>SAC-Flow 策略训练</b>
         </a><br>
         使用 SAC 训练 Flow Matching 策略 (Sim & Real)
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <!-- TODO(thumbnail): replace placeholder cover image URL for sft_openpi -->
       <img src="https://github.com/RLinf/misc/raw/main/pic/pi0_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="sft_openpi.html" style="text-decoration: underline; color: blue;">
           <b>监督微调训练</b>
         </a><br>
         支持 OpenPI 全量 SFT 与 LoRA 微调，作为强化学习前置阶段
       </p>
     </div>
   
   </div>

   <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start; flex-wrap: wrap;">
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/release_0.2/qwen2_5_sft_vlm.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="sft_vlm.html" style="text-decoration: underline; color: blue;">
           <b>VLM模型监督微调训练</b>
         </a><br>
         支持 Qwen 系列等 VLM 的全量监督微调与结果评估
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/dsrl.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="dsrl.html" style="text-decoration: underline; color: blue;">
           <b>DSRL：Pi0 噪声空间强化学习</b>
         </a><br>
         用轻量级 SAC 智能体在噪声空间引导冻结的 Pi0 扩散策略
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/dagger.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="dagger.html" style="text-decoration: underline; color: blue;">
           <b>具身策略的 DAgger 训练</b>
         </a><br>
         通过专家重标注与回放缓冲区训练推进在线模仿学习
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/recap.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="recap.html" style="text-decoration: underline; color: blue;">
           <b>RECAP：离线优势条件策略优化</b>
         </a><br>
         基于优势引导的离线策略优化
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/co_training.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="co_training.html" style="text-decoration: underline; color: blue;">
           <b>仿真-真机协同训练</b>
         </a><br>
         仿真 PPO + 真机 SFT，提升 Sim-to-Real 迁移
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/franka_arm_small.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="franka.html" style="text-decoration: underline; color: blue;">
           <b>Franka真机强化学习</b>
         </a><br>
         RLinf worker无缝对接Franka机械臂
       </p>
     </div>

      <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/franka_reward_model.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="franka_reward_model.html" style="text-decoration: underline; color: blue;">
           <b>Franka真机强化学习（基于 Reward Model ）</b>
         </a><br>
         使用 reward model 辅助完成机器人操作任务
       </p>
      </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/robotiq_zed.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="franka_zed_robotiq.html" style="text-decoration: underline; color: blue;">
           <b>Franka 真机使用 ZED 相机与 Robotiq 夹爪</b>
         </a><br>
         Franka 真机中 ZED 相机、Robotiq 夹爪安装与数据采集配置
       </p>
     </div>
    <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
      <img src="https://github.com/RLinf/misc/raw/main/pic/gello.jpeg"
           style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
      <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="franka_gello.html" style="text-decoration: underline; color: blue;">
          <b>Franka 真机使用 GELLO 遥操作设备</b>
        </a><br>
        Franka 真机中 GELLO 遥操作设备安装、配置与验证流程
      </p>
    </div>

    <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
      <img src="https://github.com/RLinf/misc/raw/main/pic/hg-dagger.jpg"
           style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
      <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="hg-dagger.html" style="text-decoration: underline; color: blue;">
          <b>Franka 机械臂上的 HG-DAgger</b>
        </a><br>
        Human-Gated 真机 DAgger 流程：数据采集、SFT 与在线干预训练
      </p>
    </div>

    <!-- TODO: 待 GimArm peg-insertion 的图片/视频上传到 RLinf/misc 后替换下方 src
         （PR #1016 中 zanghz21 的评审意见）。 -->
    <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
      <img src="https://github.com/RLinf/misc/raw/main/pic/gim_arm.jpg"
           style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
      <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="gim_arm.html" style="text-decoration: underline; color: blue;">
          <b>GimArm 真机强化学习</b>
        </a><br>
        GimArm 六自由度机械臂 + peg-insertion 任务，通过 SocketCAN 通信，并基于 Pinocchio 做正运动学
      </p>
    </div>

    <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
      <img src="https://github.com/RLinf/misc/raw/main/pic/pi0_icon.jpg"
           style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
      <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="franka_pi0_sft_deploy.html" style="text-decoration: underline; color: blue;">
          <b>Franka真机Pi0监督微调与部署全流程</b>
        </a><br>
        数据采集 + Pi0 SFT + 真机部署的完整端到端演示
      </p>
    </div>

    <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
      <img src="https://github.com/RLinf/misc/raw/main/pic/xsquare_turtle2_arm_small.jpg"
           style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
      <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="xsquare_turtle2.html" style="text-decoration: underline; color: blue;">
          <b>XSquare Turtle2 真机强化学习</b>
        </a><br>
        SAC + CNN 策略在 XSquare Turtle2 双臂机器人上的真机训练
      </p>
    </div>

    <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
      <!-- TODO(thumbnail): replace placeholder cover image URL for dosw1 -->
      <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/dos-w1.png"
           style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
      <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="dosw1.html" style="text-decoration: underline; color: blue;">
          <b>Dexmal DOS-W1 真机强化学习</b>
        </a><br>
        基于 Flow Matching 策略 + SAC 的 Dexmal DOS-W1 双臂抓取任务
      </p>
    </div>

    <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
      <!-- TODO(thumbnail): replace placeholder cover image URL for dosw1 -->
      <img src="https://github.com/RLinf/misc/raw/main/pic/gim-arm.png"
           style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
      <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="gim_arm.html" style="text-decoration: underline; color: blue;">
          <b>GimArm</b>
        </a><br>
        集成 GimArm 机械臂的数据采集
      </p>
    </div>
   </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   maniskill
   libero
   liberoplus_pro
   behavior
   metaworld
   isaaclab
   calvin
   robocasa
   robotwin
   roboverse
   frankasim
   embodichain
   opensora
   wan
   gsenv
   iql_d4rl
   pi0
   gr00t
   lingbotvla
   dexbotic
   starvla
   mlp
   sac_flow
   sft_openpi
   sft_dreamzero
   sft_vlm
   dsrl
   dagger
   recap
   co_training
   franka
   franka_reward_model
   franka_zed_robotiq
   franka_gello
   franka_pi0_sft_deploy
   hg-dagger
   gim_arm
   xsquare_turtle2
   dosw1
