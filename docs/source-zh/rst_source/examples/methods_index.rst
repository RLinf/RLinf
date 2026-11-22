RL 算法示例
===========

本类示例以 **训练算法或方法** 为主线，独立于具体基准或模型，覆盖离策略与离线 RL、在线策略蒸馏、仿真-真机协同训练以及噪声空间 policy 调控。

按训练方法选型（SAC-Flow、DSRL、RLT、OPD、RECAP、STEAM、IQL 或仿真-真机协同训练）时，从本节进入。

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/sac_flow.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/sac-flow-overview.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/sac_flow.html" style="text-decoration: underline; color: blue;">
           <b>SAC-Flow：流匹配策略 SAC 强化学习</b>
         </a><br>
         使用 SAC 训练 Flow Matching 策略（Sim &amp; Real）
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/dsrl.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/dsrl.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/dsrl.html" style="text-decoration: underline; color: blue;">
           <b>DSRL：Pi0 噪声空间强化学习</b>
         </a><br>
         用轻量级 SAC 智能体在噪声空间引导冻结的 Pi0 扩散策略
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/rlt.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/RLT.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/rlt.html" style="text-decoration: underline; color: blue;">
           <b>RL Token</b>
         </a><br>
         借助 VLA prefix 特征启动在线 RL，并训练轻量级 actor-critic
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/co_training.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/rlinf-co/overview.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/co_training.html" style="text-decoration: underline; color: blue;">
           <b>仿真-真机协同训练</b>
         </a><br>
         仿真 PPO + 真机 SFT，提升 Sim-to-Real 迁移
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/opd.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/opd.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/opd.html" style="text-decoration: underline; color: blue;">
           <b>OPD：OpenVLA-OFT 在线策略蒸馏</b>
         </a><br>
         在 student on-policy 动作 token 上用 teacher logprob 蒸馏
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/recap.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/recap.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/recap.html" style="text-decoration: underline; color: blue;">
           <b>RECAP：离线优势条件策略优化</b>
         </a><br>
         基于优势引导的离线策略优化
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/steam.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/steam.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/steam.html" style="text-decoration: underline; color: blue;">
           <b>STEAM：集成优势建模</b>
         </a><br>
         用集成优势预测器做保守样本评估
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/iql_d4rl.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/d4rl.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/iql_d4rl.html" style="text-decoration: underline;">
           <b>IQL：基于 D4RL 的离线强化学习</b>
         </a><br>
         支持 D4RL 场景的 IQL 离线训练
       </p>
     </div>

   </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   SAC-Flow <embodied/sac_flow>
   DSRL <embodied/dsrl>
   RL Token <embodied/rlt>
   Co-Training <embodied/co_training>
   OPD <embodied/opd>
   RECAP <embodied/recap>
   STEAM <embodied/steam>
   IQL (D4RL) <embodied/iql_d4rl>
