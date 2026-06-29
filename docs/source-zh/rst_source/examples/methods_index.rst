具身智能算法
========================================

本类示例以 **训练算法或方法** 为主线，独立于具体基准或模型，覆盖离线 RL、模仿学习、仿真-真机协同训练以及残差 / 噪声空间策略调控。

如果你在思考 *如何训练*（PPO 还是 SAC？IQL 还是 DAgger？RECAP？），而不是要在 *什么任务上* 训练或 *微调哪个模型*，请参考本节。

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/sac_flow.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/sac-flow-overview.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/sac_flow.html" style="text-decoration: underline; color: blue;">
           <b>SAC-Flow 策略训练</b>
         </a><br>
         使用 SAC 训练 Flow Matching 策略 (Sim &amp; Real)
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
       <a href="embodied/dagger.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/dagger.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/dagger.html" style="text-decoration: underline; color: blue;">
           <b>具身策略的 DAgger 训练</b>
         </a><br>
         通过专家重标注与回放缓冲区训练推进在线模仿学习
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
       <a href="embodied/steam.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/recap.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/steam.html" style="text-decoration: underline; color: blue;">
           <b>STEAM：集成优势建模</b>
         </a><br>
         用 worst-of-N 集成进度评论器进行离线策略优化
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
       <a href="embodied/iql_d4rl.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/d4rl.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/iql_d4rl.html" style="text-decoration: underline; color: blue;">
           <b>基于 D4RL 基准的离线强化学习</b>
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
   DAgger <embodied/dagger>
   RECAP <embodied/recap>
   STEAM <embodied/steam>
   Co-Training <embodied/co_training>
   IQL (D4RL) <embodied/iql_d4rl>
