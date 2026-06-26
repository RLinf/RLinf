真实机器人强化学习
========================================

当你的出发点是真实机器人硬件时，请使用本节。如果你使用 Franka 机械臂或 Franka-based rig，请从 Franka 开始；GimArm、XSquare Turtle2 和 Dexmal DOS-W1 请进入对应机器人页面。

每个章节都给出遥操作、数据采集、Sim-to-Real 迁移、部署或在线 RL 所需的配置路径。

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka.html" style="text-decoration: underline; color: blue;">
           <b>Franka</b>
         </a><br>
         进入 Franka 章节，查看基础真机 RL、reward model、ZED + Robotiq、GELLO、VR / PICO、双臂、灵巧手、Pi0 SFT 和 HG-DAgger
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/gim-arm.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/gim_arm.html" style="text-decoration: underline; color: blue;">
           <b>GimArm</b>
         </a><br>
         在 GimArm 六自由度机械臂上通过 SocketCAN 与 Pinocchio FK 训练 peg-insertion 任务
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/xsquare_turtle2_arm_small.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/xsquare_turtle2.html" style="text-decoration: underline; color: blue;">
           <b>XSquare Turtle2</b>
         </a><br>
         在 XSquare Turtle2 双臂机器人上运行 SAC + CNN 策略
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/dos-w1.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/dosw1.html" style="text-decoration: underline; color: blue;">
           <b>Dexmal DOS-W1</b>
         </a><br>
         在 Dexmal DOS-W1 双臂机器人上训练 Flow Matching + SAC 抓取任务
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <!-- TODO(thumbnail): replace placeholder cover image URL for so101 -->
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/so101.html" style="text-decoration: underline; color: blue;">
           <b>SO101</b>
         </a><br>
         在 LeRobot SO101 六自由度机械臂上进行双侧遥操作数据采集，并提供基于所得数据集的 π₀ SFT 配方
       </p>
     </div>

   </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   Franka <embodied/franka>
   GimArm <embodied/gim_arm>
   XSquare Turtle2 <embodied/xsquare_turtle2>
   DOS-W1 <embodied/dosw1>
   SO101 <embodied/so101>
