真实机器人强化学习
========================================

按机器人硬件选择配置与使用指南。Franka 机械臂及其组合设备请从 Franka 页面开始；GimArm、XSquare Turtle2、Dexmal DOS-W1、AgileX Piper 和 SO101 请进入对应页面。

根据硬件检查、遥操作、数据采集、Sim-to-Real 迁移、部署或在线 RL 的需求，选择相应指南。

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/franka_index.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka_index.html" style="text-decoration: underline; color: blue;">
           <b>Single-Arm Franka</b>
         </a><br>
         进入 Single-Arm Franka 章节，查看基础真机 RL、reward model、ZED + Robotiq、GELLO、VR / PICO、双臂、灵巧手、Pi0 SFT 和 HG-DAgger
       </p>
     </div>
      <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/dual_franka_index.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/dual-franka.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/dual_franka_index.html" style="text-decoration: underline; color: blue;">
           <b>Dual-Arm Franka</b>
         </a><br>
         进入 Dual-Arm Franka 章节，查看基础真机 RL、reward model、ZED + Robotiq、GELLO、VR / PICO、双臂、灵巧手、Pi0 SFT 和 HG-DAgger
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/gim_arm.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/gim-arm.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/gim_arm.html" style="text-decoration: underline; color: blue;">
           <b>GimArm</b>
         </a><br>
         在 GimArm 六自由度机械臂上通过 SocketCAN 与 Pinocchio FK 训练 peg-insertion 任务
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/xsquare_turtle2.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/xsquare_turtle2_arm_small.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/xsquare_turtle2.html" style="text-decoration: underline; color: blue;">
           <b>XSquare Turtle2</b>
         </a><br>
         在 XSquare Turtle2 双臂机器人上运行 SAC + CNN 策略
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/dosw1.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/dos-w1.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/dosw1.html" style="text-decoration: underline; color: blue;">
           <b>Dexmal DOS-W1</b>
         </a><br>
         在 Dexmal DOS-W1 双臂机器人上训练 Flow Matching + SAC 抓取任务
       </p>
     </div>

   </div>

Piper 与 SO101 配置
--------------------------------

通过以下指南连接机械臂并运行硬件测试脚本。Piper 和 SO101 目前尚未提供真机任务或训练流程。

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: AgileX Piper
      :link: embodied/piper
      :link-type: doc

      配置 CAN，运行 Piper 关节与夹爪测试脚本。

   .. grid-item-card:: SO101
      :link: embodied/so101
      :link-type: doc

      配置电机、标定 SO-101，并运行关节与夹爪测试脚本。

.. toctree::
   :hidden:
   :maxdepth: 3

   Single-Arm Franka <embodied/franka_index>
   Dual-Arm Franka <embodied/dual_franka_index>
   GimArm <embodied/gim_arm>
   XSquare Turtle2 <embodied/xsquare_turtle2>
   DOS-W1 <embodied/dosw1>
   Piper <embodied/piper>
   SO101 <embodied/so101>
