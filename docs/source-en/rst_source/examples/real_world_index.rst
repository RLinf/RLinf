RL with Real-World Robots
=========================

Use this section when your starting point is physical robot hardware. Start with Franka if you use a Franka arm or a Franka-based rig; use the other robot pages for GimArm, XSquare Turtle2, and Dexmal DOS-W1.

Each section gives the setup path for teleoperation, data collection, sim-to-real transfer, deployment, or online RL.

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka.html" style="text-decoration: underline; color: blue;">
           <b>Franka</b>
         </a><br>
         Use the Franka section for base real-world RL, reward models, ZED + Robotiq, GELLO, VR / PICO, dual-arm rigs, dexterous hands, Pi0 SFT, and HG-DAgger
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/gim-arm.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/gim_arm.html" style="text-decoration: underline; color: blue;">
           <b>GimArm</b>
         </a><br>
         Train a 6-DOF GimArm peg-insertion task over SocketCAN with Pinocchio-based FK
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/xsquare_turtle2_arm_small.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/xsquare_turtle2.html" style="text-decoration: underline; color: blue;">
           <b>XSquare Turtle2</b>
         </a><br>
         Run SAC with a CNN policy on the XSquare Turtle2 dual-arm robot
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/dos-w1.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/dosw1.html" style="text-decoration: underline; color: blue;">
           <b>Dexmal DOS-W1</b>
         </a><br>
         Train a flow-matching + SAC pick task on the Dexmal DOS-W1 dual-arm robot
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
         Bilateral teleoperation data collection on the LeRobot SO101 6-DOF arm, plus the π₀ SFT recipe for the resulting dataset
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
