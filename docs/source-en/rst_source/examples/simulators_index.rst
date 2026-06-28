RL with Embodied Simulators
===========================

This category groups examples in which the **simulator (or benchmark)** is the headline. They show how to bring up RLinf on a specific simulation platform — environment installation, asset paths, observation/action spaces, and a reference RL recipe (typically PPO or GRPO with a VLA policy).

If you are starting from "I want to train on benchmark *X*", this is the right entry point. For model-centric examples (pi₀, GR00T, …) see :doc:`vla_wam_index`. For real-robot setups, including Franka, see :doc:`real_world_index`. For LIBERO setup on AMD ROCm or Ascend CANN accelerators, see the :doc:`Supported Accelerators <../guides/index>` tutorial.

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <video controls autoplay loop muted playsinline preload="metadata" style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
         <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/embody.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/maniskill.html" style="text-decoration: underline; color: blue;">
           <b>RL with ManiSkill Benchmark</b>
         </a><br>
         ManiSkill + OpenVLA + PPO/GRPO achieves SOTA performance
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/libero_numbers.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/libero.html" style="text-decoration: underline; color: blue;">
           <b>RL with LIBERO Benchmarks</b>
         </a><br>
         OpenVLA-OFT + PPO/GRPO on LIBERO (99% success) and on the harder LIBERO-Pro / LIBERO-Plus suites
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/behavior.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/behavior.html" style="text-decoration: underline; color: blue;">
           <b>RL with Behavior Benchmark</b>
         </a><br>
         Support BEHAVIOR + OpenVLA-OFT / π₀ / π₀.₅ + PPO training
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/metaworld.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/metaworld.html" style="text-decoration: underline; color: blue;">
           <b>RL with MetaWorld Benchmark</b>
         </a><br>
         Support MetaWorld+π₀/π₀.₅+PPO/GRPO training
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/IsaacLab.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/isaaclab.html" style="text-decoration: underline; color: blue;">
           <b>RL with IsaacLab Benchmark</b>
         </a><br>
         Support IsaacLab+gr00t+PPO training
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/calvin.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/calvin.html" style="text-decoration: underline; color: blue;">
           <b>RL with CALVIN Benchmark</b>
         </a><br>
         Support CALVIN+π₀/π₀.₅+PPO/GRPO training
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/robocasa.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/robocasa.html" style="text-decoration: underline; color: blue;">
           <b>RL with RoboCasa Benchmark</b>
         </a><br>
         Support RoboCasa+π₀+GRPO training
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/main/assets/files/50_tasks.gif"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/robotwin.html" style="text-decoration: underline; color: blue;">
           <b>RL with RoboTwin Benchmark</b>
         </a><br>
         Supports RoboTwin + OpenVLA-OFT / π₀ / π₀.₅ + PPO / GRPO training
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://roboverseorg.github.io/static/images/teaser.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/roboverse.html" style="text-decoration: underline; color: blue;">
           <b>RL with RoboVerse Benchmark</b>
         </a><br>
         Support RoboVerse + π₀.₅ + PPO training
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/serl/refs/heads/RLinf/franka-sim/franka_sim/franka_sim/envs/xmls/robotiq_2f85/2f85.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/frankasim.html" style="text-decoration: underline; color: blue;">
           <b>RL with Franka-Sim Benchmark</b>
         </a><br>
         Supports Franka-Sim + MLP/CNN + PPO/SAC training
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/embodichain.gif"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/embodichain.html" style="text-decoration: underline; color: blue;">
           <b>RL with EmbodiChain</b>
         </a><br>
         MLP + PPO on EmbodiChain gym tasks
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/polaris.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/polaris.html" style="text-decoration: underline; color: blue;">
           <b>RL with PolaRiS Benchmark</b>
         </a><br>
         PolaRiS + OpenPI + PPO training
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/gsenv.gif"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/gsenv.html" style="text-decoration: underline; color: blue;">
           <b>RL with GSEnv for Real2Sim2Real</b>
         </a><br>
         Support GSEnv + π₀.₅ + PPO training
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/YilingQiao/Genesis/readme-assets/videos/HeroShot_Final.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/genesis.html" style="text-decoration: underline; color: blue;">
           <b>RL with Genesis Benchmark</b>
         </a><br>
         MLP policy training on the Genesis simulation platform
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
