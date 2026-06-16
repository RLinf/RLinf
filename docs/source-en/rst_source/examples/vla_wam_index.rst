RL on VLA / WAM Models
======================

This category groups examples in which the **vision-language-action (VLA)** or **world-action model (WAM)** is the headline. They show how to onboard a specific model family in RLinf — checkpoint loading, processor / config wiring, action head, and a reference RL fine-tuning recipe — independent of any single benchmark.

If you are starting from "I want to RL-fine-tune model *X*", this is the right entry point. For benchmark-driven examples see :doc:`simulators_index`.

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/pi0_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/pi0.html" style="text-decoration: underline; color: blue;">
           <b>RL on π₀ and π₀.₅ Models</b>
         </a><br>
         Significant improvement in RL training on π₀ and π₀.₅
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/gr00t.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/gr00t.html" style="text-decoration: underline; color: blue;">
           <b>RL on GR00T Models</b>
         </a><br>
         Support GR00T-N1.5, N1.6 and N1.7 RL fine-tuning.
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/lingbotvla.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/lingbotvla.html" style="text-decoration: underline; color: blue;">
           <b>RL with Lingbot-VLA Model</b>
         </a><br>
         Support Lingbot-VLA + RoboTwin + GRPO training
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/dexmal/dexbotic/main/resources/intro.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/dexbotic.html" style="text-decoration: underline; color: blue;">
           <b>RL on Dexbotic Model</b>
         </a><br>
         Dexbotic (π₀.₅-based) + LIBERO + PPO training
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/starvla.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/starvla.html" style="text-decoration: underline; color: blue;">
           <b>RL on StarVLA Models</b>
         </a><br>
         StarVLA + LIBERO + GRPO embodied RL training
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/ABot-M0.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/abot_m0.html" style="text-decoration: underline; color: blue;">
           <b>RL on ABot-M0 Model</b>
         </a><br>
         ABot-M0 native integration with LIBERO-plus PPO training
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/hpcaitech/Open-Sora-Demo/raw/main/readme/icon.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/opensora.html" style="text-decoration: underline; color: blue;">
           <b>RL with OpenSora World Model</b>
         </a><br>
         Support OpenSora World Model + OpenVLA-OFT + GRPO training
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/wan.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/wan.html" style="text-decoration: underline; color: blue;">
           <b>RL with Wan World Model</b>
         </a><br>
         Support Wan World Model + OpenVLA-OFT + GRPO training
       </p>
     </div>

   </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   π₀ and π₀.₅ Models <embodied/pi0>
   GR00T Models <embodied/gr00t>
   Lingbot-VLA Models <embodied/lingbotvla>
   Dexbotic Models <embodied/dexbotic>
   StarVLA Models <embodied/starvla>
   ABot-M0 <embodied/abot_m0>
   OpenSora World Model <embodied/opensora>
   Wan World Model <embodied/wan>
