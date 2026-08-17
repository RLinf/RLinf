Embodied Models
===============

This category groups examples in which the **model or policy class** is the headline. They show how to onboard a specific model family in RLinf — checkpoint loading, processor / config wiring, action head, lightweight MLP policies, and a reference RL fine-tuning recipe — independent of any single benchmark.

If you are starting from "I want to train or RL-fine-tune model *X*", this is the right entry point. For benchmark-driven examples see :doc:`simulators_index`.

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/mlp.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/3_layer_mlp.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/mlp.html" style="text-decoration: underline; color: blue;">
           <b>RL on MLP Policy</b>
         </a><br>
         Train a lightweight MLP policy with PPO, SAC, or GRPO across simulation environments
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/pi0.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/pi0.html" style="text-decoration: underline; color: blue;">
           <b>RL on π₀ and π₀.₅ Models</b>
         </a><br>
         Significant improvement in RL training on π₀ and π₀.₅
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/gr00t.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/gr00t.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/gr00t.html" style="text-decoration: underline; color: blue;">
           <b>RL on GR00T Models</b>
         </a><br>
         Support GR00T-N1.5, N1.6 and N1.7 RL fine-tuning.
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/dexbotic.html" style="display: block;"><img src="https://raw.githubusercontent.com/dexmal/dexbotic/main/resources/intro.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/dexbotic.html" style="text-decoration: underline; color: blue;">
           <b>RL on Dexbotic Model</b>
         </a><br>
         Dexbotic (π₀.₅-based) + LIBERO + PPO training
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/starvla.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/starvla.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/starvla.html" style="text-decoration: underline; color: blue;">
           <b>RL on StarVLA Models</b>
         </a><br>
         StarVLA + LIBERO + GRPO embodied RL training
       </p>
     </div>

     <!-- TODO: swap for a 3:2 pic/molmoact2.png in RLinf/misc once available. -->
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/molmoact2.html" style="display: block;"><img src="https://raw.githubusercontent.com/allenai/molmoact2/main/assets/MolmoAct2.svg"
            style="width: 100%; height: 200px; object-fit: contain; background: #ffffff; padding: 24px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/molmoact2.html" style="text-decoration: underline; color: blue;">
           <b>MolmoAct2 Evaluation</b>
         </a><br>
         Evaluate the official MolmoAct2-LIBERO checkpoint on LIBERO
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/lingbotvla.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/lingbotvla.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/lingbotvla.html" style="text-decoration: underline; color: blue;">
           <b>RL with Lingbot-VLA Model</b>
         </a><br>
         Support Lingbot-VLA + RoboTwin + GRPO training
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/evo1.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/evo1.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/evo1.html" style="text-decoration: underline; color: blue;">
           <b>RL on Evo-1 Models</b>
         </a><br>
         Train the Evo-1 vision-language-action model with embodied RL
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/abot_m0.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/ABot-M0.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/abot_m0.html" style="text-decoration: underline; color: blue;">
           <b>RL on ABot-M0 Model</b>
         </a><br>
         ABot-M0 native integration with LIBERO-plus PPO training
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
