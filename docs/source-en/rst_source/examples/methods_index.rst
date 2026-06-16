Algorithms for Embodiment
=========================

This category groups examples in which the **training algorithm or recipe** is the headline — independent of any single benchmark or model. They cover offline RL, imitation learning, hybrid sim-real co-training, residual / noise-space policy steering, and lightweight policy classes.

Use this section when you are choosing *how* to train (PPO vs SAC vs IQL vs DAgger vs RECAP …) rather than *what* to train on or *what model* to fine-tune.

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/3_layer_mlp.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/mlp.html" style="text-decoration: underline; color: blue;">
           <b>MLP Policy Training</b>
         </a><br>
         Train an MLP policy with PPO/SAC/GRPO across simulation environments
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/sac-flow-overview.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/sac_flow.html" style="text-decoration: underline; color: blue;">
           <b>SAC-Flow Policy Training</b>
         </a><br>
         Train a Flow Matching policy with SAC (Sim &amp; Real)
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/dsrl.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/dsrl.html" style="text-decoration: underline; color: blue;">
           <b>DSRL for Pi0</b>
         </a><br>
         Steer a frozen Pi0 diffusion policy with lightweight SAC in noise space
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/dagger.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/dagger.html" style="text-decoration: underline; color: blue;">
           <b>DAgger for Embodied Policies</b>
         </a><br>
         Guide online imitation learning with expert relabeling and replay-buffer updates
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/recap.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/recap.html" style="text-decoration: underline; color: blue;">
           <b>RECAP: Offline Advantage-Based Policy Optimization</b>
         </a><br>
         Offline policy optimization via advantage-guided classifier-free guidance
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/rlinf-co/overview.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/co_training.html" style="text-decoration: underline; color: blue;">
           <b>Sim-Real Co-Training</b>
         </a><br>
         PPO in sim + SFT on real data for better sim-to-real transfer
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/d4rl.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/iql_d4rl.html" style="text-decoration: underline; color: blue;">
           <b>Offline RL with D4RL Benchmark</b>
         </a><br>
         Support IQL offline training for D4RL scenarios
       </p>
     </div>

   </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   embodied/mlp
   embodied/sac_flow
   embodied/dsrl
   embodied/dagger
   embodied/recap
   embodied/co_training
   embodied/iql_d4rl
