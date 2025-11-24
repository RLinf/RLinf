Example Gallery
===============

This section presents the collection of **examples currently supported by RLinf**, 
showcasing how the framework can be applied across different scenarios and 
demonstrating its efficiency in practice.
This example gallery is continuously expanding, covering new scenarios and tasks to highlight RLinf's flexibility and efficiency.

Embodied Intelligence Scenarios
-------------------------------

This category includes embodied training examples with SOTA models (e.g., pi0, pi0.5, OpenVLA-OFT) and different benchmarks (e.g., LIBERO, ManiSkill, RoboTwin, MetaWorld),
as well as reinforcement learning training examples on real robots.

.. raw:: html

   <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start; flex-wrap: wrap;">
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <video controls autoplay loop muted playsinline preload="metadata" style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
         <source src="https://github.com/RLinf/misc/raw/main/pic/embody.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/maniskill.html" target="_blank" style="text-decoration: underline; color: blue;">
          <b>RL with ManiSkill Benchmark</b>
         </a><br>
         ManiSkill + OpenVLA + PPO/GRPO achieves SOTA performance
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/libero_numbers.jpeg" 
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/libero.html" target="_blank" style="text-decoration: underline; color: blue;">
          <b>RL with LIBERO Benchmark</b>
         </a><br>
         LIBERO + OpenVLA-OFT + GRPO reaches 99% success rate
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/pi0_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/pi0.html" target="_blank" style="text-decoration: underline; color: blue;">
          <b>RL on π₀ and π₀.₅ Models</b>
         </a><br>
         Significant improvement in RL training on π₀ and π₀.₅
       </p>
     </div>
   </div>

   <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start; flex-wrap: wrap;">
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/behavior.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/behavior.html" target="_blank" style="text-decoration: underline; color: blue;">
          <b>RL with Behavior Benchmark</b>
         </a><br>
         Support Behavior+OpenVLA-OFT+PPO/GRPO training
       </p>
     </div>
     
      <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/metaworld.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/metaworld.html" target="_blank" style="text-decoration: underline; color: blue;">
          <b>RL with MetaWorld Benchmark</b>
         </a><br>
         Support MetaWorld+π₀/π₀.₅+PPO/GRPO training
       </p>
     </div>

      <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/gr00t.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/gr00t.html" target="_blank" style="text-decoration: underline; color: blue;">
          <b>RL on GR00T-N1.5 Model</b>
         </a><br>
         Support GR00T-N1.5 RL fine-tuning.
       </p>
     </div>
   </div>

   <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start; flex-wrap: wrap;">
      <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/main/assets/files/50_tasks.gif" 
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" 
            data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <b>[Ongoing]RL with RoboTwin Benchmark</b><br>
         RoboTwin + OpenVLA-OFT + PPO achieves SOTA performance
       </p>
     </div>

      <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/calvin.png" 
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" 
            data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <b>[Ongoing]RL with CALVIN Benchmark</b><br>
         Support CALVIN+π₀/π₀.₅+PPO/GRPO training
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/franka_arm_small.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <b>[Ongoing]Real-World RL with Franka</b><br>
         RLinf worker seamlessly integrates with the Franka robotic arm
       </p>
     </div>
   </div>


Reasoning Scenarios
-------------------

Reinforcement learning is a key approach to improving reasoning capabilities. RLinf supports mainstream models such as Qwen and Qwen-next for RL training in tasks like Math, achieving SOTA results.

.. raw:: html

   <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start; flex-wrap: wrap;">
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/math_numbers_small.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/reasoning.html" target="_blank" style="text-decoration: underline; color: blue;">
          <b>RL Training for Math Reasoning</b>
         </a><br>
         Achieves SOTA results on AIME24/AIME25/GPQA-diamond benchmarks
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/waiting_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <b>[Ongoing]RL Training for MoE Models</b><br>
         RL training speed improved by xx% compared to other tools
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/waiting_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <b>[Ongoing]RL Training for Qwen-next</b><br>
         Achieves SOTA training performance with Qwen-next
       </p>
     </div>
   </div>


Agent Scenarios
---------------

RLinf's worker abstraction, flexible communication modules, and support for various accelerators make it naturally suited for building agent workflows and training agents.
The following examples include agent workflow construction, online RL training, and environment integration.

.. raw:: html

   <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start; flex-wrap: wrap;">
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/coding_online_rl_offline_numbers.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/coding_online_rl.html" target="_blank" style="text-decoration: underline; color: blue;">
          <b>Open-Source Online RL for Code Completion</b>
         </a><br>
         End-to-end online RL with RLinf + Continue, improving model performance by 52%
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/waiting_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <b>[Ongoing]rStar2-agent RL Training</b><br>
         Flexible resource allocation and scheduling across components
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/waiting_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <b>[Ongoing]SWE-agent</b><br>
         Unified deployment, inference, and training with high flexibility and performance
       </p>
     </div>
   </div>


Practical System Features
-------------------------

RLinf's overall design is simple and modular.
Workers abstract components for RL and agents, with a flexible and efficient communication library enabling inter-component interaction.
Thanks to this decoupled design, workers can be flexibly and dynamically scheduled to computing resources or assigned to the most suitable accelerators.

.. raw:: html

   <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start; flex-wrap: wrap;">
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/waiting_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <b>[Ongoing]Hot Scaling/Switching of Workers (Components)</b><br>
         Hot switching reduces training time by 50%+
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/waiting_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <b>[Ongoing]Hybrid Training on Heterogeneous Accelerator</b><br>
         Flexible inter-operability between components on different accelerators to build training workflows
       </p>
     </div>
   </div>


.. toctree::
   :hidden:
   :maxdepth: 2

   maniskill
   libero
   behavior
   metaworld
   pi0
   gr00t
   reasoning
   coding_online_rl
