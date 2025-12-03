示例库
========

本节展示了 **RLinf 目前支持的示例集合**，  
展示该框架如何应用于不同场景，并演示其在实际中的高效性。示例库会随着时间不断扩展，涵盖新的场景和任务，以展示 RLinf 的多样性和可扩展性。

具身智能场景
----------------

具身智能场景包含SOTA模型（如pi0、pi0.5、OpenVLA-OFT）和不同模拟器（如LIBERO、ManiSkill、RoboTwin、MetaWorld）的训练示例，以及真机强化学习训练示例等。

.. raw:: html

   <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start; flex-wrap: wrap;">
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <video controls autoplay loop muted playsinline preload="metadata" style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
         <source src="https://github.com/RLinf/misc/raw/main/pic/embody.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/maniskill.html" target="_blank" style="text-decoration: underline; color: blue;">
          <b>基于ManiSkill的强化学习</b>
         </a><br>
         ManiSkill+OpenVLA+PPO/GRPO达到SOTA训练效果
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/libero_numbers.jpeg" 
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/libero.html" target="_blank" style="text-decoration: underline; color: blue;">
          <b>基于LIBERO的强化学习</b>
         </a><br>
         LIBERO+OpenVLA-OFT+GRPO成功率达99%
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/pi0_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/pi0.html" target="_blank" style="text-decoration: underline; color: blue;">
          <b>π₀和π₀.₅模型强化学习训练</b>
         </a><br>
         在π₀和π₀.₅上实现强化学习的效果跃升
       </p>
     </div>
   </div>

   <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start; flex-wrap: wrap;">
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/behavior.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/behavior.html" target="_blank" style="text-decoration: underline; color: blue;">
          <b>基于Behavior的强化学习</b>
         </a><br>
         支持Behavior+OpenVLA-OFT+PPO/GRPO训练
       </p>
     </div>
     
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/metaworld.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/metaworld.html" target="_blank" style="text-decoration: underline; color: blue;">
          <b>基于MetaWorld的强化学习</b>
         </a><br>
         支持MetaWorld+π₀/π₀.₅+PPO/GRPO训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/IsaacLab.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/metaworld.html" target="_blank" style="text-decoration: underline; color: blue;">
          <b>基于IsaacLab的强化学习</b>
         </a><br>
         支持IsaacLab+gr00t+PPO训练
       </p>
     </div>

      <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/gr00t.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/gr00t.html" target="_blank" style="text-decoration: underline; color: blue;">
          <b>GR00T-N1.5模型强化学习训练</b>
         </a><br>
         支持GR00T-N1.5强化学习微调
       </p>
     </div>
   </div>

   <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start; flex-wrap: wrap;">
      <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
        <img src="https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/main/assets/files/50_tasks.gif" 
              style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" 
              data-target="animated-image.originalImage">
        <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
          <b>[开发中]基于RoboTwin的强化学习</b><br>
          RoboTwin+OpenVLA-OFT+PPO达到SOTA训练效果
        </p>
      </div>

      <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/calvin.png" 
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" 
            data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <b>[开发中]基于CALVIN的强化学习</b><br>
         支持CALVIN+π₀/π₀.₅+PPO/GRPO训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/franka_arm_small.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <b>[开发中]Franka真机强化学习</b><br>
         RLinf worker无缝对接Franka机械臂
       </p>
     </div>
   </div>



推理场景
--------------

强化学习是提升模型推理能力的关键手段，RLinf支持主流模型如Qwen、Qwen-next在Math等场景的强化学习训练，并达到SOTA的训练效果。

.. raw:: html

   <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start; flex-wrap: wrap;">
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/math_numbers_small.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/reasoning.html" target="_blank" style="text-decoration: underline; color: blue;">
          <b>Math推理的强化学习训练</b>
         </a><br>
         AIME24/AIME25/GPQA-diamond评测结果达到SOTA
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/waiting_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <b>[开发中]MoE模型强化学习训练</b><br>
         MoE RL训练速度相比同类工具提升xx%
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/waiting_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <b>[开发中]Qwen-next强化学习训练</b><br>
         Qwen-next强化学习训练效果达到SOTA
       </p>
     </div>
   </div>


智能体场景
--------------

RLinf的worker抽象、灵活的通信组件、以及对不同类型加速器的支持使RLinf天然支持智能体工作流的构建，以及智能体的训练。以下示例包含智能体工作流构建、在线强化学习训练、环境接入等示例。

.. raw:: html

   <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start; flex-wrap: wrap;">
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/coding_online_rl_offline_numbers.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/coding_online_rl.html" target="_blank" style="text-decoration: underline; color: blue;">
          <b>代码补全在线强化学习开源版</b>
         </a><br>
         基于RLinf+continue实现端到端在线强化学习，模型效果提升52%
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/waiting_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <b>[适配中]rStar2-agent强化学习</b><br>
         支持各组件所用资源量的灵活配置与调度
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/waiting_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <b>[适配中]SWE-agent</b><br>
         部署、推理、训练一体，高灵活性、高性能
       </p>
     </div>
   </div>


实用系统功能
--------------------

RLinf的整体设计简洁且模块化，以Worker为抽象封装强化学习训练、智能体中的组件，提供灵活高效的通信库做组件间通信。基于这种解耦的设计，可以灵活调度Worker所使用的计算资源，也可以将Worker分配到更适配的加速器上。

.. raw:: html

   <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start; flex-wrap: wrap;">
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/waiting_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <b>[开发中]Worker(组件)间秒级热切换</b><br>
         秒级热切换提升训练速度50%+
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/waiting_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <b>[开发中]异构加速器混合训练</b><br>
         使用不同加速器运行的组件间灵活互通，构建训练工作流
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
   isaaclab
   pi0
   gr00t
   reasoning
   coding_online_rl
