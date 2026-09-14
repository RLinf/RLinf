具身模型
========

按模型或 policy 类型查找安装、checkpoint 和强化学习训练流程。若从基准出发选择示例，请参阅 :doc:`simulators_index`。

.. _model-hardware-support:

硬件支持
------------

模型示例默认使用 NVIDIA。AMD ROCm、华为昇腾 CANN 和摩尔线程 MUSA 也支持下列三个模型系列在 LIBERO 与 ManiSkill 上运行。选择模型链接可查看安装与启动步骤；在非 CUDA 后端上，ManiSkill 使用 CPU 运行 PhysX，并独立选择 renderer。

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - 模型
     - 硬件
     - 环境与范围
   * - :doc:`OpenVLA-OFT <embodied/openvla_oft>`
     - :ref:`AMD ROCm · 华为昇腾 CANN · 摩尔线程 MUSA <openvla-oft-hardware>`
     - LIBERO · ManiSkill
   * - :doc:`GR00T N1.5 <embodied/gr00t>`
     - :ref:`AMD ROCm · 华为昇腾 CANN · 摩尔线程 MUSA <gr00t-hardware>`
     - LIBERO · ManiSkill（需要带有 ``maniskill_widowx`` head 的 checkpoint）
   * - :doc:`π₀ / π₀.₅ (OpenPI) <embodied/pi0>`
     - :ref:`AMD ROCm · 华为昇腾 CANN · 摩尔线程 MUSA <pi0-hardware>`
     - LIBERO · ManiSkill

模型示例
--------

选择模型页面，查看完整流程及其支持的环境。

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/openvla_oft.html" style="display: block;"><img src="https://openvla-oft.github.io/static/images/libero_task_performance_results.png"
            style="width: 100%; height: 200px; object-fit: contain; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/openvla_oft.html" style="text-decoration: underline; color: blue;">
           <b>OpenVLA-OFT</b>
         </a><br>
         在 NVIDIA CUDA、AMD ROCm、华为昇腾 CANN 和摩尔线程 MUSA 上运行 LIBERO 训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/mlp.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/3_layer_mlp.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/mlp.html" style="text-decoration: underline; color: blue;">
           <b>MLP 策略强化学习</b>
         </a><br>
         使用 PPO、SAC 或 GRPO 在多种仿真环境中训练轻量级 MLP 策略
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/pi0.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/pi0.html" style="text-decoration: underline; color: blue;">
           <b>π₀和π₀.₅模型强化学习训练</b>
         </a><br>
         在π₀和π₀.₅上实现强化学习的效果跃升
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/gr00t.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/gr00t.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/gr00t.html" style="text-decoration: underline; color: blue;">
           <b>GR00T模型强化学习训练</b>
         </a><br>
         支持GR00T-N1.5，N1.6与N1.7强化学习微调
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/dexbotic.html" style="display: block;"><img src="https://raw.githubusercontent.com/dexmal/dexbotic/main/resources/intro.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/dexbotic.html" style="text-decoration: underline; color: blue;">
           <b>基于 Dexbotic 模型的强化学习训练</b>
         </a><br>
         Dexbotic（基于 π₀.₅）+ LIBERO + PPO 训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/starvla.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/starvla.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/starvla.html" style="text-decoration: underline; color: blue;">
           <b>StarVLA 模型强化学习训练</b>
         </a><br>
         StarVLA + LIBERO + GRPO 具身强化学习训练
       </p>
     </div>

     <!-- TODO: swap for a 3:2 pic/molmoact2.png in RLinf/misc once available. -->
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/molmoact2.html" style="display: block;"><img src="https://raw.githubusercontent.com/allenai/molmoact2/main/assets/MolmoAct2.svg"
            style="width: 100%; height: 200px; object-fit: contain; background: #ffffff; padding: 24px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/molmoact2.html" style="text-decoration: underline; color: blue;">
           <b>MolmoAct2 模型评测</b>
         </a><br>
         在 LIBERO 上评测官方 MolmoAct2-LIBERO checkpoint
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/lingbotvla.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/lingbotvla.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/lingbotvla.html" style="text-decoration: underline; color: blue;">
           <b>基于 Lingbot-VLA 模型的强化学习</b>
         </a><br>
         支持 Lingbot-VLA + RoboTwin + GRPO 训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/evo1.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/evo1.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/evo1.html" style="text-decoration: underline; color: blue;">
           <b>Evo-1 模型强化学习训练</b>
         </a><br>
         使用 Evo-1 视觉语言动作模型进行具身强化学习训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/abot_m0.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/ABot-M0.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/abot_m0.html" style="text-decoration: underline; color: blue;">
           <b>ABot-M0 模型强化学习训练</b>
         </a><br>
         ABot-M0 原生集成与 LIBERO-plus PPO 训练
       </p>
     </div>

   </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   OpenVLA-OFT <embodied/openvla_oft>
   MLP <embodied/mlp>
   π₀ / π₀.₅ <embodied/pi0>
   GR00T <embodied/gr00t>
   Dexbotic <embodied/dexbotic>
   StarVLA <embodied/starvla>
   MolmoAct2 <embodied/molmoact2>
   Lingbot-VLA <embodied/lingbotvla>
   Evo-1 <embodied/evo1>
   ABot-M0 <embodied/abot_m0>
