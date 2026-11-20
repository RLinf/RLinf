Embodied Models
===============

Choose a model or policy class to find its installation, checkpoint, and RL training workflow. For examples organized by benchmark, see :doc:`simulators_index`.

.. _model-hardware-support:

Hardware Support
----------------

Model recipes default to NVIDIA CUDA. AMD ROCm, Huawei Ascend CANN, Moore Threads MUSA, Intel XPU, Kunlunxin XPU, MetaX MACA, and Biren SUPA also support the model and environment combinations below. Follow a model or hardware link for the training recipe and available backend setup instructions. On non-CUDA backends, ManiSkill runs PhysX on CPU and selects the renderer independently.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Model
     - Hardware
     - Environment and Scope
   * - :doc:`OpenVLA <embodied/maniskill>`
     - AMD ROCm · Huawei Ascend CANN · Moore Threads MUSA · MetaX MACA
     - ManiSkill
   * - :doc:`OpenVLA-OFT <embodied/openvla_oft>`
     - :ref:`AMD ROCm · Huawei Ascend CANN · Moore Threads MUSA <openvla-oft-hardware>` · Kunlunxin XPU
     - LIBERO · ManiSkill (AMD, Ascend, MUSA) · :ref:`RoboTwin on ROCm <robotwin-hardware>` · :ref:`Wan world model <wan-hardware>` (Ascend, MUSA, ROCm, Kunlunxin)
   * - :doc:`GR00T N1.5 / N1.7 <embodied/gr00t>`
     - :ref:`AMD ROCm · Huawei Ascend CANN · Moore Threads MUSA <gr00t-hardware>` · Intel XPU
     - N1.5 on LIBERO · ManiSkill (all four) · N1.7 on LIBERO · ManiSkill (Intel XPU); ManiSkill requires a ``maniskill_widowx`` checkpoint
   * - :doc:`π₀ / π₀.₅ (OpenPI) <embodied/pi0>`
     - :ref:`AMD ROCm · Huawei Ascend CANN · Moore Threads MUSA <pi0-hardware>` · Intel XPU · Kunlunxin XPU · MetaX MACA · Biren SUPA
     - LIBERO (AMD, Ascend, MUSA, Intel, Kunlunxin, MetaX) · ManiSkill (AMD, Ascend, MUSA, Intel) · :ref:`RoboTwin on ROCm <robotwin-hardware>` · :doc:`SFT <embodied/sft_openpi>` (Biren SUPA)
   * - :doc:`LingbotVLA 1.0 <embodied/lingbotvla>`
     - :ref:`AMD ROCm <robotwin-hardware>`
     - :ref:`RoboTwin <robotwin-hardware>`
   * - :doc:`StarVLA (QwenOFT) <embodied/starvla>`
     - :ref:`Huawei Ascend CANN <starvla-hardware>`
     - LIBERO
   * - :doc:`DreamZero <embodied/sft_dreamzero>`
     - :ref:`Huawei Ascend CANN <sft-dreamzero-hardware>`
     - SFT (LeRobot datasets)

Model Recipes
-------------

Open a model page for the full workflow and its supported environments.

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/openvla_oft.html" style="display: block;"><img src="https://openvla-oft.github.io/static/images/libero_task_performance_results.png"
            style="width: 100%; height: 200px; object-fit: contain; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/openvla_oft.html" style="text-decoration: underline; color: blue;">
           <b>OpenVLA-OFT</b>
         </a><br>
         LIBERO training on NVIDIA CUDA, AMD ROCm, Huawei Ascend CANN, and Moore Threads MUSA
       </p>
     </div>

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
       <a href="embodied/pi0_fast.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/pi0_fast.html" style="text-decoration: underline; color: blue;">
           <b>RL on π₀-FAST</b>
         </a><br>
         LeRobot π₀-FAST + LIBERO-Long + token-level GRPO
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
       <a href="embodied/lingbotvla_index.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/lingbotvla.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/lingbotvla_index.html" style="text-decoration: underline; color: blue;">
           <b>RL with LingbotVLA Models</b>
         </a><br>
         Support LingbotVLA 1.0 (GRPO) and 2.0 (PPO) + RoboTwin
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

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/psi0.html" style="display: block;"><img src="https://raw.githubusercontent.com/physical-superintelligence-lab/Psi0/a32e57a3fabb8590c80677f9cd3d1fc3db60eb06/assets/media/arch.png" alt="Psi0 model architecture"
            style="width: 100%; height: 200px; object-fit: contain; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/psi0.html" style="text-decoration: underline; color: blue;">
           <b>RL on Psi0 Model</b>
         </a><br>
         Psi0 + SIMPLE + PPO humanoid RL training
       </p>
     </div>

   </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   OpenVLA-OFT <embodied/openvla_oft>
   MLP <embodied/mlp>
   π₀ / π₀.₅ <embodied/pi0>
   π₀-FAST <embodied/pi0_fast>
   GR00T <embodied/gr00t>
   Dexbotic <embodied/dexbotic>
   StarVLA <embodied/starvla>
   MolmoAct2 <embodied/molmoact2>
   LingbotVLA <embodied/lingbotvla_index>
   Evo-1 <embodied/evo1>
   ABot-M0 <embodied/abot_m0>
   Psi0 <embodied/psi0>
