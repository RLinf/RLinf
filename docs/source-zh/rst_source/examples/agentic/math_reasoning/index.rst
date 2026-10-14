数学推理
========

面向大语言模型的数学推理强化学习训练——基于 Qwen 与 DeepSeek-V3（Moonlight-16B）架构的 GRPO 与 PPO——使用 Megatron-Bridge actor 与 SGLang rollout。

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="reasoning.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/math_numbers_small.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="reasoning.html" style="text-decoration: underline; color: blue;">
          <b>使用 GRPO 训练 Math 推理任务</b>
         </a><br>
         使用强化学习提升数学推理能力，在 AIME24/AIME25/GPQA-diamond 上达到 SOTA
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="reasoning_ppo.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/ppo_vs_grpo.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="reasoning_ppo.html" style="text-decoration: underline; color: blue;">
          <b>使用 PPO 训练 Math 推理任务</b>
         </a><br>
           使用 PPO 算法进行数学推理强化学习训练，示例配置基于 Qwen2.5-1.5B
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="moonlight.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/moonlight/moonlight_grpo_reward.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="moonlight.html" style="text-decoration: underline; color: blue;">
          <b>Moonlight-16B GRPO</b>
         </a><br>
         使用 GRPO 训练 Moonlight-16B-A3B-Instruct（DeepSeek-V3 MLA + MoE）
       </p>
     </div>
   </div>

.. toctree::
   :hidden:
   :maxdepth: 1

   Qwen2.5 GRPO <reasoning>
   Qwen2.5 PPO <reasoning_ppo>
   Moonlight-16B GRPO <moonlight>
