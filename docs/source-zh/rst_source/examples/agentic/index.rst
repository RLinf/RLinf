智能体场景
========================================

RLinf的worker抽象、灵活的通信组件、以及对不同类型加速器的支持使RLinf天然支持智能体工作流的构建，以及智能体的训练。以下示例包含数学推理强化学习与智能体 AI 工作流，例如智能体工作流构建、在线强化学习训练、环境接入，以及 **以推理为中心的智能体强化学习** 等场景。

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="wideseek_r1/index.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/wideseek_r1/scaling.png"
            style="width: 100%; height: 200px; object-fit: contain; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="wideseek_r1/index.html" style="text-decoration: underline; color: blue;">
          <b>WideSeek-R1</b>
         </a><br>
         通过多智能体强化学习探索用于广泛信息检索的宽度扩展方法
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="agentlightning_calc_x.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/agentlightning_calcx.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="agentlightning_calc_x.html" style="text-decoration: underline; color: blue;">
          <b>AgentLightning Calc-X</b>
         </a><br>
         基于 Agent Lightning 编排智能体强化学习，RLinf 作为分布式训练后端
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="coding_online_rl.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/coding_online_rl_offline_numbers.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="coding_online_rl.html" style="text-decoration: underline; color: blue;">
          <b>代码补全在线强化学习开源版</b>
         </a><br>
         基于RLinf+continue实现端到端在线强化学习，模型效果提升52%
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="searchr1.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/searchr1.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="searchr1.html" style="text-decoration: underline; color: blue;">
          <b>Search-R1强化学习</b>
         </a><br>
         训练LLM调用搜索工具回答问题，RLinf加速训练过程55%
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="rstar2.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/rstar2-RLinf-7b.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="rstar2.html" style="text-decoration: underline; color: blue;">
          <b>rStar2-agent强化学习</b>
         </a><br>
         通过强化学习让模型学会使用Python工具进行自主推理和反思,以极低计算成本达到数学推理领域的顶尖水平
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/waiting_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <b>[适配中]SWE-agent</b><br>
         部署、推理、训练一体，高灵活性、高性能
       </p>
     </div>

    <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
      <a href="math_reasoning/index.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/math_numbers_small.jpg"
           style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
      <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
       <a href="math_reasoning/index.html" style="text-decoration: underline; color: blue;">
         <b>数学推理</b>
        </a><br>
        使用 GRPO 与 PPO 训练 Qwen2.5 与 Moonlight-16B 的数学推理能力
      </p>
    </div>

    <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
      <a href="qwen3_vl_geo3k.html" style="display: block;"><img src="https://github.com/RLinf/misc/raw/main/pic/geo_problem.png"
           style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
      <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
       <a href="qwen3_vl_geo3k.html" style="text-decoration: underline; color: blue;">
         <b>使用 GRPO 训练 Qwen3-VL 视觉语言推理</b>
        </a><br>
        基于 GRPO 的视觉语言模型强化学习训练，用于几何问题求解（Geo3K）
      </p>
    </div>

   </div>

扩展智能体强化学习
------------------

使用这些指南扩展智能体与推理工作负载背后的训练系统。

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 指南
     - 内容
   * - :doc:`动态调度 <dynamic_scheduling>`
     - 在训练持续运行时重新分配 actor、rollout 与 inference 资源。
   * - :doc:`5D 并行 <5D>`
     - 组合 Megatron-LM 并行维度以训练大模型。

.. toctree::
   :hidden:
   :maxdepth: 2

   WideSeek-R1 <wideseek_r1/index>
   AgentLightning <agentlightning_calc_x>
   Coding Online RL <coding_online_rl>
   Search-R1 <searchr1>
   rStar2 <rstar2>
   数学推理 <math_reasoning/index>
   Qwen3 VL GRPO <qwen3_vl_geo3k>
   动态调度 <dynamic_scheduling>
   5D 并行 <5D>
