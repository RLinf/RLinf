智能体场景
==========

RLinf的worker抽象、灵活的通信组件、以及对不同类型加速器的支持使RLinf天然支持智能体工作流的构建，以及智能体的训练。以下示例包含数学推理强化学习与智能体 AI 工作流，例如智能体工作流构建、在线强化学习训练、环境接入，以及**以推理为中心的智能体强化学习**等场景。

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
       <img src="https://github.com/RLinf/misc/raw/main/pic/searchr1.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/searchr1.html" target="_blank" style="text-decoration: underline; color: blue;">
          <b>Search-R1强化学习</b>
         </a><br>
         训练LLM调用搜索工具回答问题，RLinf加速训练过程55%
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

   <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
     <img src="https://github.com/RLinf/misc/raw/main/pic/math_numbers_small.jpg"
          style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
     <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
       <a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/agentic/reasoning.html" target="_blank" style="text-decoration: underline; color: blue;">
        <b>Math推理的强化学习训练</b>
       </a><br>
       使用强化学习提升数学推理能力，在 AIME24/AIME25/GPQA-diamond 上达到 SOTA
     </p>
   </div>
   </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   coding_online_rl
   searchr1
   reasoning