Search-R1 强化学习训练
======================

在 RLinf 中复现 `Search-R1 论文 <https://arxiv.org/abs/2503.09516>`__ 的多轮搜索强化学习实验，训练大语言模型调用本地搜索工具回答问题。先启动本地 Wiki-2018 检索服务，再选择下方的训练配方。

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: 本地 Wiki 检索服务
      :link: retriever
      :link-type: doc

      运行任意配方前，先用预构建的 Qdrant（推荐）或 FAISS GPU 启动共享的 Wiki-2018 检索服务。

   .. grid-item-card:: Qwen2.5-3B Search-R1
      :link: qwen2.5
      :link-type: doc

      .. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/searchr1.png
         :alt: Qwen2.5-3B Search-R1

      在 8×H100 上训练 Search-R1，相比原版实现加速 55%。

   .. grid-item-card:: GLM-4.7-Flash Search-R1
      :link: glm4.7_flash
      :link-type: doc

      .. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_reward.png
         :alt: GLM-4.7-Flash Search-R1

      在 8×A100 或 2×8 A100 上训练 MoE 模型（激活参数约 3B）。

   .. grid-item-card:: Search-R1 异步训练
      :link: async
      :link-type: doc

      .. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_step_time.png
         :alt: Search-R1 异步训练

      rollout 与训练重叠：2×8 A100 上步时从 278.6 s 降到 139.0 s。

.. toctree::
   :hidden:
   :maxdepth: 1

   本地 Wiki 检索服务 <retriever>
   Qwen2.5-3B <qwen2.5>
   GLM-4.7-Flash <glm4.7_flash>
   异步训练 <async>
