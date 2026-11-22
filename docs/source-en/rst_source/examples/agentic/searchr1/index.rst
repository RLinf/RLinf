Search-R1 RL Training
=====================

Reproduce the `Search-R1 paper <https://arxiv.org/abs/2503.09516>`__ in RLinf, training LLMs with multi-turn reinforcement learning to answer questions by calling a local search tool. Start the local Wiki-2018 retrieval server first, then pick a training recipe below.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Local Wiki Retriever
      :link: retriever
      :link-type: doc

      Start the shared Wiki-2018 retrieval server with prebuilt Qdrant (recommended) or FAISS GPU before running any recipe.

   .. grid-item-card:: Qwen2.5-3B Search-R1
      :link: qwen2.5
      :link-type: doc

      .. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/searchr1.png
         :alt: Qwen2.5-3B Search-R1

      Search-R1 on 8×H100, 55% faster than the original implementation.

   .. grid-item-card:: GLM-4.7-Flash Search-R1
      :link: glm4.7_flash
      :link-type: doc

      .. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_reward.png
         :alt: GLM-4.7-Flash Search-R1

      Search-R1 for an MoE model (~3B active parameters) on 8×A100 or 2×8 A100.

   .. grid-item-card:: Asynchronous Search-R1
      :link: async
      :link-type: doc

      .. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/glm_searchr1_async_step_time.png
         :alt: Asynchronous Search-R1

      Overlap rollout with training: step time drops from 278.6 s to 139.0 s on 2×8 A100.

.. toctree::
   :hidden:
   :maxdepth: 1

   Local Wiki Retriever <retriever>
   Qwen2.5-3B <qwen2.5>
   GLM-4.7-Flash <glm4.7_flash>
   Async Training <async>
