智能体与生成
============

基于 RLinf 的分布式 worker 运行时、Megatron/FSDP 训练后端与 SGLang/vLLM 推理引擎，训练数学推理 LLM、多轮工具调用智能体、多智能体系统以及图像/视频扩散模型。

运行智能体或推理示例前，先通过 ``bash requirements/install.sh agentic`` 安装环境（见 :doc:`安装指南 <../../start/installation>`\ ）；图像与视频生成示例使用 ``bash requirements/install.sh embodied --model diffusion``。

如果只有单机或单卡，可以先从 :doc:`Qwen3-VL GRPO <qwen3_vl_geo3k>`\ （单机 FSDP2）、带单机快速启动覆盖参数的 :doc:`Qwen2.5 数学推理 GRPO <math_reasoning/reasoning>` 或 :doc:`SD3.5 Medium 图像生成 <../video_models>` 入手。

按场景选择
----------

.. list-table::
   :header-rows: 1
   :widths: 20 26 22 32

   * - 方向
     - 模型与后端
     - 算法 / 模式
     - 包含示例
   * - :doc:`数学推理 <math_reasoning/index>`
     - Qwen2.5、Moonlight-16B（Megatron + SGLang/vLLM）
     - GRPO、PPO
     - :doc:`Qwen2.5 GRPO <math_reasoning/reasoning>`、:doc:`Qwen2.5 PPO <math_reasoning/reasoning_ppo>`、:doc:`Moonlight-16B <math_reasoning/moonlight>`
   * - :doc:`Search-R1 <searchr1/index>`
     - Qwen2.5-3B、GLM-4.7-Flash（本地 Wiki / Qdrant）
     - 多轮检索 RL（同步与异步）
     - :doc:`Qwen2.5-3B <searchr1/qwen2.5>`、:doc:`GLM-4.7-Flash <searchr1/glm4.7_flash>`、:doc:`异步训练 <searchr1/async>`
   * - :doc:`WideSeek-R1 <wideseek_r1/index>`
     - Qwen3-4B 主智能体 + 并行子智能体
     - 多智能体强化学习（MARL）
     - :doc:`环境与工具 <wideseek_r1/tools>`、:doc:`训练 <wideseek_r1/train>`、:doc:`评测 <wideseek_r1/eval>`
   * - :doc:`rStar2 <rstar2>`
     - Qwen2.5-7B + Python 执行服务
     - 多轮反思推理 RL
     - :doc:`rStar2-Agent <rstar2>`
   * - :doc:`AgentLightning <agentlightning_calc_x>`
     - Agent Lightning 编排 + RLinf 训练后端
     - 工具调用 RL（Calc-X）
     - :doc:`AgentLightning Calc-X <agentlightning_calc_x>`
   * - :doc:`Coding Online RL <coding_online_rl>`
     - Qwen2.5-Coder-1.5B + Continue 插件
     - 在线 PPO / GRPO
     - :doc:`代码补全在线 RL <coding_online_rl>`
   * - :doc:`Qwen3-VL GRPO <qwen3_vl_geo3k>`
     - Qwen3-VL（FSDP + SGLang）
     - 多模态 GRPO（Geo3K）
     - :doc:`Qwen3-VL Geo3K <qwen3_vl_geo3k>`
   * - :doc:`图像与视频生成 <../video_models>`
     - SD3.5 Medium、Wan2.2 TI2V 5B（Diffusers）
     - Diffusion-NFT、DiffusionART、ReFL
     - :doc:`Diffusion-NFT <../diffusion_nft>`、:doc:`DiffusionART <../diffusion_art>`、:doc:`ReFL <../embodied/refl>`

推理与多模态强化学习
--------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: 数学推理
      :link: math_reasoning/index
      :link-type: doc

      基于 Qwen2.5（1.5B/7B）与 Moonlight-16B-A3B（DeepSeek-V3 MLA + MoE）的 GRPO/PPO 训练，覆盖 AIME24、AIME25 与 GPQA-diamond。

   .. grid-item-card:: Qwen3-VL GRPO
      :link: qwen3_vl_geo3k
      :link-type: doc

      基于 FSDP 与 SGLang 的视觉语言模型 GRPO 训练，面向 Geo3K 几何推理任务。

工具调用与多智能体强化学习
--------------------------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Search-R1
      :link: searchr1/index
      :link-type: doc

      基于 Qwen2.5-3B 与 GLM-4.7-Flash 的多轮搜索调用 RL（支持同步与异步训练），配套 FAISS 或 Qdrant 本地检索服务。

   .. grid-item-card:: WideSeek-R1
      :link: wideseek_r1/index
      :link-type: doc

      面向广泛信息检索的主从多智能体强化学习框架，4B 模型在 WideSearch 上达到 40.0% item F1。

   .. grid-item-card:: rStar2
      :link: rstar2
      :link-type: doc

      结合本地 Python 代码执行服务，训练模型完成多轮数学推理与自我反思。

   .. grid-item-card:: AgentLightning
      :link: agentlightning_calc_x
      :link-type: doc

      使用 Agent Lightning 编排智能体工作流，并将分布式 RL 训练交给 RLinf 后端（Calc-X 示例）。

   .. grid-item-card:: Coding Online RL
      :link: coding_online_rl
      :link-type: doc

      结合 RLinf 与 Continue IDE 插件的端到端代码补全在线强化学习流程。

图像与视频生成
--------------

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: 图像与视频生成总览
      :link: ../video_models
      :link-type: doc

      SD3.5 Medium 与 Wan2.2 TI2V 5B 的公共扩散环境安装、OCR 提示数据集与权重下载。

   .. grid-item-card:: Diffusion-NFT
      :link: ../diffusion_nft
      :link-type: doc

      使用正负速度候选与 OCR 奖励微调 SD3.5 Medium 和 Wan2.2。

   .. grid-item-card:: DiffusionART
      :link: ../diffusion_art
      :link-type: doc

      在 prompt 组内构造共享奖励引导，面向 SD3.5 Medium 图像 OCR 与 Wan2.2 视频 OCR 训练。

   .. grid-item-card:: ReFL
      :link: ../embodied/refl
      :link-type: doc

      使用可微 ImageReward 反馈与可选监督图文更新微调 SD3.5 Medium。
