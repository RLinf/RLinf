Agent 与 LLM
============

基于 RLinf 的分布式 worker 运行时、Megatron / FSDP 训练后端与 SGLang / vLLM 推理引擎，覆盖 **推理训练**\ （文本与多模态）、**Agent 训练**\ （多轮工具调用、多 Agent 与在线代码补全）以及 **图像与视频生成**。

运行推理或 Agent 示例前，先按 :doc:`安装 <../../start/installation>` 执行 ``bash requirements/install.sh agentic``；图像与视频生成示例使用 ``bash requirements/install.sh embodied --model diffusion``。

按方向选择
----------

.. list-table::
   :header-rows: 1
   :widths: 20 26 22 32

   * - 方向
     - 模型与后端
     - 算法 / 模式
     - 包含示例
   * - :doc:`推理训练 <math_reasoning/index>`
     - Moonlight-16B、Qwen2.5、Qwen3-VL（Megatron / FSDP + SGLang / vLLM）
     - GRPO、PPO（文本与视觉推理）
     - :doc:`Moonlight-16B <math_reasoning/moonlight>`、:doc:`Qwen2.5 GRPO <math_reasoning/reasoning>`、:doc:`Qwen2.5 PPO <math_reasoning/reasoning_ppo>`、:doc:`Qwen3-VL Geo3K <qwen3_vl_geo3k>`
   * - :doc:`Agent 训练 <agents/index>`
     - Agent Lightning、GLM-4.7-Flash、Qwen2.5、Qwen3-4B
     - 多轮工具调用、MARL、在线 RL
     - :doc:`AgentLightning <agentlightning_calc_x>`、:doc:`Coding Online RL <coding_online_rl>`、:doc:`rStar2 <rstar2>`、:doc:`Search-R1 <searchr1/index>`、:doc:`WideSeek-R1 <wideseek_r1/index>`
   * - :doc:`图像与视频生成 <../video_models>`
     - SD3.5 Medium、Wan2.2 TI2V 5B（Diffusers）
     - DiffusionART、Diffusion-NFT、ReFL
     - :doc:`DiffusionART <../diffusion_art>`、:doc:`Diffusion-NFT <../diffusion_nft>`、:doc:`ReFL <../embodied/refl>`

推理训练
--------

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: Moonlight-16B（MoE GRPO）
      :link: math_reasoning/moonlight
      :link-type: doc

      通过 Megatron-Bridge 训练 DeepSeek-V3 MLA + MoE 架构模型。

   .. grid-item-card:: Qwen2.5 数学推理（GRPO）
      :link: math_reasoning/reasoning
      :link-type: doc

      在 AReaL-boba 上用 GRPO 训练 DeepSeek-R1-Distill-Qwen（1.5B / 7B），覆盖 AIME24、AIME25 与 GPQA-diamond。

   .. grid-item-card:: Qwen2.5 数学推理（PPO）
      :link: math_reasoning/reasoning_ppo
      :link-type: doc

      基于 Qwen2.5-1.5B 的 PPO 数学推理训练，同时更新 actor 与 critic。

   .. grid-item-card:: Qwen3-VL 视觉推理（Geo3K）
      :link: qwen3_vl_geo3k
      :link-type: doc

      基于 FSDP 与 SGLang 的视觉语言模型 GRPO 训练，面向 Geo3K 几何推理任务。

Agent 训练
----------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: AgentLightning
      :link: agentlightning_calc_x
      :link-type: doc

      使用 Agent Lightning 编排 Agent 工作流，并将分布式 RL 训练交给 RLinf 后端（Calc-X 示例）。

   .. grid-item-card:: Coding Online RL
      :link: coding_online_rl
      :link-type: doc

      结合 RLinf 与 Continue IDE 插件的端到端代码补全在线强化学习流程。

   .. grid-item-card:: rStar2
      :link: rstar2
      :link-type: doc

      结合本地 Python 代码执行服务，训练模型完成多轮数学推理与自我反思。

   .. grid-item-card:: Search-R1
      :link: searchr1/index
      :link-type: doc

      基于 Qwen2.5-3B 与 GLM-4.7-Flash 的多轮搜索调用 RL（支持同步与异步训练），配套 FAISS 或 Qdrant 本地检索服务。

   .. grid-item-card:: WideSeek-R1
      :link: wideseek_r1/index
      :link-type: doc

      面向广泛信息检索的主从多 Agent 强化学习，4B 模型在 WideSearch 上达到 40.0% item F1。

如需接入自己的 Agent（白盒或 OpenAI API 黑盒接入），见 :doc:`新增 Agent </rst_source/extending/agents/index>`。

图像与视频生成
--------------

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: 图像与视频生成总览
      :link: ../video_models
      :link-type: doc

      SD3.5 Medium 与 Wan2.2 TI2V 5B 的公共扩散环境安装、OCR 提示数据集与权重下载。

   .. grid-item-card:: DiffusionART
      :link: ../diffusion_art
      :link-type: doc

      在 prompt 组内构造共享奖励引导，面向 SD3.5 Medium 图像 OCR 与 Wan2.2 视频 OCR 训练。

   .. grid-item-card:: Diffusion-NFT
      :link: ../diffusion_nft
      :link-type: doc

      使用正负速度候选与 OCR Reward 微调 SD3.5 Medium 和 Wan2.2。

   .. grid-item-card:: ReFL
      :link: ../embodied/refl
      :link-type: doc

      使用可微 ImageReward 反馈与可选监督图文更新微调 SD3.5 Medium。
