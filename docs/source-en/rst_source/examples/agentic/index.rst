Agent & LLM
===========

Train reasoning LLMs, multi-turn tool-using agents, multi-agent systems, and diffusion models with RLinf's distributed worker runtime, Megatron/FSDP training backends, and SGLang/vLLM rollout engines.

Before starting an agentic or reasoning recipe, install the agentic stack with ``bash requirements/install.sh agentic`` (see :doc:`Installation <../../start/installation>`). Image and video generation recipes use ``bash requirements/install.sh embodied --model diffusion``.

Starting on a single node or a single GPU? Begin with :doc:`Qwen3-VL GRPO <qwen3_vl_geo3k>` (single-node FSDP2), :doc:`Qwen2.5 Math GRPO <math_reasoning/reasoning>` using its single-node quick-start override, or :doc:`SD3.5 Medium <../video_models>` for image generation.

Choose a Track
--------------

.. list-table::
   :header-rows: 1
   :widths: 22 24 24 30

   * - Track
     - Models & Backends
     - Algorithm / Mode
     - Recipes
   * - :doc:`Math Reasoning <math_reasoning/index>`
     - Qwen2.5, Moonlight-16B (Megatron + SGLang/vLLM)
     - GRPO, PPO
     - :doc:`Qwen2.5 GRPO <math_reasoning/reasoning>`, :doc:`Qwen2.5 PPO <math_reasoning/reasoning_ppo>`, :doc:`Moonlight-16B <math_reasoning/moonlight>`
   * - :doc:`Search-R1 <searchr1/index>`
     - Qwen2.5-3B, GLM-4.7-Flash (local Wiki / Qdrant)
     - Multi-turn search RL (sync & async)
     - :doc:`Qwen2.5-3B <searchr1/qwen2.5>`, :doc:`GLM-4.7-Flash <searchr1/glm4.7_flash>`, :doc:`Async Training <searchr1/async>`
   * - :doc:`WideSeek-R1 <wideseek_r1/index>`
     - Qwen3-4B lead-agent + parallel subagents
     - Multi-agent RL (MARL)
     - :doc:`Setup <wideseek_r1/tools>`, :doc:`Training <wideseek_r1/train>`, :doc:`Evaluation <wideseek_r1/eval>`
   * - :doc:`rStar2 <rstar2>`
     - Qwen2.5-7B + Python tool server
     - Multi-turn reflection RL
     - :doc:`rStar2-Agent <rstar2>`
   * - :doc:`AgentLightning <agentlightning_calc_x>`
     - Agent Lightning orchestration + RLinf backend
     - Tool-use RL (Calc-X)
     - :doc:`AgentLightning Calc-X <agentlightning_calc_x>`
   * - :doc:`Coding Online RL <coding_online_rl>`
     - Qwen2.5-Coder-1.5B + Continue IDE
     - Online PPO / GRPO
     - :doc:`Coding Online RL <coding_online_rl>`
   * - :doc:`Qwen3-VL GRPO <qwen3_vl_geo3k>`
     - Qwen3-VL (FSDP + SGLang)
     - Multimodal GRPO (Geo3K)
     - :doc:`Qwen3-VL Geo3K <qwen3_vl_geo3k>`
   * - :doc:`Image & Video Generation <../video_models>`
     - SD3.5 Medium, Wan2.2 TI2V 5B (Diffusers)
     - Diffusion-NFT, DiffusionART, ReFL
     - :doc:`Diffusion-NFT <../diffusion_nft>`, :doc:`DiffusionART <../diffusion_art>`, :doc:`ReFL <../embodied/refl>`

Reasoning & Multimodal RL
-------------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Math Reasoning
      :link: math_reasoning/index
      :link-type: doc

      GRPO and PPO recipes on Qwen2.5 (1.5B/7B) and Moonlight-16B-A3B (DeepSeek-V3 MLA + MoE) for AIME24, AIME25, and GPQA-diamond.

   .. grid-item-card:: Qwen3-VL GRPO
      :link: qwen3_vl_geo3k
      :link-type: doc

      Vision-language GRPO training on geometric reasoning (Geo3K) with FSDP and SGLang.

Tool-Using & Multi-Agent RL
---------------------------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Search-R1
      :link: searchr1/index
      :link-type: doc

      Multi-turn search tool RL on Qwen2.5-3B and GLM-4.7-Flash (sync and async), with FAISS or Qdrant local wiki retrieval.

   .. grid-item-card:: WideSeek-R1
      :link: wideseek_r1/index
      :link-type: doc

      Lead-agent and parallel subagent MARL for broad information seeking on WideSearch, reaching 40.0% item F1 with a 4B model.

   .. grid-item-card:: rStar2
      :link: rstar2
      :link-type: doc

      Multi-turn mathematical reasoning and self-reflection using a local Python execution server.

   .. grid-item-card:: AgentLightning
      :link: agentlightning_calc_x
      :link-type: doc

      Use Agent Lightning for agent workflow orchestration while delegating distributed RL training to RLinf on Calc-X.

   .. grid-item-card:: Coding Online RL
      :link: coding_online_rl
      :link-type: doc

      End-to-end online RL for IDE code completion combining RLinf with the Continue extension.

Image & Video Generation
------------------------

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: Image & Video Generation Overview
      :link: ../video_models
      :link-type: doc

      Shared diffusion setup, OCR prompt dataset, and checkpoint downloads for SD3.5 Medium and Wan2.2 TI2V 5B.

   .. grid-item-card:: Diffusion-NFT
      :link: ../diffusion_nft
      :link-type: doc

      RL fine-tuning of SD3.5 Medium and Wan2.2 with positive/negative velocity candidates and OCR rewards.

   .. grid-item-card:: DiffusionART
      :link: ../diffusion_art
      :link-type: doc

      Group reward guidance against frozen velocity targets for SD3.5 Medium image OCR and Wan2.2 video OCR.

   .. grid-item-card:: ReFL
      :link: ../embodied/refl
      :link-type: doc

      Fine-tune SD3.5 Medium with differentiable ImageReward feedback and optional supervised image-text updates.
