Agent & LLM
===========

Train text and multimodal reasoning models, multi-turn tool-using agents, multi-agent systems, and diffusion models with RLinf's distributed worker runtime, Megatron / FSDP training backends, and SGLang / vLLM rollout engines.

Before starting a reasoning or agent recipe, install the agentic stack with ``bash requirements/install.sh agentic`` (see :doc:`Installation <../../start/installation>`). Image and video generation recipes use ``bash requirements/install.sh embodied --model diffusion``.

Choose a Track
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 26 22 32

   * - Track
     - Models & Backends
     - Algorithm / Mode
     - Recipes
   * - :doc:`Reasoning <math_reasoning/index>`
     - Moonlight-16B, Qwen2.5, Qwen3-VL (Megatron / FSDP + SGLang / vLLM)
     - GRPO, PPO (text & visual reasoning)
     - :doc:`Moonlight-16B <math_reasoning/moonlight>`, :doc:`Qwen2.5 GRPO <math_reasoning/reasoning>`, :doc:`Qwen2.5 PPO <math_reasoning/reasoning_ppo>`, :doc:`Qwen3-VL Geo3K <qwen3_vl_geo3k>`
   * - :doc:`Agentic RL <agents/index>`
     - Agent Lightning, GLM-4.7-Flash, Qwen2.5, Qwen3-4B
     - Multi-turn tools, MARL, online RL
     - :doc:`AgentLightning <agentlightning_calc_x>`, :doc:`Coding Online RL <coding_online_rl>`, :doc:`rStar2 <rstar2>`, :doc:`Search-R1 <searchr1/index>`, :doc:`WideSeek-R1 <wideseek_r1/index>`
   * - :doc:`Image & Video Generation <../video_models>`
     - SD3.5 Medium, Wan2.2 TI2V 5B (Diffusers)
     - DiffusionART, Diffusion-NFT, ReFL
     - :doc:`DiffusionART <../diffusion_art>`, :doc:`Diffusion-NFT <../diffusion_nft>`, :doc:`ReFL <../embodied/refl>`

Reasoning
---------

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: Moonlight-16B (MoE GRPO)
      :link: math_reasoning/moonlight
      :link-type: doc

      GRPO training for DeepSeek-V3 MLA + MoE architecture via Megatron-Bridge.

   .. grid-item-card:: Qwen2.5 Math (GRPO)
      :link: math_reasoning/reasoning
      :link-type: doc

      GRPO training for DeepSeek-R1-Distill-Qwen (1.5B / 7B) on AReaL-boba, evaluated on AIME24, AIME25, and GPQA-diamond.

   .. grid-item-card:: Qwen2.5 Math (PPO)
      :link: math_reasoning/reasoning_ppo
      :link-type: doc

      PPO math reasoning on Qwen2.5-1.5B updating both actor and critic.

   .. grid-item-card:: Qwen3-VL Visual Reasoning (Geo3K)
      :link: qwen3_vl_geo3k
      :link-type: doc

      Vision-language GRPO training on geometric reasoning (Geo3K) with FSDP and SGLang.

Agentic RL
----------

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: AgentLightning
      :link: agentlightning_calc_x
      :link-type: doc

      Use Agent Lightning for agent workflow orchestration while delegating distributed RL training to RLinf on Calc-X.

   .. grid-item-card:: Coding Online RL
      :link: coding_online_rl
      :link-type: doc

      End-to-end online RL for IDE code completion combining RLinf with the Continue extension.

   .. grid-item-card:: rStar2
      :link: rstar2
      :link-type: doc

      Multi-turn mathematical reasoning and self-reflection using a local Python execution server.

   .. grid-item-card:: Search-R1
      :link: searchr1/index
      :link-type: doc

      Multi-turn search tool RL on Qwen2.5-3B and GLM-4.7-Flash (sync and async), with FAISS or Qdrant local wiki retrieval.

   .. grid-item-card:: WideSeek-R1
      :link: wideseek_r1/index
      :link-type: doc

      Lead-agent and parallel subagent MARL for broad information seeking on WideSearch, reaching 40.0% item F1 with a 4B model.

To plug in your own white-box or OpenAI API black-box agent, see :doc:`Add an Agent </rst_source/extending/agents/index>`.

Image & Video Generation
------------------------

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: Image & Video Generation Overview
      :link: ../video_models
      :link-type: doc

      Shared diffusion setup, OCR prompt dataset, and checkpoint downloads for SD3.5 Medium and Wan2.2 TI2V 5B.

   .. grid-item-card:: DiffusionART
      :link: ../diffusion_art
      :link-type: doc

      Group reward guidance against frozen velocity targets for SD3.5 Medium image OCR and Wan2.2 video OCR.

   .. grid-item-card:: Diffusion-NFT
      :link: ../diffusion_nft
      :link-type: doc

      RL fine-tuning of SD3.5 Medium and Wan2.2 with positive/negative velocity candidates and OCR rewards.

   .. grid-item-card:: ReFL
      :link: ../embodied/refl
      :link-type: doc

      Fine-tune SD3.5 Medium with differentiable ImageReward feedback and optional supervised image-text updates.
