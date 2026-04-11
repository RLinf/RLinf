Quickstart
==========

Welcome to the RLinf Quickstart Guide. This section will walk you through launching RLinf for the first time. 
We present three concise examples to demonstrate the framework's workflow and help you get started quickly.


- **Installation:** Two installation methods for RLinf are supported: using a Docker image or a custom user environment (see :doc:`installation`).

- **Embodied training:** Training in the ManiSkill3 environment with the OpenVLA and OpenVLA-OFT models using the PPO algorithm (see :doc:`vla`).

- **Agentic training:** Training on the boba dataset with the DeepSeek-R1-Distill-Qwen-1.5B model using the GRPO algorithm (see :doc:`llm`).

- **Distributed training:** Multi-node training for embodied/agentic tasks (see :doc:`distribute`).

- **Evaluation:** Assessing model performance on embodied intelligence (see :doc:`vla-eval`) and assessing model performance on long-chain-of-thought agentic tasks (see :doc:`llm-eval`).


SOTA RL Training Reproduction
=====================================

RLinf provides end-to-end recipes that reproduce or match **state-of-the-art (SOTA) RL results** out of the box—users can directly run our configs and scripts to obtain published numbers without custom engineering.

For embodied tasks, RLinf reaches or matches SOTA success rates on benchmarks such as **LIBERO**, **ManiSkill**, **RoboTwin**, and more with OpenVLA, OpenVLA-OFT, π₀/π₀.₅, GR00T and other VLAs (see the :doc:`../examples/embodied/index` gallery and :doc:`../tutorials/rlalg/index` for algorithm details).

For agentic tasks (including math reasoning), RLinf achieves SOTA performance on **AIME24/AIME25/GPQA-diamond** benchmarks with DeepSeek-R1-Distill-Qwen models, and supports single-agent and multi-agent training tasks such as Search-R1 and Coding-Online-RL (see :doc:`../examples/agentic/index`).

.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   vla
   llm
   distribute
   vla-eval
   llm-eval
