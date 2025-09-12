Quickstart
==========

Welcome to the RLinf Quickstart Guide. This section will walk you through launching RLinf for the first time. 
We present three concise examples to demonstrate the framework's workflow and help you get started quickly.


- **Installation:** Two installation methods for RLinf are supported: using a Docker image or a custom user environment (see :doc:`installation`).

- **Embodied training:** Training in the ManiSkill3 environment with the OpenVLA and OpenVLA-OFT models using the PPO algorithm (see :doc:`vla`).

- **Mathematical training:** Training on the boba dataset with the DeepSeek-R1-Distill-Qwen-1.5B model using the GRPO algorithm (see :doc:`llm`).

- **Distributed training:** Multi-node training for mathematical tasks (see :doc:`distribute`).

- **Evaluation:** Assessing model performance on embodied intelligence (see :doc:`vlm-eval`) and assessing model performance on long-chain-of-thought mathematical reasoning (see :doc:`llm-eval`).

.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   vla
   llm
   distribute
   vla-eval
   llm-eval
