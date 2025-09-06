快速开始
==========

欢迎使用 RLinf 快速上手指南。本节将带你一步步运行 RLinf，帮助你快速了解整个框架的使用流程。

我们提供了三个简洁示例，展示 RLinf 的基本工作流程，帮助你快速开始使用：

- **安装方式：** RLinf 支持两种安装方法：使用 Docker 镜像，或自定义用户环境（详见 :doc:`installation`）。

- **具身智能训练：** 在 ManiSkill3 环境中，使用 PPO 算法对 OpenVLA 和 OpenVLA-OFT 模型进行训练（详见 :doc:`vla`）。

- **数学任务训练：** 使用 GRPO 算法，在 boba 数据集上训练 DeepSeek-R1-Distill-Qwen-1.5B 模型（详见 :doc:`llm`）。

- **分布式训练：** 支持多节点数学任务训练（详见 :doc:`distribute`）。

- **模型评估：** 评估模型在具身智能场景任务下的表现（详见 :doc:`vla-eval`）， 以及评估模型在长链式数学推理任务中的表现（详见 :doc:`llm-eval`）。


.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   vla
   llm
   distribute
   vla-eval
   llm-eval
