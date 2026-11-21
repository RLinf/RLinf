使用 PPO 训练 Math 推理任务
========================================

需要在 :doc:`reasoning` 中的同一数学推理任务上使用 actor-critic PPO 时，使用本配方。PPO 与 GRPO 共享大部分启动和数据设置，因此本页只列出 PPO 差异。

概述
----------------------------------------

希望在与 GRPO 示例相同的数学推理任务上使用 actor-critic PPO 时，请使用本配方。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen2.5-1.5B

   .. grid-item-card:: 算法
      :text-align: center

      PPO，使用 GAE 优势估计与 critic

   .. grid-item-card:: 数据
      :text-align: center

      AReaL-boba 数学推理数据

   .. grid-item-card:: 硬件
      :text-align: center

      多 GPU Megatron 训练

PPO 工作方式
----------------------------------------

使用带 GAE advantage 和 critic 的标准 PPO（Proximal Policy Optimization）。算法细节见 :doc:`PPO <../../../reference/algorithms/ppo>`。

一键上手
----------------------------------------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__。通用路径、集群和 runner 字段见 :doc:`训练配置 <../../../reference/configuration>`。

.. tab-set::

   .. tab-item:: Qwen2.5-1.5B-Instruct（Megatron PPO）

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. 下载模型与数据**

      下载基础模型和 boba 数据集（数据格式细节见 :doc:`reasoning`）：

      .. code-block:: bash

         hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./Qwen2.5-1.5B-Instruct
         hf download --repo-type dataset inclusionAI/AReaL-boba-Data --local-dir ./AReaL-boba-Data

      **3. 启动训练**

      在 ``examples/reasoning/config/math/qwen2.5-1.5b-ppo-megatron.yaml`` 中填写模型与数据路径。PPO 与 GRPO 共用入口脚本 ``run_main_grpo_math.sh``，RLinf 根据 YAML 中的 critic 配置及 ``adv_type`` 取值（通常为 ``gae``）自动启用 PPO 训练：

      .. code-block:: bash

         bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-ppo-megatron


可视化与结果
----------------------------------------

我们基于 Qwen2.5-1.5B-Instruct 模型使用 PPO 训练。橙色为 RLinf，蓝色为 VeRL；两者使用相同算法配置。通用指标含义见 :doc:`训练指标 <../../../reference/metrics>`。

由于 Qwen2.5-1.5B-Instruct 模型基础能力较弱，整体 reward 数值较低，但随着训练推进，reward 数值明显上升。

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/ppo_rlinf_vs_verl.jpg" style="width: 50%;"/>
       <p><em>MATH 1.5B PPO</em></p>
     </div>
   </div>
