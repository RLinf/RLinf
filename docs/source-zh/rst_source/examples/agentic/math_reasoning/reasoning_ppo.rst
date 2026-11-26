Qwen2.5 数学推理（PPO）
=======================

使用带 critic 和 GAE Advantage 的 :doc:`PPO </rst_source/reference/algorithms/ppo>` 在 AReaL-boba 数学数据上训练 Qwen2.5-1.5B-Instruct。本配方与 :doc:`Qwen2.5 数学推理 GRPO <reasoning>` 共用启动脚本和数据集格式。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen2.5-1.5B-Instruct

   .. grid-item-card:: 算法
      :text-align: center

      PPO（GAE + critic，Megatron + SGLang）

   .. grid-item-card:: 任务
      :text-align: center

      数学推理（AReaL-boba）

   .. grid-item-card:: 硬件
      :text-align: center

      单机（1–8 张 GPU）

| **你将完成：** 安装 ``agentic`` 依赖 → 下载模型与 AReaL-boba 数据集 → 使用 PPO 配置运行 ``run_main_grpo_math.sh`` → 观察 Reward 曲线。
| **前置条件：** :doc:`安装与硬件参考 </rst_source/start/installation>` · 支持 SGLang 与 Megatron 的 NVIDIA GPU 节点。

安装与运行
----------

中国大陆用户可在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__。

.. tab-set::

   .. tab-item:: Qwen2.5-1.5B-Instruct

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. 下载模型与数据**

      下载基础模型和 `AReaL-boba-Data <https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/>`__：

      .. code-block:: bash

         hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./Qwen2.5-1.5B-Instruct
         hf download --repo-type dataset inclusionAI/AReaL-boba-Data --local-dir ./AReaL-boba-Data

      **3. 启动训练**

      PPO 复用 ``run_main_grpo_math.sh``，由 ``examples/reasoning/config/math/qwen2.5-1.5b-ppo-megatron.yaml`` 中的 ``critic.use_critic_model: true`` 与 ``algorithm.adv_type: gae`` 开启 critic 与 GAE。通过 Hydra 参数指定本地模型、数据与输出目录：

      .. code-block:: bash

         bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-ppo-megatron \
           rollout.model.model_path=./Qwen2.5-1.5B-Instruct \
           data.train_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]' \
           data.val_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]' \
           runner.output_dir=./results

.. dropdown:: 进阶配置与多节点扩展（可选）

   - **数据集字段**：自定义 ``data.prompt_key``、``data.answer_key`` 与 ``data.apply_chat_template`` 见 :doc:`Qwen2.5 数学推理 GRPO <reasoning>`。
   - **多节点扩展**：按 :doc:`/rst_source/guides/multi_node` 启动 Ray 集群并调大 ``cluster.num_nodes``，在 head 节点运行启动命令。通用集群与 runner 字段见 :doc:`训练配置 </rst_source/reference/configuration>`。

可视化与结果
------------

运行 ``tensorboard --logdir ./results --port 6006`` 查看训练曲线；指标含义及 W&B / SwanLab 配置见 :doc:`训练指标 </rst_source/reference/metrics>`。

下图对比了 RLinf（橙线）与 VeRL（蓝线）在相同 PPO 配置下微调 Qwen2.5-1.5B-Instruct 的曲线。该基座模型初始能力较弱，但 Reward 随训练推进稳步上升。

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/ppo_rlinf_vs_verl.jpg" style="width: 50%;"/>
       <p><em>MATH 1.5B PPO</em></p>
     </div>
   </div>
