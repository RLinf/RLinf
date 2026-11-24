Moonlight-16B（MoE GRPO）
=========================

使用 :doc:`GRPO </rst_source/reference/algorithms/grpo>` 在 AReaL-boba 数学数据上训练 Moonlight-16B-A3B-Instruct（DeepSeek-V3 MLA + 64 专家 MoE，约 3B 激活参数），通过 Megatron-Bridge（TP=8、EP=8）与 SGLang 在单机 8 卡上运行。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Moonlight-16B-A3B-Instruct（MLA + MoE）

   .. grid-item-card:: 算法
      :text-align: center

      GRPO（token 级 Loss，Megatron-Bridge + SGLang）

   .. grid-item-card:: 任务
      :text-align: center

      数学推理（AReaL-boba）

   .. grid-item-card:: 硬件
      :text-align: center

      8×A100（TP=8、EP=8）

| **你将完成：** 安装 ``agentic`` 依赖 → 下载 Moonlight-16B 模型与 AReaL-boba 数据集 → 运行 ``run_main_grpo_math.sh`` → 在 TensorBoard 查看 Reward 曲线。
| **前置条件：** :doc:`安装与硬件参考 </rst_source/start/installation>` · 8×A100 GPU · NVIDIA 驱动 ≥ 580 及 CUDA 13.3 工具链（``nvcc``）。

安装与运行
----------

中国大陆用户可在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__。

.. tab-set::

   .. tab-item:: Moonlight-16B-A3B-Instruct

      **1. 安装环境**

      默认 ``agentic`` 环境包含 SGLang 0.5.12.post1、PyTorch 2.11、Megatron-Core 0.18 与 ``megatron-bridge`` 0.5.0：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. 下载模型与数据**

      下载模型和 `AReaL-boba-Data <https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/>`__：

      .. code-block:: bash

         hf download moonshotai/Moonlight-16B-A3B-Instruct --local-dir ./Moonlight-16B-A3B-Instruct
         hf download --repo-type dataset inclusionAI/AReaL-boba-Data --local-dir ./AReaL-boba-Data

      **3. 启动训练**

      使用 ``examples/reasoning/config/math/moonlight-16b-grpo-megatron.yaml`` 启动单机 8 卡训练：

      .. code-block:: bash

         bash examples/reasoning/run_main_grpo_math.sh moonlight-16b-grpo-megatron \
           rollout.model.model_path=./Moonlight-16B-A3B-Instruct \
           actor.tokenizer.tokenizer_model=./Moonlight-16B-A3B-Instruct \
           data.train_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]' \
           data.val_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]'

.. dropdown:: 进阶配置与多节点扩展（可选）

   ``examples/reasoning/config/math/moonlight-16b-grpo-megatron.yaml`` 默认启用 DeepSeek-V3 MLA + MoE（TP=8、EP=8）、Megatron-Bridge 权重转换、SGLang 专家并行 rollout、precision-aware optimizer 以及 +5 / -5 数学 Reward。

   - **数据集字段**：自定义 prompt、answer 和 chat template 字段见 :doc:`Qwen2.5 数学推理 GRPO <reasoning>`。

   - **多节点扩展**：按 :doc:`/rst_source/guides/multi_node` 启动 Ray 集群并调大 ``cluster.num_nodes``，在 head 节点运行启动命令。

可视化与结果
------------

运行 ``tensorboard --logdir ./logs --port 6006`` 查看训练曲线；指标含义及 W&B / SwanLab 配置见 :doc:`训练指标 </rst_source/reference/metrics>`。

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/moonlight/moonlight_grpo_reward.jpeg" style="width: 100%;"/>
       <p><em>Moonlight-16B GRPO reward 曲线</em></p>
     </div>
   </div>
