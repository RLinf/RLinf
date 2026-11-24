Qwen3-VL 视觉推理（Geo3K）
==========================

使用带截断重要性采样（TIS）的 :doc:`GRPO </rst_source/reference/algorithms/grpo>`，基于 FSDP2 与 SGLang 在 Geo3K 几何推理数据集上训练 Qwen3-VL-2B-Instruct。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen3-VL-2B-Instruct

   .. grid-item-card:: 算法
      :text-align: center

      GRPO + TIS（FSDP2 + SGLang）

   .. grid-item-card:: 任务
      :text-align: center

      几何视觉推理（Geo3K）

   .. grid-item-card:: 硬件
      :text-align: center

      单机（1–8 张 GPU）

| **你将完成：** 安装 Qwen3-VL 依赖 → 下载 Qwen3-VL-2B-Instruct 与 Geo3K 数据集 → 运行 ``run_main_grpo_vqa.sh`` → 观察 Reward 与熵曲线。
| **前置条件：** :doc:`安装与硬件参考 </rst_source/start/installation>` · ``torch >= 2.8.0``、``sglang == 0.5.4`` 与 ``transformers == 4.57.1``\ （下方命令自动安装）。

安装与运行
----------

中国大陆用户可在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__。

.. tab-set::

   .. tab-item:: Qwen3-VL-2B-Instruct

      **1. 安装环境**

      安装 Qwen3-VL 所需依赖版本（FSDP2 无需 Apex，``--no-apex`` 可跳过编译）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic \
             --torch 2.8.0 \
             --sglang 0.5.4 \
             --transformers 4.57.1 \
             --no-apex && source .venv/bin/activate

      **2. 下载模型与数据**

      下载基础模型和 `Geo3K 数据集 <https://huggingface.co/datasets/CAIR-HKISI/geo3k>`__：

      .. code-block:: bash

         hf download Qwen/Qwen3-VL-2B-Instruct --local-dir ./Qwen3-VL-2B-Instruct
         hf download --repo-type dataset CAIR-HKISI/geo3k --local-dir ./geo3k

      **3. 启动训练**

      使用 ``examples/reasoning/config/vqa/qwen3-vl-2b-grpo-fsdp-geo3k.yaml`` 启动单机训练（actor、rollout 与 reward 共置在全部 GPU 上）：

      .. code-block:: bash

         bash examples/reasoning/run_main_grpo_vqa.sh qwen3-vl-2b-grpo-fsdp-geo3k \
           rollout.model.model_path=./Qwen3-VL-2B-Instruct \
           data.train_data_paths='["./geo3k/data/train-00000-of-00001.parquet"]' \
           data.val_data_paths='["./geo3k/data/validation-00000-of-00001.parquet"]'

.. dropdown:: 数据集格式与进阶配置（可选）

   Geo3K 样本包含 ``problem``\ （含 ``<image>`` 占位符）、``images``\ （图像字节数组）和 ``answer``\ （``\boxed{x}``）。若 prompt 不含图像占位符，框架会将图像放在文本之前。

   .. list-table::
      :header-rows: 1
      :widths: 34 66

      * - 配置项
        - 说明
      * - ``data.prompt_key`` / ``data.answer_key`` / ``data.image_keys``
        - 映射自定义 VQA 字段（其他来源的 base64 或 PIL 图像需先转换格式）。
      * - ``data.system_prompt``
        - 在原始问题前统一追加提示词，如 ``Answer: \boxed{$Answer}``。
      * - ``algorithm.importance_sampling_fix`` / ``importance_sampling_clip``
        - 开启 TIS（clip = 2），抑制熵发散与输出过长。
      * - ``reward.reward_max_val`` / ``reward.reward_min_val``
        - 调整正确 / 错误答案的 Reward 分值（默认 +1 / 0）。
      * - ``runner.enable_dynamic_batch_size`` / ``runner.max_tokens_per_mbs``
        - 控制 FSDP2（``actor.fsdp_config.strategy: fsdp2``）下的动态序列打包。

   同时开启 ``recompute_logprobs`` 与 ``return_logprobs`` 后会自动记录 ``actor/rollout_train_kl``。多机扩展按 :doc:`/rst_source/guides/multi_node` 启动 Ray 集群，在 head 节点运行启动命令。

可视化与结果
------------

运行 ``tensorboard --logdir ./logs --port 6006`` 查看训练曲线；指标含义及 W&B / SwanLab 配置见 :doc:`训练指标 </rst_source/reference/metrics>`。

在 Qwen3-VL-2B-Instruct 上开启 TIS 后，训练至 750 步 Reward 仍持续上升，熵曲线趋于收敛：

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/qwen3vl_grpo_reward.jpeg" style="width: 100%;"/>
       <p><em>开启 TIS 的 Reward 曲线</em></p>
     </div>
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/qwen3vl_grpo_entropy.jpeg" style="width: 100%;"/>
       <p><em>开启 TIS 的熵曲线</em></p>
     </div>
   </div>

关闭 TIS 时，Reward 往往在训练中途崩溃，熵曲线也更不稳定：

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/qwen3vl_grpo_reward_wo_tis.jpeg" style="width: 100%;"/>
       <p><em>关闭 TIS 的 Reward 曲线</em></p>
     </div>
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/qwen3vl_grpo_entropy_wo_tis.jpeg" style="width: 100%;"/>
       <p><em>关闭 TIS 的熵曲线</em></p>
     </div>
   </div>
