π₀-FAST 强化学习
=================

本示例将 LeRobot 的 ``PI0FastPolicy`` 接入 RLinf，用于 LIBERO-Long 的确定性评测和 token-level GRPO 微调。接入层保留 π₀-FAST 原生的自回归动作序列，并在 actor update 阶段通过 teacher forcing 重放 rollout 时采样的 token。

概览
----

在 LIBERO-10（Long）上评测并用 GRPO 微调 LeRobot π₀-FAST。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      LIBERO-10（Long）

   .. grid-item-card:: 算法
      :text-align: center

      GRPO（逐 token clipping）

   .. grid-item-card:: 任务
      :text-align: center

      10 个长程 LIBERO 任务

   .. grid-item-card:: 硬件
      :text-align: center

      2 节点 · 16 GPU（支持单节点覆盖）

| **你将完成：** 安装已验证的 π₀-FAST 运行时 → 下载三个固定版本权重 → 启动 ``run_embodiment.sh``\ （或 ``run_eval.sh``）→ 观察 ``eval/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 已获取 ``google/paligemma-3b-pt-224`` 的 Hugging Face 访问权限。

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 22 34 24

   * - 环境
     - 任务 / 套件
     - 配置 / 权重
     - 重点
   * - LIBERO
     - LIBERO-10（Long）
     - ``libero_10_grpo_pi0_fast``、``lerobot/pi0fast-libero``
     - 使用 all-linear LoRA（rank 16）做逐 token GRPO（验证环境：Python 3.12.12 / PyTorch 2.11.0 + CUDA 12.8）。
   * - LIBERO
     - LIBERO-10（Long）
     - ``libero_10_pi0_fast_eval``
     - 在 500 个固定 episode 上运行确定性贪婪评测。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 字段
     - 说明
   * - 观测
     - 第三人称（agentview）与腕部 RGB 图像，以及 8 维机器人本体状态。
   * - 动作
     - 由原生自回归 FAST 动作 token 解码得到的 10 步连续动作块（每步 7 自由度）。
   * - 奖励
     - 稀疏 LIBERO 任务成功奖励（0/1）。
   * - 提示词
     - LIBERO 自然语言任务指令。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: GRPO 微调
      :sync: grpo

      **1. 安装环境**

      以下命令使用本示例验证过的 π₀-FAST 运行时组合（跳过 Flash Attention；如需安装，去掉 ``--no-flash-attn``）。首次运行前需在 Hugging Face 接受 PaliGemma 访问条款，并执行 ``hf auth login``：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         UV_TORCH_BACKEND=cu128 bash requirements/install.sh embodied \
            --model pi0_fast --env libero \
            --python 3.12.12 --torch 2.11.0 --no-flash-attn && source .venv/bin/activate

      **2. 下载模型**

      启动 RLinf 前先下载三个固定版本的 Hugging Face 资产：

      .. code-block:: bash

         hf download lerobot/pi0fast-libero \
            --revision 840f4b503f4c09110421c33c810a85b6684fd658 \
            --local-dir ./pi0fast-libero
         hf download google/paligemma-3b-pt-224 \
            --revision 35e4f46485b4d07967e7e9935bc3786aad50687c \
            --local-dir ./paligemma-3b-pt-224
         hf download jadechoghari/tokenizer-lib-mean \
            --revision 79ae83e3cbd8786dcb84b628569f8d076ca8151e \
            --local-dir ./tokenizer-lib-mean

      修改 ``examples/embodiment/config/model/pi0_fast.yaml`` 指向下载目录（``num_action_chunks`` 和 ``action_dim`` 必须与 checkpoint 的 ``n_action_steps`` 及 action feature 维度一致，否则会在加载阶段直接报错）：

      .. code-block:: yaml

         model_path: "./pi0fast-libero"
         num_action_chunks: 10
         action_dim: 7
         pi0_fast:
           text_tokenizer_name: "./paligemma-3b-pt-224"
           action_tokenizer_name: "./tokenizer-lib-mean"

      **3. 启动训练**

      ``libero_10_grpo_pi0_fast.yaml`` 默认配置为 **2 节点**\ （``cluster.num_nodes: 2``，共 16 卡；每次 update 采样 1,024 条轨迹：256 并行环境 × 4 rollout epoch，group size 8，micro batch size 16，采样温度 0.3，all-linear LoRA，并启用支持 FSDP1 ``NO_SHARD`` 的可选 FP32-master AdamW）：

      .. code-block:: bash

         # 2 节点（16 卡）参考配置：
         bash examples/embodiment/run_embodiment.sh libero_10_grpo_pi0_fast

         # 单节点快速启动（覆盖 cluster.num_nodes=1）：
         bash examples/embodiment/run_embodiment.sh libero_10_grpo_pi0_fast cluster.num_nodes=1

   .. tab-item:: Baseline 评测
      :sync: eval

      **1. 安装环境**

      安装已验证的 π₀-FAST 运行时（下载前先在 Hugging Face 接受 PaliGemma 访问条款并执行 ``hf auth login``）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         UV_TORCH_BACKEND=cu128 bash requirements/install.sh embodied \
            --model pi0_fast --env libero \
            --python 3.12.12 --torch 2.11.0 --no-flash-attn && source .venv/bin/activate

      **2. 下载模型**

      下载三个固定版本的 Hugging Face 资产，并按 GRPO 标签页中的示例更新 ``examples/embodiment/config/model/pi0_fast.yaml``：

      .. code-block:: bash

         hf download lerobot/pi0fast-libero \
            --revision 840f4b503f4c09110421c33c810a85b6684fd658 \
            --local-dir ./pi0fast-libero
         hf download google/paligemma-3b-pt-224 \
            --revision 35e4f46485b4d07967e7e9935bc3786aad50687c \
            --local-dir ./paligemma-3b-pt-224
         hf download jadechoghari/tokenizer-lib-mean \
            --revision 79ae83e3cbd8786dcb84b628569f8d076ca8151e \
            --local-dir ./tokenizer-lib-mean

      **3. 启动评测**

      运行确定性 baseline 评测（greedy decoding、seed 0、500 个有序固定 LIBERO episode；与训练共用同一套 FAST 采样与 detokenize 路径，不收集 replay log-probability）：

      .. code-block:: bash

         bash evaluations/run_eval.sh libero libero_10_pi0_fast_eval

.. dropdown:: 策略语义与进阶选项

   **固定依赖坐标**

   .. list-table::
      :header-rows: 1
      :widths: 28 38 34

      * - 依赖
        - 仓库
        - revision
      * - LeRobot 源码
        - ``huggingface/lerobot``
        - ``3e37269dc60ee6195e44dd2810c95f1df5dfda99``
      * - 策略 checkpoint
        - ``lerobot/pi0fast-libero``
        - ``840f4b503f4c09110421c33c810a85b6684fd658``
      * - 文本 tokenizer
        - ``google/paligemma-3b-pt-224``
        - ``35e4f46485b4d07967e7e9935bc3786aad50687c``
      * - 动作 tokenizer
        - ``jadechoghari/tokenizer-lib-mean``
        - ``79ae83e3cbd8786dcb84b628569f8d076ca8151e``

   **策略语义**

   π₀-FAST 原生生成完整动作字符串，RLinf 不预先注入 ``Action:`` 前缀。policy mask 包含模型生成的前缀、动作正文和第一个完整的 ``|`` 结束标记，不包含 padding 和结束标记之后的 token。同一条轨迹的所有 token 共享轨迹级 GRPO advantage；PPO 对每个 token 独立 clipping，之后再按 mask 聚合 token loss。

   非法序列不会重采样，而是在 action postprocessor 之后执行安全零动作，由环境正常返回失败反馈，并继续参与策略目标。这样可以保持 on-policy 采样，且训练与评测行为一致。

可视化与结果
--------------------

除了 ``env/success_once`` 和 ``eval/success_once``，建议关注分组成功直方图与保留比例、token entropy、gradient norm、``approx_kl``，以及 log-ratio 的 finite/min/max 指标（完整指标见 :doc:`训练指标 <../../reference/metrics>`）。第一次 actor update 在优化器修改权重前，应满足 replay logprob 全部 finite 且 ratio 接近 1。

**Baseline 评测（500 episodes）**

.. list-table::
   :header-rows: 1
   :widths: 25 25 25

   * - Episode 数
     - ``success_once``
     - ``success_at_end``
   * - 500
     - 85.8%
     - 75.8%

**GRPO 微调（seed 1234 开发长训）**

一次 actor seed 1234 的开发阶段长训（每 10 步在 256 个固定 episode 上评测）得到以下 ``success_once``\ （用于验证接入链路，不代表多 seed 收敛保证）：

.. list-table::
   :header-rows: 1
   :widths: 20 20

   * - Step
     - ``success_once``
   * - 290
     - 95.70%
   * - 300
     - 99.22%
   * - 310
     - 95.70%
   * - 320
     - 97.27%
   * - 330
     - 94.92%
   * - 均值
     - 96.56%
