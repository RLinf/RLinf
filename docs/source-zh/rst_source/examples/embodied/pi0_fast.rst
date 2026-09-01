基于 LeRobot 的 PI0-FAST 强化学习
============================================================

本示例将 LeRobot 的 ``PI0FastPolicy`` 接入 RLinf，用于 LIBERO-Long 的确定性
评测和 token-level GRPO 微调。接入层保留 PI0-FAST 原生的自回归动作序列，并在
actor update 阶段通过 teacher forcing 重放 rollout 时采样的 token。

概览
--------------------

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - 项目
     - 配置
   * - 环境
     - LIBERO-10（Long）
   * - 基础策略
     - ``lerobot/pi0fast-libero``
   * - 算法
     - GRPO，逐 token PPO clipping
   * - 可训练参数
     - all-linear LoRA，rank 16
   * - 已验证运行时
     - Python 3.12.12、PyTorch 2.11.0 + CUDA 12.8、Transformers 5.5.4

安装
--------------------

下面的命令使用本示例验证过的 PI0-FAST 运行时组合。这些版本表示已测试配置，安装脚本
不会强制覆盖用户传入的版本：

.. code:: bash

   UV_TORCH_BACKEND=cu128 bash requirements/install.sh embodied \
      --model pi0_fast --env libero \
      --python 3.12.12 --torch 2.11.0 --no-flash-attn
   source .venv/bin/activate

需要 GitHub 和 PyPI 镜像时可增加 ``--use-mirror``。上述已验证命令跳过 Flash
Attention；如需安装，可去掉 ``--no-flash-attn``。
首次运行前需要在 Hugging Face 接受 PaliGemma 的访问条款，并执行 ``hf auth login``；
固定版本的文本 tokenizer 位于该 gated repository 中。

固定依赖坐标
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 38 34

   * - 依赖
     - 仓库
     - revision
   * - LeRobot 源码
     - ``huggingface/lerobot``
     - ``8a74e0ac6d01706d67fddfed682a09d694d9c8c0``
   * - 策略 checkpoint
     - ``lerobot/pi0fast-libero``
     - ``840f4b503f4c09110421c33c810a85b6684fd658``
   * - 文本 tokenizer
     - ``google/paligemma-3b-pt-224``
     - ``35e4f46485b4d07967e7e9935bc3786aad50687c``
   * - 动作 tokenizer
     - ``jadechoghari/tokenizer-lib-mean``
     - ``79ae83e3cbd8786dcb84b628569f8d076ca8151e``

启动 RLinf 前先下载三个资产：

.. code:: bash

   hf download lerobot/pi0fast-libero \
      --revision 840f4b503f4c09110421c33c810a85b6684fd658 \
      --local-dir /path/to/pi0fast-libero
   hf download google/paligemma-3b-pt-224 \
      --revision 35e4f46485b4d07967e7e9935bc3786aad50687c \
      --local-dir /path/to/paligemma-3b-pt-224
   hf download jadechoghari/tokenizer-lib-mean \
      --revision 79ae83e3cbd8786dcb84b628569f8d076ca8151e \
      --local-dir /path/to/tokenizer-lib-mean

然后修改 ``examples/embodiment/config/model/pi0_fast.yaml`` 中的三个字段：

.. code:: yaml

   model_path: "/path/to/pi0fast-libero"
   pi0_fast:
     text_tokenizer_name: "/path/to/paligemma-3b-pt-224"
     action_tokenizer_name: "/path/to/tokenizer-lib-mean"

GRPO 微调
------------------------

使用下面的命令启动双机 16 卡参考配置：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh libero_10_grpo_pi0_fast

参考配置每次 update 采样 1,024 条轨迹（256 个并行环境、4 个 rollout epoch），
group size 为 8，actor micro batch size 为 16，并且每 10 步在 256 个固定 episode
上评测。训练采样温度为 0.3，评测使用 greedy decoding。actor 采用 all-linear
LoRA，并启用可选的 FP32-master AdamW。该优化器支持 FSDP1 的 ``NO_SHARD``
和 ``FULL_SHARD``；参考配置使用 ``NO_SHARD``。

一次 actor seed 1234 的开发阶段长训得到以下 ``success_once``。这些数据用于证明
接入链路可以训练，不代表多 seed 的收敛保证：

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

Baseline 评测
------------------------

评测配置采用 greedy decoding、seed 0、LIBERO 有序固定 reset state，共运行 500 个
episode：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh libero_10_eval_pi0_fast

固定运行时和依赖坐标后，开发阶段得到以下结果：

.. list-table::
   :header-rows: 1
   :widths: 25 25 25

   * - Episode 数
     - ``success_once``
     - ``success_at_end``
   * - 500
     - 85.8%
     - 75.8%

策略语义
--------------------

PI0-FAST 原生生成完整动作字符串，RLinf 不预先注入 ``Action:`` 前缀。policy mask
包含模型生成的前缀、动作正文和第一个完整的 ``|`` 结束标记，不包含 padding 和结束
标记之后的 token。同一条轨迹的所有 token 共享轨迹级 GRPO advantage；PPO 对每个
token 独立 clipping，之后再按 mask 聚合 token loss。

非法序列不会重采样，而是执行安全零动作，由环境正常返回失败反馈，并继续参与策略
目标。这样可以保持 on-policy 采样。

监控指标
--------------------

除了 ``env/success_once`` 和 ``eval/success_once``，建议关注分组成功直方图与保留比例、
token entropy、gradient norm、``approx_kl``，以及 log-ratio 的 finite/min/max 指标。
第一次 actor update 在优化器修改权重前，应满足 replay logprob 全部 finite 且 ratio
接近 1。
