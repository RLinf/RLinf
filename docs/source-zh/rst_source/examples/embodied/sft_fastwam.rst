FastWAM 评测与监督微调
========================

.. figure:: https://yuantianyuan01.github.io/FastWAM/static/images/teaser_main.png
   :align: center
   :width: 90%

   Fast-WAM 保留视频协同训练，但评测时无需执行未来视频去噪即可生成动作。

在 LIBERO 或 LIBERO-Plus 上运行发布的
`FastWAM <https://github.com/yuantianyuan01/FastWAM>`__ 模型，并通过 RLinf 的
FSDP SFT 流水线监督微调其世界/动作专家。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      LIBERO · LIBERO-Plus

   .. grid-item-card:: 算法
      :text-align: center

      评测 · SFT

   .. grid-item-card:: 任务
      :text-align: center

      Spatial · Object · Goal · Long

   .. grid-item-card:: 硬件
      :text-align: center

      CUDA GPU · 多 GPU SFT

| **你将完成：** 安装 → 下载检查点与统计信息 → 评测 → 准备 LeRobot 数据和文本 embedding → 启动 SFT。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · CUDA GPU · 能从 Hugging Face 获取 Wan2.2 组件。

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - 环境
     - 任务 / 套件
     - 配置 / 权重
     - 重点
   * - LIBERO
     - Spatial / Object / Goal / Long
     - 四个 ``libero_*_fastwam_eval`` 配置
     - 每次运行评测一个 suite 的全部 500 个初始状态。
   * - LIBERO-Plus
     - Spatial 扰动
     - 同一配置加 ``LIBERO_TYPE=plus``
     - 评测全部或单一扰动类型。
   * - 离线数据
     - LIBERO LeRobot
     - ``libero_sft_fastwam``
     - 对 MoT 专家执行全参数 FSDP SFT。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 字段
     - 说明
   * - 观测
     - 主视角和腕部 RGB 图像，以及 8 维 LIBERO 机器人状态。
   * - 提示词
     - 由 FastWAM 文本编码器处理的 LIBERO 自然语言指令。
   * - 动作
     - 长度为 32、每步 7 维的动作预测；RLinf 每次执行
       ``num_action_chunks`` 步后重新规划。
   * - 训练目标
     - 来自 FastWAM ``training_loss`` 的视频流匹配和动作流匹配 loss。

安装
----

.. include:: _setup_common.rst

评测时，一起安装 FastWAM 和 LIBERO：

.. code-block:: bash

   bash requirements/install.sh embodied --model fastwam --env libero
   source .venv/bin/activate

只做离线 SFT、不需仿真器时：

.. code-block:: bash

   bash requirements/install.sh embodied --model fastwam
   source .venv/bin/activate

安装器会 clone 固定版本的 FastWAM，安装其非 Torch 依赖，并通过 RLinf
平台感知的 Torch 覆盖机制默认选择 Torch 2.7.1（TorchCodec 0.5 所需）。
显式传入的 ``--torch`` 仍然优先。要复用现有 checkout，请在安装前设置
``FASTWAM_PATH=/path/to/FastWAM``。

LIBERO-Plus 请使用 ``--env liberoplus``。还需按
:ref:`zh-liberopro-plus-benchmark` 安装附加 assets。

下载模型
--------

如果要评测发布策略或从发布策略继续微调，可以下载 checkpoint
及匹配的归一化统计信息：

.. code-block:: bash

   huggingface-cli download yuanty/fastwam \
     libero_uncond_2cam224.pt \
     libero_uncond_2cam224_dataset_stats.json \
     --local-dir /your_path_to/fastwam

对于下面介绍的官方 base-model SFT，保持 ``model_path: null``。
SFT 配置会加载官方 Wan2.2 video DiT，并使用官方插值后的 ActionDiT
backbone payload 初始化动作分支。归一化统计信息仍通过
``dataset_stats_path`` 指定：

.. code-block:: yaml

   model_type: fastwam
   model_path: null
   dataset_stats_path: /your_path_to/fastwam/libero_uncond_2cam224_dataset_stats.json

FastWAM 与 RLinf 配置
------------------------

评测 YAML 会继承 ``examples/embodiment/config/model/fastwam.yaml``。
启动评测前，直接修改这个共享 preset 中的两个路径：

.. code-block:: yaml

   model_path: /your_path_to/fastwam/libero_uncond_2cam224.pt  # https://huggingface.co/yuanty/fastwam
   dataset_stats_path: /your_path_to/fastwam/libero_uncond_2cam224_dataset_stats.json

RLinf 通过 OmegaConf 组合 FastWAM 上游 YAML，不会修改 Hydra 全局状态。
两层配置的职责分工如下：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 层级
     - 职责
   * - ``model.fastwam.config_name``
     - 选择上游架构、processor、数据形状、scheduler 和训练 loss 默认值。
       RLinf 默认使用 ``sim_libero``。
   * - ``model.fastwam.overrides``
     - 应用兼容上游的 dot-list 覆盖，例如在使用缓存文本 embedding 的 SFT 中设置
       ``model.load_text_encoder=false``。
   * - RLinf 模型字段
     - ``model_path``、``dataset_stats_path``、动作分块、采样参数与可选的未来视频可视化。
       这些值优先于 FastWAM 的评测默认值。
   * - RLinf FSDP 配置
     - 管理混合精度与梯度 checkpoint。SFT 的模型 preset 保持
       ``precision: bf16``；FSDP2 不额外 cast，由 worker 使用 bf16 autocast，
       与上游 Accelerator 路径一致。

.. note::

   FSDP 通用启动日志可能建议将模型参数设为 fp32，以适配常规模型的优化器策略。
   这条建议不适用于 FastWAM：其官方 SFT 路径就是 bf16 模型参数加 bf16 autocast，
   且 FSDP2 不应再额外进行参数 dtype cast。看到该通用提示不表示 FastWAM 配置错误，
   不需要为消除提示而改为 fp32。

FastWAM 检查点只使用 ``model_path``，不支持 ``checkpoint_path`` 别名。

评测
----

FastWAM 为四个 LIBERO suite 分别提供配置。设置共享模型 YAML 中的 checkpoint
和统计路径后，直接运行需要评测的 suite：

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval
   bash evaluations/run_eval.sh libero libero_object_fastwam_eval
   bash evaluations/run_eval.sh libero libero_goal_fastwam_eval
   bash evaluations/run_eval.sh libero libero_10_fastwam_eval

每个配置覆盖对应 suite 的全部 500 个固定初始状态，并固定使用 GPU ``0-3``
和 20 个环境（每个 worker 5 个）。由于
``ignore_terminations=True`` 会让每条轨迹运行到 ``max_episode_steps``，每个环境
恰好执行 25 个 episode，即 ``20 * 25 = 500`` 个 LIBERO 固定初始状态。这样既避免
启动 500 个 simulator/EGL context，又保留 FastWAM 的批量推理。

不要在 8 卡机器上把 placement 改成 ``env,rollout: all``。非解耦 channel 要求
``total_num_envs`` 能整除 rollout world size，但 500 条轨迹不能被 8 整除；如果没有
padding 或终止槽位屏蔽，8-worker 布局会遗漏状态或让部分槽位提前耗尽。因此这些配置在
4 卡和 8 卡机器上都使用 4 个 worker。

各 suite 的 YAML 已固定官方 FastWAM 使用的 controller 步数，无需再通过命令行覆盖：

.. list-table::
   :header-rows: 1
   :widths: 18 42 20 20

   * - Suite
     - 配置
     - ``max_episode_steps``
     - ``max_steps_per_rollout_epoch``
   * - Spatial
     - ``libero_spatial_fastwam_eval``
     - ``400``
     - ``10000``
   * - Object
     - ``libero_object_fastwam_eval``
     - ``400``
     - ``10000``
   * - Goal
     - ``libero_goal_fastwam_eval``
     - ``400``
     - ``10000``
   * - Long
     - ``libero_10_fastwam_eval``
     - ``700``
     - ``17500``

**LIBERO-Plus：** 通过环境变量选择全部扰动或单一类型，YAML 保持不变：

.. code-block:: bash

   LIBERO_TYPE=plus LIBERO_SUFFIX=all \
     bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval

   LIBERO_TYPE=plus LIBERO_SUFFIX=language \
     bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval

**未来视频可视化：** 动作生成仍为批量执行；可选的未来想象只针对第一个样本，
并受 ``max_video_saves`` 限制。

在评测 YAML 中设置以下字段，然后仍使用上面的单 suite 启动命令：

.. code-block:: yaml

   env:
     eval:
       total_num_envs: 2
       video_cfg:
         save_video: false
   rollout:
     model:
       visualize_future_video: true
       future_video_dir: /your_path_to/future_video_demo

监督微调
--------

RLinf 配置从官方 Wan2.2 base model 和插值后的 ActionDiT backbone 开始训练，
不会从发布的 FastWAM 策略继续训练。资源准备步骤与 FastWAM 上游一致，但本接入
直接在 RLinf YAML 中填写路径，不再使用一组额外的路径环境变量。

下载并解压四套 LIBERO LeRobot 数据：

.. code-block:: bash

   huggingface-cli download yuanty/LIBERO-fastwam \
     --repo-type dataset \
     --local-dir /your_path_to/LIBERO-fastwam

   for archive in /your_path_to/LIBERO-fastwam/*.tar.gz; do
     tar -xzf "$archive" -C /your_path_to/LIBERO-fastwam
   done

下载 Wan 组件并生成 ActionDiT backbone：

.. code-block:: bash

   huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
     --local-dir /your_path_to/checkpoints/Wan-AI/Wan2.2-TI2V-5B
   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
     --include "google/umt5-xxl/*" \
     --local-dir /your_path_to/checkpoints/Wan-AI/Wan2.1-T2V-1.3B

   export DIFFSYNTH_MODEL_BASE_PATH=/your_path_to/checkpoints
   python .venv/FastWAM/scripts/preprocess_action_dit_backbone.py \
     --model-config .venv/FastWAM/configs/model/fastwam.yaml \
     --output /your_path_to/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
     --device cuda \
     --dtype bfloat16

为同样的四个数据目录预计算 T5 text cache：

.. code-block:: bash

   python .venv/FastWAM/scripts/precompute_text_embeds.py \
     task=libero_uncond_2cam224_1e-4 \
     "data.train.dataset_dirs=[/your_path_to/LIBERO-fastwam/libero_spatial_no_noops_lerobot,/your_path_to/LIBERO-fastwam/libero_object_no_noops_lerobot,/your_path_to/LIBERO-fastwam/libero_goal_no_noops_lerobot,/your_path_to/LIBERO-fastwam/libero_10_no_noops_lerobot]" \
     "data.train.text_embedding_cache_dir=/your_path_to/text_embeds_cache/libero" \
     model.redirect_common_files=true

最后，修改 ``examples/sft/config/libero_sft_fastwam.yaml`` 和
``examples/sft/config/model/fastwam.yaml`` 中的占位路径。模型和数据路径旁
已经给出相应的 Hugging Face 地址或准备命令说明。

在仓库根目录启动 SFT：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_fastwam

如果需要防止终端断开影响训练，可自行使用 tmux 等会话管理工具。

官方 FastWAM 与 RLinf 的差异
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

官方流程使用 preprocess_action_dit_backbone.py、缓存的 T5 embedding，
以及 Accelerate + DeepSpeed ZeRO-1；官方 README 的 LIBERO 示例使用 8 张
GPU。RLinf 的差异是：

* 复用上游 RobotVideoDataset、FastWAMProcessor 和 training_loss；
* 使用通用的 train_vla_sft.py / FSDP2 worker，而不是上游 train_zero1.sh；
* 组合上游 sim_libero 配置但不修改 Hydra 全局状态，从官方 Wan2.2
  base 和插值后的 ActionDiT backbone 开始训练，只训练 MoT 和 proprio encoder；
* 模型 preset 保持 precision: bf16，并在 loss 外层启用 bf16 autocast，
  与上游 Accelerator 路径一致。FSDP2 不额外启用 cast，避免第二条混合精度路径；
  RLinf wrapper 仍会将 VAE、text context、action 和 proprio 的直连输入对齐到模型 dtype。

这是有意的集成差异：RLinf 不承诺逐字节复现上游 optimizer 和分布式
launcher，但保持上游模型 loss、数据变换、mask、归一化和文本 cache 格式兼容。
