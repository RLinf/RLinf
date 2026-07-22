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

      LIBERO Spatial

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
     - LIBERO-Spatial
     - ``libero_spatial_fastwam_eval``
     - 批量、仅动作的评测。
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

下载已发布的 LIBERO 检查点和与之匹配的归一化统计信息：

.. code-block:: bash

   hf download yuanty/fastwam \
     libero_uncond_2cam224.pt \
     libero_uncond_2cam224_dataset_stats.json \
     --local-dir /workspace/checkpoints/fastwam

在 ``examples/embodiment/config/model/fastwam.yaml`` 和
``examples/sft/config/model/fastwam.yaml`` 中设置这两个路径：

.. code-block:: yaml

   model_type: fastwam
   model_path: /workspace/checkpoints/fastwam/libero_uncond_2cam224.pt
   dataset_stats_path: /workspace/checkpoints/fastwam/libero_uncond_2cam224_dataset_stats.json

FastWAM 与 RLinf 配置
--------------------

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
       ``precision: fp32``；FSDP 对 forward/backward 应用 bf16 精度。

FastWAM 检查点只使用 ``model_path``，不支持 ``checkpoint_path`` 别名。

评测
----

单一的 ``libero_spatial_fastwam_eval.yaml`` 取代了原先分开的小规模、
大规模、LIBERO-Plus、仅语言扰动和未来视频 YAML。

**标准 LIBERO smoke 评测：**

.. code-block:: bash

   MUJOCO_GL=egl bash evaluations/run_eval.sh \
     libero libero_spatial_fastwam_eval

**更大规模评测：** 通过 8 个可复用环境进程运行 80 条轨迹，并关闭录像：

.. code-block:: bash

   MUJOCO_GL=egl bash evaluations/run_eval.sh \
     libero libero_spatial_fastwam_eval \
     env.eval.total_num_envs=8 \
     env.eval.max_steps_per_rollout_epoch=2400 \
     env.eval.video_cfg.save_video=false

**LIBERO-Plus：** 通过环境变量选择全部扰动或单一类型，YAML 保持不变：

.. code-block:: bash

   LIBERO_TYPE=plus LIBERO_SUFFIX=all MUJOCO_GL=egl \
     bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval

   LIBERO_TYPE=plus LIBERO_SUFFIX=language MUJOCO_GL=egl \
     bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval \
     env.eval.total_num_envs=8 env.eval.video_cfg.save_video=false

**未来视频可视化：** 动作生成仍为批量执行；可选的未来想象只针对第一个样本，
并受 ``max_video_saves`` 限制。

.. code-block:: bash

   MUJOCO_GL=egl bash evaluations/run_eval.sh \
     libero libero_spatial_fastwam_eval \
     env.eval.total_num_envs=2 \
     env.eval.video_cfg.save_video=false \
     rollout.model.visualize_future_video=true \
     rollout.model.future_video_dir=/workspace/future_video_demo

监督微调
--------

下载 `FastWAM LIBERO 数据集
<https://huggingface.co/datasets/yuanty/LIBERO-fastwam>`__，并使用上游脚本预计算 T5 文本 embedding：

.. code-block:: bash

   python "$FASTWAM_PATH/scripts/precompute_text_embeds.py" \
     task=libero_uncond_2cam224_1e-4 \
     'data.train.dataset_dirs=[/path/to/libero_spatial_no_noops_lerobot]' \
     data.train.text_embedding_cache_dir=/workspace/data/text_embeds_cache/libero \
     model.redirect_common_files=false

更新 ``examples/sft/config/libero_sft_fastwam.yaml`` 中的
``data.train_data_paths`` 和 ``data.text_embedding_cache_dir``，然后启动：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_fastwam

FastWAM 的 MoT 会直接访问视频与动作 Transformer block，因此示例有意使用整模型
FSDP2 包装。完整可训练 MoT 不适合普通单 GPU SFT；请使用多 GPU，并按显存调整 batch size。
