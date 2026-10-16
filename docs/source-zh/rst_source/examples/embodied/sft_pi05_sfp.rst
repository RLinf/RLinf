Streaming Flow Policy 监督微调
==========================================

.. figure:: https://raw.githubusercontent.com/gen-robot/StreamingVLA/main/assets/final_tea_01.png
   :align: center
   :width: 70%

   Streaming Flow Policy，来自 `StreamingVLA <https://github.com/gen-robot/StreamingVLA>`_ 项目。

使用 **Streaming Flow Policy（SFP）** 目标在 LIBERO 上微调 π₀.₅ 权重。SFP 让 action expert 去拟合一条动作\ *轨迹*\ ，而不是对整个 action chunk 去噪，调用方因而可以一边执行一边生成后续动作。改变的只有训练目标：网络结构、参数形状和 checkpoint 布局都仍是 π₀.₅ 的，所以训练可以直接从官方 ``pi05_libero`` 权重开始，产出的权重也能按原样载回。

概览
----------------------------------------

在 LIBERO 示例数据上用 SFP 目标对 π₀.₅ 做单机全量微调。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀.₅（``openpi``）

   .. grid-item-card:: 方法
      :text-align: center

      Streaming Flow Policy

   .. grid-item-card:: 数据
      :text-align: center

      LIBERO · LeRobot format

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 4 GPU

| **你将完成：** 安装 OpenPI → 转换 LIBERO 数据 → 计算归一化统计 → 转换 π₀.₅ 权重 → 启动 ``run_vla_sft.sh`` → 观察训练损失。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 足够存放 LIBERO RLDS 原始数据及其 LeRobot 副本的磁盘空间。

.. warning::

   SFP 评测走轨迹采样器（``task: eval`` 且 ``use_sfp``），不是普通 π₀.₅ 的 flow-matching Euler。RL、DAgger、DSRL 仍然拒绝 ``use_sfp``。用 ``use_sfp: False`` 加载 SFP 权重会静默积错场，请使用 SFP 评测配方。

SFP 与 flow matching 的区别
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

π₀.₅ 一次性对整个 action chunk 去噪。它的 flow 时间是一个去噪坐标，和机器人的物理时间无关，因此必须走完全部去噪步，chunk 里才有可执行的动作。

SFP 则把 chunk 累加成一条轨迹，并把 flow 时间映射到这条轨迹上：``t=0`` 是机器人当前已经到达的动作状态，``t=1`` 是 chunk 末尾，action expert 用一个 suffix token 回归 ``t`` 处的轨迹速度。这样训练出的 policy 可以从当前位置继续往前积分。

配置里有两处设置由此而来：``actor.model.openpi.use_sfp`` 选择训练目标，``sfp_sigma`` 和 ``sfp_noise_decay`` 分别控制轨迹起点注入的噪声大小和它沿轨迹衰减的快慢。

准备数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SFP 比 flow matching 多需要一个字段 ``action_states``，即每一帧之前所有动作的累积值，也就是该帧轨迹的起点。下载公开的 `OpenVLA modified LIBERO RLDS 数据集 <https://huggingface.co/datasets/openvla/modified_libero_rlds>`_ 并转换：

.. code:: bash

   hf download openvla/modified_libero_rlds \
       --repo-type dataset \
       --local-dir /data/libero-rlds

   export HF_LEROBOT_HOME=/data/lerobot

   python toolkits/lerobot/convert_libero_data_to_lerobot.py \
       --data-dir /data/libero-rlds \
       --repo-name local/libero_sfp

转换脚本把四个 LIBERO 任务套件合并写入 ``$HF_LEROBOT_HOME/local/libero_sfp``，并为每一帧记录 ``action_states``。用 ``--raw-dataset-names`` 可以只转换其中几个套件，``--overwrite`` 用于覆盖该路径下已有的数据集。

.. note::

   ``--repo-name`` 是相对于 ``HF_LEROBOT_HOME`` 解析的，传绝对路径会被拒绝。计算统计量和训练时要保持 ``HF_LEROBOT_HOME`` 取值不变，否则加载器找不到数据集。

归一化统计量
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

转换后的数据集中动作仍是原始量纲，因此下一步要统计出模型训练所用的尺度：

.. code:: bash

   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_libero_sfp \
       --repo-id local/libero_sfp \
       --output-dir /data/assets/libero_sfp

命令生成 ``/data/assets/libero_sfp/norm_stats.json``，包含 ``state``、``actions`` 和 ``action_states`` 三项统计量。``action_states`` 是 ``actions`` 的副本，因为 SFP 把这两项加在同一套尺度上。

同样的要求决定了 SFP 采用除以 ``max(|q01|, |q99|)`` 的纯缩放，而不用 OpenPI 在别处采用的仿射 quantile 映射：纯缩放与累加可交换，先归一化再累加与先累加再归一化结果一致；仿射映射会给每个值加上偏移，这个等式就不成立了。

安装
----------------------------------------

.. include:: _setup_common.rst

SFP 运行在 OpenPI 环境中，不需要单独的安装目标。

**方式一：使用 Docker 镜像** —— 镜像标签 ``agentic-rlinf0.4-maniskill_libero``：

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.4-maniskill_libero
      # 国内镜像加速：infinigence-ai-registry.cn-beijing.cr.aliyuncs.com/rlinf/rlinf:agentic-rlinf0.4-maniskill_libero

   # 进入容器后，切换到 OpenPI 虚拟环境：
   source switch_env openpi

**方式二：自建环境** —— 安装套件 ``--env maniskill_libero``：

.. code:: bash

   # 为提高国内依赖安装速度，可以添加 --use-mirror。
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

下载模型
----------------------------------------

训练从官方 ``pi05_libero`` 权重开始，需要转换成 RLinf 使用的 PyTorch 布局，也就是一个包含 ``model.safetensors`` 的目录。在 `OpenPI <https://github.com/Physical-Intelligence/openpi>`_ 仓库中下载并转换：

.. code:: bash

   python -c "from openpi.shared import download; download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero')"

   python examples/convert_jax_model_to_pytorch.py \
       --checkpoint_dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero" \
       --config_name pi05_libero \
       --output_path /data/checkpoints/pi05_libero_pytorch

运行
----------------------------------------

**1. 配置**

这个示例拆成两个文件：不含路径的模型模板 ``examples/sft/config/model/pi0_5_sfp.yaml``，以及实验配置 ``examples/sft/config/libero_sft_pi05_sfp.yaml``。把实验配置指向上面生成的三个路径：

.. code:: yaml

   data:
     train_data_paths: local/libero_sfp

   actor:
     model:
       model_path: /data/checkpoints/pi05_libero_pytorch
       openpi_data:
         norm_stats_path: /data/assets/libero_sfp/norm_stats.json

模型模板同时决定训练目标和数据管线：

.. code:: yaml

   openpi:
     task: sft
     config_name: "pi05_libero_sfp"
     use_sfp: True
     sfp_sigma: 0.16
     sfp_noise_decay: 4.0

``config_name`` 起两个作用：既选中负责 repack 和补齐 ``action_states`` 的 OpenPI 数据配置，也通过名字匹配到对应的 SFT 数据加载器。

配套配置在四张卡上做全量微调，精度沿用 OpenPI 默认设置：Gemma 和 SigLIP 使用 bf16，action head 保持 fp32。SFP 的轨迹目标始终在 fp32 下构建。

.. warning::

   ``global_batch_size`` 必须是 ``micro_batch_size`` 乘以 world size 的整数倍，否则 actor 在启动时会拒绝该配置。

**2. 启动**

.. code:: bash

   source .venv/bin/activate
   export HF_LEROBOT_HOME=/data/lerobot
   bash examples/sft/run_vla_sft.sh libero_sft_pi05_sfp

该命令做三件事：1. 组合 Hydra 配置并启动 Ray 集群；2. 构建带 SFP 目标的 π₀.₅ 模型并加载转换后的权重；3. 让 LIBERO 数据流经 SFP 管线并开始训练。

可视化与结果
----------------------------------------

关注 **训练损失** 即可确认模型是否在拟合示例数据。各项指标的含义见 :doc:`训练指标 <../../reference/metrics>`。

.. code-block:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs

``run_vla_sft.sh`` 会把 ``runner.logger.log_path`` 覆盖成带时间戳的目录，因此 checkpoint 位于 ``logs/<时间戳>-libero_sft_pi05_sfp/checkpoints/global_step_<N>/``。需要继续训练时，把 ``runner.resume_dir`` 指向其中一个目录再启动即可。

评测 checkpoint
----------------------------------------

SFP 评测走轨迹采样器（``task: eval`` 且 ``use_sfp``），不是普通 π₀.₅ 的 flow-matching Euler。把 ``evaluations/libero/libero_spatial_openpi_pi05_sfp_eval.yaml`` 指向 SFT checkpoint，以及训练时同一份 ``norm_stats.json``：

.. code:: bash

   bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_sfp_eval \
     rollout.model.model_path=/path/to/pi05_libero_sfp \
     rollout.model.openpi_data.norm_stats_path=/data/assets/libero_sfp/norm_stats.json

RTC 与 SFP 不兼容：RTC 积分的是 flow-matching 速度场。
