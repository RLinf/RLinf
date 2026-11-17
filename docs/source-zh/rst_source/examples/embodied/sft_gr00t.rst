GR00T-N1.7 监督微调
===================

使用 LIBERO Spatial 示范数据全参数微调 GR00T-N1.7，并保存可恢复训练的 checkpoint。本示例从 NVIDIA 的 3B 权重开始，用 FSDP 训练视觉编码器、语言模型和 action head。

概览
----

模型读取双相机图像、机器人状态和任务指令，学习预测 16 步动作片段。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      GR00T-N1.7-3B

   .. grid-item-card:: 方法
      :text-align: center

      全参数 SFT

   .. grid-item-card:: 数据
      :text-align: center

      Physical Intelligence LIBERO

   .. grid-item-card:: 硬件
      :text-align: center

      8 张 NVIDIA GPU · FSDP

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 任务套件
     - 示范数据
     - 配置
   * - LIBERO Spatial：10 个任务
     - 432 条轨迹，52,970 个时间步，采样频率 10 Hz
     - ``libero_sft_gr00t_n1d7``

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 字段
     - 训练数据
   * - 观测
     - 主相机与腕部相机 RGB 图像；8 维末端与夹爪状态
   * - 动作
     - 16 步、每步 7 维的笛卡尔位移、旋转增量和夹爪控制
   * - 指令
     - 示范轨迹中的任务描述

安装
----

.. include:: _setup_common.rst

安装并激活 GR00T-N1.7 环境：

.. code-block:: bash

   bash requirements/install.sh embodied --model gr00t_n1d7 --env maniskill_libero
   source .venv/bin/activate

也可以激活此前按上述模型和环境选项安装的已有环境。后续命令均在所选环境、RLinf 仓库根目录执行。示例配置使用单机八卡；调整 GPU 分配时，同时调整 batch size。

下载模型与数据
--------------

先在 Hugging Face 接受 `Cosmos-Reason2-2B <https://huggingface.co/nvidia/Cosmos-Reason2-2B>`_ 的访问条款，再下载 GR00T 权重以及 backbone 配置和 tokenizer：

.. code-block:: bash

   hf auth login
   hf download nvidia/GR00T-N1.7-3B --local-dir checkpoints/GR00T-N1.7-3B
   hf download nvidia/Cosmos-Reason2-2B \
     --include '*.json' '*.txt' --local-dir checkpoints/Cosmos-Reason2-2B

GR00T checkpoint 包含 backbone 权重，Cosmos 目录用于提供加载所需的配置和 tokenizer。

下载 `Physical Intelligence LIBERO 数据集 <https://huggingface.co/datasets/physical-intelligence/libero>`_：

.. code-block:: bash

   hf download physical-intelligence/libero \
     --repo-type dataset --local-dir datasets/physical-intelligence-libero

完整数据集包含 40 个任务、1,693 条轨迹，双相机图像直接存放在 Parquet 文件中。保留 ``meta/`` 和 ``data/`` 目录；已有完整下载可直接在下方填写路径。示例用 ``data.task_names`` 筛选 10 个 Spatial 任务，将夹爪动作转换为 GR00T 约定，再计算所选轨迹的归一化统计。

运行
----

编辑 ``examples/sft/config/libero_sft_gr00t_n1d7.yaml``，设置数据集、模型权重和 backbone 配置的路径，再设置 GPU 分配、batch size 和训练预算：

.. code-block:: yaml

   cluster:
     component_placement:
       actor: 0-7
   data:
     train_data_paths:
       - /absolute/path/to/datasets/physical-intelligence-libero
     format: physical_libero
     statistics_path: ./data_statistics/libero_spatial.json
   runner:
     max_steps: 20000
     save_interval: 1000
   actor:
     micro_batch_size: 1
     global_batch_size: 32
     model:
       model_path: /absolute/path/to/checkpoints/GR00T-N1.7-3B
       backbone_model_path: /absolute/path/to/checkpoints/Cosmos-Reason2-2B
     optim:
       total_training_steps: 20000

保留配置中已有的 Spatial 任务列表。``data.statistics_path`` 指定缓存目录和文件名前缀，每个数据集和 episode 选择分别生成带哈希后缀的缓存文件。归一化统计也会随 checkpoint 保存。

全局 batch size 必须能被 GPU 数与 microbatch size 的乘积整除。保留 ``runner.max_epochs: -1``，训练持续采样，到 ``runner.max_steps`` 指定的更新次数后停止。``actor.optim.total_training_steps`` 决定学习率计划长度。20,000 次更新是示例预算，不代表已经测得的收敛目标。

使用修改后的配置启动单机训练。指定本地 Ray 实例，避免连入机器上的其他集群：

.. code-block:: bash

   export RLINF_NODE_RANK=0
   export RAY_ADDRESS=local
   bash examples/sft/run_vla_sft.sh libero_sft_gr00t_n1d7

启动脚本将控制台日志和 TensorBoard 事件写入 ``logs/<timestamp>-libero_sft_gr00t_n1d7/``。训练过程中查看 ``train/loss`` 和全局更新次数；checkpoint 位于该运行目录下的 ``libero_sft_gr00t_n1d7/checkpoints/global_step_N/``。

恢复训练
~~~~~~~~

在原 YAML 中，将 ``runner.resume_dir`` 设为已保存的 ``global_step_N`` 目录的绝对路径，再执行上面的启动命令。保持数据集、GPU 拓扑、batch size 和 dataloader worker 数不变。``runner.max_steps`` 表示包含已完成更新在内的总更新次数。

短程检查保存与恢复流程时，首次启动设置 ``runner.max_steps: 10``、``runner.save_interval: 10``、``actor.optim.lr_warmup_steps: 0`` 和 ``actor.optim.total_training_steps: 20``。运行结束后，把 ``runner.max_steps`` 改为 20，并将 ``runner.resume_dir`` 指向刚保存的 ``global_step_10``，再启动一次。第二次运行应从第 10 次更新继续，保存 ``global_step_20``。checkpoint 恢复详见 :doc:`../../guides/resume`。

.. note::

   dataloader 通过重放数据流恢复读取位置，随机图像增强的结果不保证与恢复前完全一致。

可视化与结果
------------

启动 TensorBoard 查看训练记录：

.. code-block:: bash

   tensorboard --logdir logs

用 ``train/loss`` 观察动作拟合情况，并确认保存的更新目录与预算一致。checkpoint 包含模型、优化器、训练归一化统计、processor 设置和 dataloader 恢复状态。指标定义见 :doc:`../../reference/metrics`。

准备训练模型并评测
~~~~~~~~~~~~~~~~~~

将训练保存的模型配置和 processor 放入评测模型目录。先把 ``SFT_ACTOR`` 设为待评测更新对应的 ``actor/`` 目录：

.. code-block:: bash

   BASE_MODEL="$(realpath checkpoints/GR00T-N1.7-3B)"
   SFT_ACTOR=/absolute/path/to/global_step_N/actor
   EVAL_MODEL="$(realpath -m checkpoints/GR00T-N1.7-SFT)"
   mkdir -p "$EVAL_MODEL"
   ln -s "$BASE_MODEL"/model*.safetensors "$EVAL_MODEL/"
   cp "$BASE_MODEL/model.safetensors.index.json" "$EVAL_MODEL/"
   cp "$SFT_ACTOR"/{config.json,processor_config.json,statistics.json,embodiment_id.json} "$EVAL_MODEL/"

评测目录用基座权重的符号链接构建模型，并使用 SFT 保存的 processor。请保留基座目录。下面的评测命令从同一个 SFT checkpoint 加载训练权重。

评测先加载 Cosmos backbone，再恢复 GR00T 权重。将完整的 Cosmos checkpoint 下载到 ``checkpoints/Cosmos-Reason2-2B``：

.. code-block:: bash

   hf download nvidia/Cosmos-Reason2-2B --local-dir checkpoints/Cosmos-Reason2-2B

复用现有 LIBERO Spatial 配置，只启动评测。将完整模型配置传给 rollout，并关闭 value head：

.. code-block:: bash

   bash evaluations/run_eval.sh libero_spatial_ppo_gr00t_n1d7 \
     runner.only_eval=True \
     'rollout.model=${actor.model}' \
     actor.model.model_path="$EVAL_MODEL" \
     actor.model.backbone_model_path="$(realpath checkpoints/Cosmos-Reason2-2B)" \
     actor.model.add_value_head=False \
     runner.ckpt_path="$SFT_ACTOR/model_state_dict/full_weights.pt" \
     env.eval.ignore_terminations=False

``rollout.model.model_path`` 选择训练保存的 processor，``runner.ckpt_path`` 指向配套的训练权重。命令沿用原配置的 GPU 分配和 rollout 预算，按评测协议调整这两项。评测设置见 :doc:`../../evaluations/index`，LIBERO 环境准备见 :doc:`gr00t`。
