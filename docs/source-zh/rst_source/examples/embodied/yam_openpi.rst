YAM Pi0.5 训练与评估
=======================

按照 :doc:`yam` 采集示教数据，计算归一化统计量，微调 OpenPI Pi0.5，然后在双臂 YAM 上评估模型检查点。

观测与动作
----------

``pi05_yam_joint`` 使用 14 维状态和绝对位置动作，顺序为 ``[左臂关节 (6), 左夹爪, 右臂关节 (6), 右夹爪]``。关节位置的单位为弧度，夹爪归一化到 ``[0, 1]``。三路 RGB 图像依次来自顶部、左腕和右腕相机。

训练时，12 个关节动作转换为相对当前状态的增量，夹爪保持绝对目标值。共用 OpenPI 变换先对 14 维向量归一化，再补齐到模型的 32 维输入宽度。Pi0.5 还会将归一化后的 14 维状态与任务提示词一起编码为 token。模型预测 30 步动作；评估每次执行前 5 步，然后重新请求预测。

安装
----

GPU 训练节点和 YAM 控制节点使用相同的 RLinf 版本，在同一环境中安装模型和机器人依赖：

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   bash requirements/install.sh embodied --model openpi --env yam
   source .venv/bin/activate

按照 :doc:`yam` 配置 CAN 接口并获取三路相机序列号。

数据集与归一化
--------------

在 YAM 控制节点上，按照 :doc:`yam` 采集成功的示教。采集器会打印包含 ``lerobot_dataset/`` 的运行目录，将 ``YAM_COLLECTION`` 设置为该目录。下面的命令将其中的分片合并为新的本地数据集：

.. code-block:: bash

   export YAM_COLLECTION=/absolute/path/to/collection/run
   export YAM_DATASET="$HOME/yam-data/merged"
   mkdir -p "$(dirname "$YAM_DATASET")"
   python toolkits/lerobot/merge_lerobot_datasets.py \
       --source-dir "$YAM_COLLECTION/lerobot_dataset" --output-dir "$YAM_DATASET" \
       --only-success --dry-run
   python toolkits/lerobot/merge_lerobot_datasets.py \
       --source-dir "$YAM_COLLECTION/lerobot_dataset" --output-dir "$YAM_DATASET" \
       --only-success
   head -n 5 "$YAM_DATASET/meta/tasks.jsonl"
   PYTHONPATH=$(pwd) python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_yam_joint --repo-id "$YAM_DATASET"
   test -s "$YAM_DATASET/norm_stats.json"

评估提示词使用 ``meta/tasks.jsonl`` 中记录的任务文本。将统计量保留在该数据集中，训练与评估使用同一份统计量文件。

转换模型和训练前，先在 GPU 节点准备数据集。使用共享存储时，将 ``YAM_DATASET`` 设置为该节点上数据集的绝对路径。通过 SSH 复制时，在控制节点运行 ``whoami`` 获取用户名、``hostname -I`` 获取可达地址、``echo "$YAM_DATASET"`` 获取源目录，并用这些值替换下面的 SSH 用户名、控制节点地址和源路径。在 GPU 节点的 RLinf 仓库目录中，将整个合并数据集及统计量复制到新目录：

.. code-block:: bash

   source .venv/bin/activate
   export YAM_DATASET="$HOME/yam-data/merged"
   mkdir -p "$(dirname "$YAM_DATASET")"
   test ! -e "$YAM_DATASET" && \
       scp -r '<ssh-user>@<controller-ip>:/absolute/path/to/merged' "$YAM_DATASET" && \
       test -s "$YAM_DATASET/norm_stats.json"

复制成功后，在 GPU 节点继续后续步骤。目标目录检查会保护已有数据集；若目录已存在，先为 ``YAM_DATASET`` 选择新的路径。

预训练模型
----------

共用安装流程提供 ``gsutil``。在 GPU 节点下载官方 Pi0.5 基础模型，并转换为 RLinf 使用的 OpenPI PyTorch 格式：

.. code-block:: bash

   export YAM_MODELS="$HOME/yam-models"
   mkdir -p "$YAM_MODELS/jax"
   gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base "$YAM_MODELS/jax/"
   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
       --input-model "$YAM_MODELS/jax/pi05_base" \
       --input-norm-stats "$YAM_DATASET/norm_stats.json" \
       --output-model "$YAM_MODELS/pi05_base_openpi" \
       --output-norm-stats "$YAM_MODELS/pi05_base_openpi/yam/norm_stats.json" \
       --action-dim 32 --action-horizon 30 --max-token-len 200
   test -s "$YAM_MODELS/pi05_base_openpi/model.safetensors"

监督微调
--------

在 GPU 节点编辑 ``examples/sft/config/realworld_dual_yam_sft_openpi_pi05.yaml``，将以下字段替换为前面得到的绝对路径：

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - 字段
     - 值
   * - ``data.train_data_paths``
     - 合并后的 ``YAM_DATASET`` 目录。
   * - ``actor.model.model_path``
     - ``YAM_MODELS/pi05_base_openpi``。
   * - ``actor.model.openpi_data.norm_stats_path``
     - ``YAM_DATASET/norm_stats.json``。

此配置使用 FP32 参数、BF16 计算和 ``no_shard``，详见 :doc:`sft_openpi`。每张 actor GPU 都保存完整的模型、梯度和优化器状态；增加 GPU 提高吞吐量，而不是减少这部分单卡显存占用。

``cluster.component_placement.actor: all`` 使用已分配的 GPU。运行 ``nvidia-smi`` 查看显存和活跃进程。单张 48 GB GPU 可先在 YAML 中设置 ``actor.micro_batch_size: 1``、``actor.global_batch_size: 1`` 和 ``actor.model.precision: bf16``；这会改变默认的 FP32 参数及优化器精度，应按任务验证训练质量。确认完整的优化器更新步骤能运行后，再增加批量。全局批量必须能被单卡微批量与 actor 工作进程数的乘积整除。

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh realworld_dual_yam_sft_openpi_pi05

启动脚本会打印本次运行的日志目录。在 GPU 节点的 RLinf 仓库目录中另开一个终端，查看训练曲线：

.. code-block:: bash

   source .venv/bin/activate
   tensorboard --logdir logs --port 6006

检查点保存后，在同一仓库目录的另一个终端中列出检查点目录：

.. code-block:: bash

   find logs -type d -path '*/checkpoints/global_step_*'

从 ``find`` 输出中选择一个已保存的 ``checkpoints/global_step_<N>`` 目录，其中的 ``actor/model_state_dict/full_weights.pt`` 可由评估加载器直接读取。``<N>`` 是已保存的训练步数。

评估
----

.. warning::

   评估会连接两路 CAN 设备和三路相机。若 SDK 的 ``gripper_limits`` 未设置，连接时会自动开合两只夹爪进行标定，应保持夹爪空载并清空开合区域。操作员应守在急停旁并监督整个运行过程。回合结束或停止进程前，必须确认释放电机输出后双臂仍能得到安全支撑。此配置由操作员负责复位和停放。

GPU 节点使用 rank ``0``，YAM 控制节点使用 rank ``1``。按照 :doc:`../../guides/multi_node`，先在两台机器上安装相同的依赖，再启动 Ray。运行 ``ray status`` 检查已有集群；使用分配给本次运行的机器，不要停止其他用户的集群。

在两台机器上运行 ``hostname -I``，选择彼此可达的地址，分别导出为 ``GPU_IP`` 和 ``CONTROL_IP``。在 GPU 节点运行 ``ping -c 1 "$CONTROL_IP"``，在控制节点运行 ``ping -c 1 "$GPU_IP"``，检查彼此可达。将 ``YAM_TASK_DESCRIPTION`` 设置为数据集中的任务文本。在控制节点启动 Ray 前，导出 :doc:`yam` 中已确认的 CAN 通道和顶部、左腕、右腕相机序列号。

按照 `Ray 自管理集群说明 <https://docs.ray.io/en/latest/ray-security/token-auth.html#self-managed-clusters>`_，在两个节点上配置相同的认证 token。限制 token 文件的访问权限，使用分配给本次运行的集群凭据，不要覆盖其他运行的凭据。

GPU 节点：

.. code-block:: bash

   source .venv/bin/activate
   export RAY_AUTH_MODE=token
   export YAM_TASK_DESCRIPTION='pick the block'
   export RLINF_NODE_RANK=0
   ray start --head --port=6379 --node-ip-address="$GPU_IP"

控制节点使用同一任务文本：

.. code-block:: bash

   source .venv/bin/activate
   export RAY_AUTH_MODE=token
   export YAM_TASK_DESCRIPTION='pick the block'
   export RLINF_NODE_RANK=1
   ray start --address="$GPU_IP:6379" --node-ip-address="$CONTROL_IP"

GPU 节点上的 ``ray status`` 应列出两个存活节点。启用 policy 控制前，按照采集指南确认关节限位、相机画面、急停和工作区域安全。操作员将双臂摆到任务起始位置；默认由操作员负责复位和停放。

将 ``YAM_CHECKPOINT`` 设置为前面选定的 ``global_step_<N>`` 目录，用 ``test -s "$YAM_CHECKPOINT/actor/model_state_dict/full_weights.pt"`` 检查权重文件，然后在 GPU 节点启动评估：

.. code-block:: bash

   bash evaluations/run_eval.sh realworld realworld_dual_yam_openpi_pi05_eval \
       rollout.model.model_path="$YAM_CHECKPOINT" \
       rollout.model.openpi_data.norm_stats_path="$YAM_DATASET/norm_stats.json"

启动脚本会打印评估日志目录，视频也保存在该次运行目录下。YAM 环境记录执行指标，每步奖励为零；操作员根据机器人执行情况和视频判断任务是否完成。

RPent 部署
----------

RPent 的 YAM OpenPI 接口采用相同的 ``pi05_yam_joint`` 约定：14 维动作目标（12 个关节位置和 2 个夹爪目标），顶部、左腕、右腕相机，预测 30 步，每次执行 5 步，控制频率 30 Hz。在 RPent 两个节点上，从前面使用的 RLinf 仓库目录执行：

.. code-block:: bash

   source .venv/bin/activate
   export RPENT_RLINF_ROOT="$(pwd)"
   git clone --branch main https://github.com/RLinf/RPent.git "$HOME/RPent-yam"
   uv pip install -e "$HOME/RPent-yam[yam]" "numpy<2"

这会在现有的模型及机器人环境中安装 RPent 的 YAM 可选依赖，并保留已有依赖约束。按照 `RPent YAM 部署说明 <https://github.com/RLinf/RPent/blob/main/docs/source-zh/rst_source/real_world_robots/yam.rst>`_ 的 RPC 和控制服务启动章节，使用微调检查点及其数据集统计量；其中的 RLinf 路径设置采用前面导出的 ``RPENT_RLINF_ROOT``。启动 RPent 控制服务前，先结束原生评估，确保只有一个进程控制双臂。
