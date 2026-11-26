YAM π₀.₅ SFT 与评测
===================

合并 :doc:`yam` 采集的示教分片并计算归一化统计量，在 GPU 节点微调 OpenPI Pi0.5，再部署到双臂 YAM 真机评估（或可选切换至 RPent 部署）。

概览
----

本页承接 :doc:`yam` 采集的 LeRobot 数据，完成 ``pi05_yam_joint`` 统计量计算、JAX 到 OpenPI PyTorch 权重转换、监督微调（SFT）以及双节点真机评估。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 机器人
      :text-align: center

      2× YAM · ``flexible_4310`` 夹爪

   .. grid-item-card:: 遥操作与相机
      :text-align: center

      3 台 RealSense RGB（顶 / 左腕 / 右腕）

   .. grid-item-card:: 工作流
      :text-align: center

      合并与统计量 → π₀.₅ SFT → 真机评估

   .. grid-item-card:: 硬件
      :text-align: center

      GPU 节点（rank 0）· YAM 控制节点（rank 1）

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 说明
   * - 观测
     - 顶部、左腕、右腕三路 RealSense RGB 图像，加 14 维状态 ``[左臂关节 (6), 左夹爪, 右臂关节 (6), 右夹爪]``\ （弧度；夹爪范围 ``[0, 1]``）。
   * - 动作
     - 14 维绝对关节与夹爪目标（训练时 12 个关节转为相对当前状态的增量、夹爪保持绝对目标，经归一化后补齐至 32 维）。模型每次预测 30 步，评估执行前 5 步后重新请求预测。
   * - 奖励与提示词
     - ``meta/tasks.jsonl`` 中的任务文本与归一化状态一起编码。评估记录执行指标且每步奖励为零，由操作员结合真机表现与视频判定成败。

硬件与安全准备
--------------

SFT 在 GPU 节点运行。真机评估使用双节点 Ray 集群：GPU 节点为 rank ``0``，YAM 控制节点为 rank ``1``\ （两台机器保持相同 RLinf 版本与依赖，详见 :doc:`../../guides/multi_node`）。按 :doc:`yam` 配置 CAN 接口与相机序列号，并按 `Ray 自管理集群说明 <https://docs.ray.io/en/latest/ray-security/token-auth.html#self-managed-clusters>`_ 在两节点配置相同的认证 token（保护好 token 文件，切勿停止其他用户的集群）。

.. warning::

   评估会连接两路 CAN 设备和全部 3 台相机。若 SDK 的 ``gripper_limits`` 未设置，连接时会自动开合两只夹爪完成标定——请保持夹爪空载并清空开合区域。操作员须全程守在急停旁监督运行。回合结束或停止进程前，必须确认释放电机输出后双臂仍能得到安全支撑；本配置默认由操作员负责复位与停放。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 1. 安装与准备数据

      **在 GPU 节点与 YAM 控制节点安装相同环境**\ （使用同一 RLinf 代码版本）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env yam && source .venv/bin/activate

      **在 YAM 控制节点合并成功分片并计算归一化统计量**\ （将 ``YAM_COLLECTION`` 设为包含 ``lerobot_dataset/`` 的采集运行目录）：

      .. code-block:: bash

         export YAM_COLLECTION="$PWD/logs/20261121-120000-realworld_dual_yam_collect_data_pico"  # 替换为实际采集目录
         export YAM_DATASET="$HOME/yam-data/merged"
         mkdir -p "$(dirname "$YAM_DATASET")"

         # 合并成功回合（可先加 --dry-run 预览）
         python -m toolkits.lerobot.merge_lerobot_datasets \
             --source-dir "$YAM_COLLECTION/lerobot_dataset" --output-dir "$YAM_DATASET" \
             --only-success

         # 查看记录的任务文本并生成 norm_stats.json
         head -n 5 "$YAM_DATASET/meta/tasks.jsonl"
         python -m toolkits.lerobot.calculate_norm_stats \
             --config-name pi05_yam_joint --repo-id "$YAM_DATASET"

      评估提示词请直接使用 ``meta/tasks.jsonl`` 中记录的任务文本，并在训练与评估阶段共用同一份 ``norm_stats.json``。

      **在 GPU 节点准备合并后的数据集：** 使用共享存储时直接指向绝对路径；或通过 SSH 从控制节点复制（将 ``user``、``192.168.1.20`` 与 ``/data/yam_merged`` 替换为控制节点上 ``whoami``、``hostname -I`` 与 ``echo "$YAM_DATASET"`` 的输出）：

      .. code-block:: bash

         export YAM_DATASET="$HOME/yam-data/merged"
         mkdir -p "$(dirname "$YAM_DATASET")"
         scp -r 'user@192.168.1.20:/data/yam_merged' "$YAM_DATASET"  # 替换为控制节点用户名、IP 与源路径

   .. tab-item:: 2. SFT 训练

      **在 GPU 节点下载并转换 Pi0.5 基础权重**\ （共用安装脚本已提供 ``gsutil``）：

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

      **在** ``examples/sft/config/realworld_dual_yam_sft_openpi_pi05.yaml`` **中配置三个绝对路径**\ （或直接用下方 Hydra CLI 参数传入）：

      .. list-table::
         :header-rows: 1
         :widths: 52 48

         * - 字段
           - 值
         * - ``data.train_data_paths``
           - 合并后的 ``YAM_DATASET`` 目录。
         * - ``actor.model.model_path``
           - ``YAM_MODELS/pi05_base_openpi``。
         * - ``actor.model.openpi_data.norm_stats_path``
           - ``YAM_DATASET/norm_stats.json``。

      该配置采用 FP32 参数、BF16 计算与 ``no_shard``\ （见 :doc:`sft_openpi`），每张 actor GPU 均保存完整模型、梯度与优化器状态。单张 48 GB GPU 建议先设为 ``actor.micro_batch_size: 1``、``actor.global_batch_size: 1`` 与 ``actor.model.precision: bf16``\ （按任务验证质量，且全局批量须能被微批量与 actor 进程数的乘积整除）。

      **启动 SFT 并查找已保存的 checkpoint：**

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh realworld_dual_yam_sft_openpi_pi05 \
             data.train_data_paths="$YAM_DATASET" \
             actor.model.model_path="$YAM_MODELS/pi05_base_openpi" \
             actor.model.openpi_data.norm_stats_path="$YAM_DATASET/norm_stats.json"

         # 在 GPU 节点另开终端列出包含 actor/model_state_dict/full_weights.pt 的目录
         find logs -type d -path '*/checkpoints/global_step_*'

   .. tab-item:: 3. 真机评测与部署

      **检查网络连通性并启动双节点 Ray 集群**\ （两台机器均保持 ``.venv`` 已激活，并使用共享认证 token）。在两机用 ``hostname -I`` 确定 ``GPU_IP`` 与 ``CONTROL_IP`` 并互测 ``ping -c 1``；在控制节点启动 Ray 前先按 :doc:`yam` 导出 ``LEFT_CAN_CHANNEL``、``RIGHT_CAN_CHANNEL`` 与 ``CAMERA_SERIALS``：

      .. code-block:: bash

         # GPU 节点（rank 0）
         export RAY_AUTH_MODE=token
         export YAM_TASK_DESCRIPTION='pick the block'
         export RLINF_NODE_RANK=0
         ray start --head --port=6379 --node-ip-address="$GPU_IP"

         # YAM 控制节点（rank 1；保持相同的 YAM_TASK_DESCRIPTION）
         export RAY_AUTH_MODE=token
         export YAM_TASK_DESCRIPTION='pick the block'
         export RLINF_NODE_RANK=1
         ray start --address="$GPU_IP:6379" --node-ip-address="$CONTROL_IP"

      在 GPU 节点运行 ``ray status`` 确认两个存活节点，摆好双臂起始位姿，并复核关节限位、相机画面、急停与工作区域安全。

      **在 GPU 节点启动真机评估**\ （将 ``YAM_CHECKPOINT`` 设为选定的 ``global_step_<N>`` 目录）：

      .. code-block:: bash

         bash evaluations/run_eval.sh realworld_dual_yam_openpi_pi05_eval \
             rollout.model.model_path="$YAM_CHECKPOINT" \
             rollout.model.openpi_data.norm_stats_path="$YAM_DATASET/norm_stats.json"

.. dropdown:: RPent 部署（可选）

   RPent 的 YAM OpenPI 服务沿用相同的 ``pi05_yam_joint`` 约定（14 维目标、顶/左/右三路相机、预测 30 步、30 Hz 下执行 5 步）。启动前先结束原生评估，确保只有一个进程控制机械臂，再在两台 RPent 节点的 RLinf 仓库目录下保持 ``.venv`` 激活并执行：

   .. code-block:: bash

      export RPENT_RLINF_ROOT="$(pwd)"
      git clone --branch main https://github.com/RLinf/RPent.git "$HOME/RPent-yam"
      uv pip install -e "$HOME/RPent-yam[yam]" "numpy<2"

   随后按 `RPent YAM 部署说明 <https://github.com/RLinf/RPent/blob/main/docs/source-zh/rst_source/real_world_robots/yam.rst>`_ 启动 RPC 与控制服务，传入微调 checkpoint、``norm_stats.json`` 以及 ``RPENT_RLINF_ROOT``。

结果与常见问题
--------------

在 GPU 节点运行 ``tensorboard --logdir logs --port 6006`` 监控训练曲线。评估脚本会打印本次运行的日志目录，并在该目录下保存 rollout 视频供人工核验。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 现象
     - 处理方法
   * - 48 GB 单卡 SFT 显存不足（OOM）
     - 在 SFT 配置中先设为 ``actor.micro_batch_size: 1``、``actor.global_batch_size: 1`` 与 ``actor.model.precision: bf16``，用 ``nvidia-smi`` 排查空闲显存，跑通完整更新步后再增大批量。
   * - Ray worker 无法加入 head 节点
     - 检查 ``GPU_IP`` 与 ``CONTROL_IP`` 双向 ``ping -c 1``，确认两机 ``RAY_AUTH_MODE=token`` 配置了完全一致的认证 token，并用 ``ray status`` 复核。
   * - 评估启动时夹爪自动开合
     - SDK 未设置 ``gripper_limits`` 时的自动标定行为；启动前保持夹爪空载并清空行程。
   * - RPent 控制服务与原生评估冲突
     - 启动 RPent 前彻底停止 ``run_eval.sh``，保证同一时间仅有一个进程占用 CAN 接口。
