基于 GenieSim 的强化学习训练
========================================

.. |pi| replace:: π₀.₅

.. figure:: /_static/images/genie_sim_system_overview.webp
   :align: center
   :width: 100%
   :alt: Genie Sim 平台与功能概览。

   Genie Sim 系统概览（图片来源：`GenieSim <https://github.com/AgibotTech/genie_sim>`__）。

本项目是 `GenieSim <https://github.com/AgibotTech/genie_sim>`__ 官方基于 RLinf 的后训练项目，支持视觉语言动作（VLA）模型的训练与评测。Genie Sim 是智元机器人推出的一站式具身智能仿真平台，集成场景重建、资产生成、数据采集、自动评测、模型后训练功能，为具身智能模型的训练、验证与迭代提供统一的仿真工具链。任务涵盖指令理解、空间理解和操作技能，通过多样化的物体、场景与指令组合，支持模型泛化能力的研究。项目支持使用 PPO 和 GRPO 微调 VLA 模型，提供单任务与多任务训练流程，提升双臂操作能力。

GenieSim 仿真与 RLinf 训练分别运行在两个 GPU 节点上，其中仿真节点使用 NVIDIA RTX 系列显卡。

概览
----------------------------------------

通过 PPO 和 GRPO 提升自然语言引导的双臂操作能力。相较于 SFT 检查点，10 任务 PPO 将按颜色抓取方块的成功率从 71% 提升至 95%，按尺寸抓取方块从 60% 提升至 92%。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      VLA 模型

   .. grid-item-card:: 算法
      :text-align: center

      PPO、GRPO

   .. grid-item-card:: 任务
      :text-align: center

      指令理解 · 空间理解 · 操作技能

   .. grid-item-card:: 硬件
      :text-align: center

      两个 GPU 节点

| **你将完成：** 安装 → 启动 GenieSim 服务 → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 可连通的 GenieSim 服务 · 一个 VLA 检查点。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

25 个任务覆盖指令理解、空间理解和操作技能，可通过下文的单任务或多任务配方开展训练。完整任务标识见文末 :ref:`任务附录 <geniesim-training-tasks>`。

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - 类别
     - 任务数
     - 能力
   * - 指令理解
     - 10
     - 根据颜色、尺寸、形状、数字、类别和语言描述选择物体。
   * - 空间理解
     - 8
     - 根据空间关系定位、摆放、排序和堆叠物体。
   * - 操作技能
     - 7
     - 开门、操作容器、整理货架和清理桌面。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下文 G2 实验使用的 OpenPI |pi| 配置结合三路相机图像、双臂与夹爪状态，以及自然语言任务指令。:ref:`任务附录 <geniesim-training-tasks>` 按能力类别列出 25 个任务。

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 规格
   * - 观测
     - 三路 640×480 RGB 图像。输入模型前保持宽高比缩放并填充至 224×224。本体状态使用 14 个臂部关节位置和 2 个夹爪开合角度。
   * - 动作
     - 16 维连续动作：14 个臂部关节位置目标与 2 个夹爪目标，每次执行 10 步动作块，增量预测。
   * - 奖励
     - 使用稀疏奖励，成功时奖励为 1，否则为 0。
   * - 任务指令
     - 使用 ``task_descriptions`` 中的完整自然语言任务指令。

模型在 32 维输入与输出空间中使用 16 个有效状态和动作维度。仿真器状态中的额外通道会被掩码，模型输出则截取为机器人控制所需的 16 维。状态与动作的归一化使用检查点中的 ``assets/norm_stats.json``。

安装
----------------------------------------

在 RLinf 主机上执行以下源码准备命令，命令在容器外运行。

.. include:: _setup_common.rst
   :end-before: 然后，使用下列两种方式之一准备依赖

在 RLinf 仓库根目录，将 GenieSim 克隆到同级目录，并设置客户端包路径：

.. code:: bash

   git clone https://github.com/AgibotTech/genie_sim.git ../genie_sim
   export GENIESIM_CLIENT_PATH="$(realpath ../genie_sim/source/rlinf_geniesim)"

以下两种安装方式任选其一，都会从这份源码安装 ``rlinf-geniesim-client``。前置依赖和 GPU 驱动要求见 :doc:`安装 </rst_source/start/installation>`。

**自定义环境**

在 RLinf 主机的 RLinf 仓库根目录安装 OpenPI 与 GenieSim 客户端，并激活环境：

.. code:: bash

   # 中国大陆用户可追加 --use-mirror。
   bash requirements/install.sh embodied --model openpi --env geniesim
   source .venv/bin/activate

**Docker 镜像**

在 RLinf 主机的 RLinf 仓库根目录构建镜像，通过 ``geniesim`` 构建上下文传入客户端源码：

.. code:: bash

   docker build -f docker/Dockerfile \
     --build-arg BUILD_TARGET=embodied-geniesim \
     --build-arg NO_MIRROR=1 \
     --build-context "geniesim=$GENIESIM_CLIENT_PATH" \
     -t rlinf:embodied-geniesim .

镜像提供基于 NVIDIA CUDA 12.8 的 OpenPI 训练环境和 GenieSim 客户端，连接 GenieSim 服务即可开展训练。

仍在 RLinf 主机的 RLinf 仓库根目录，启动容器并挂载代码仓库：

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 32g \
      --network host \
      --name rlinf \
      -v "$(pwd):/workspace/RLinf" \
      --workdir /workspace/RLinf \
      rlinf:embodied-geniesim

容器启动后位于 ``/workspace/RLinf``。在 RLinf 容器内激活 OpenPI：

.. code:: bash

   source switch_env openpi

下载模型
----------------------------------------

下载面向 GenieSim 的 OpenPI |pi| SFT 检查点，作为 PPO 或 GRPO 训练的起点。

该检查点使用 `GenieSim3.0-Dataset 的 instruction_and_robust 任务集 <https://huggingface.co/datasets/agibot-world/GenieSim3.0-Dataset/tree/main/task_suite/instruction_and_robust>`__ 中的 5,000 条轨迹进行 SFT，训练 45,000 步，action chunk 长度设为 10。

**模型仓库：上传后补充。** 请将下方的 ``<GENIESIM_PI05_SFT_REPO>`` 替换为发布的 Hugging Face 仓库 ID。

在已激活的 OpenPI 环境中下载模型：Docker 用户在 RLinf 容器内的 ``/workspace/RLinf`` 执行，自定义环境用户在主机的 RLinf 仓库根目录执行。已有检查点时，将 ``GENIESIM_CHECKPOINT_PATH`` 设为当前环境可访问的检查点目录，然后继续“运行”部分。

.. code:: bash

   pip install huggingface-hub

   # 中国大陆用户可设置镜像：
   # export HF_ENDPOINT=https://hf-mirror.com

   export GENIESIM_CHECKPOINT_PATH="$PWD/checkpoints/geniesim_sft"
   hf download "<GENIESIM_PI05_SFT_REPO>" \
      --local-dir "$GENIESIM_CHECKPOINT_PATH"

示例将检查点保存到 RLinf 仓库下的目录；使用其他目录时，在下载前修改 ``GENIESIM_CHECKPOINT_PATH``。训练与评测均通过该变量读取模型路径，也可以直接在 YAML 中设置 RLinf 进程可访问的路径：

.. include:: _model_path.rst

运行
----------------------------------------

在仿真主机上，按照 `GenieSim post-training server 说明 <https://github.com/AgibotTech/genie_sim#39-post-training-server>`__ 准备 ``geniesim_assets`` 并构建基于 Isaac Sim 6.0.1 的镜像。GenieSim 源码版本应与已安装的客户端一致。

**启动仿真 server**

在仿真主机上、容器外的 GenieSim 仓库根目录，启动并进入 server 容器：

.. code:: bash

   geniesim docker --profile post-train up --headless
   geniesim docker --profile post-train into

在 ``post_train_server`` 容器内的 ``/workspace`` 目录，启动 server 并查看状态。下方数量、起始端口和 GPU 均为示例，请按所用 RLinf 配置调整：

.. code:: bash

   geniesim post-train server start --count 2 --base-port 5601 --devices 0
   geniesim post-train server status --count 2 --base-port 5601

在仿真主机另开终端，于容器外的任意目录查看主机 IP 和监听中的 TCP 端口：

.. code:: bash

   hostname -I
   ss -ltn

将下文的 ``sim-host`` 替换为 RLinf 主机可访问的仿真主机 IP，并使用 server 实际监听的端口。详细参数见 `GenieSim server 说明 <https://github.com/AgibotTech/genie_sim/blob/HEAD/source/rlinf_geniesim/README.md>`__。

以下 RLinf 命令均在已激活的 OpenPI 环境和 RLinf 仓库根目录执行：Docker 用户使用 RLinf 容器内的 ``/workspace/RLinf``，自定义环境用户使用主机上的仓库目录。检查点路径沿用上文设置的 ``GENIESIM_CHECKPOINT_PATH``。

.. note::

   在 RTX 4090 上，将 ``env.train.override_cfg.render_width`` 设为 360 时，单个 GenieSim server 最多支持 128 个并行环境；使用默认渲染宽度 640 时，单个 server 最多支持 32 个并行环境。

   ``env.train.override_cfg.images_at_chunk_end_only`` 默认开启。开启后，RLinf 与仿真端以整个 action chunk 为单位交互，并只渲染该 chunk 最后一帧的图像。

**训练**

运行单任务 PPO：

.. code:: bash

   export GENIESIM_HOST_0=sim-host
   bash examples/embodiment/run_embodiment.sh geniesim_pick_block_color_ppo_openpi_pi05

GRPO 和多任务配方见 `可视化与结果`_。各配方的 ``env.train.override_cfg.addresses`` 列出所需的 server 端点；使用其他主机或端口时，修改该列表。

训练默认关闭验证。将 ``runner.val_check_interval`` 设为正数启用后，还需将 ``GENIESIM_EVAL_HOST`` 设为验证用仿真主机，并启动 ``env.eval.override_cfg.addresses`` 列出的额外 server。

**评测**

设置仿真 server 的地址并运行独立评测。下方地址仅为示例，请替换为实际使用的 IP 和端口：

.. code:: bash

   export GENIESIM_EVAL_SERVER_ADDR=tcp://sim-host:5601
   bash evaluations/run_eval.sh geniesim_pick_block_color_eval

``GENIESIM_EVAL_SERVER_ADDR`` 对应 ``evaluations/geniesim/geniesim_pick_block_color_eval.yaml`` 中的 ``env.eval.override_cfg.address``，变量值应包含实际使用的端口。更多选项见 :doc:`评测 CLI <../../evaluations/reference/cli>`。

可视化与结果
----------------------------------------

在 RLinf 仓库根目录启动 TensorBoard：

.. code:: bash

   tensorboard --logdir ../results --port 6006

最关键的信号是 ``env/success_once``。全部可记录指标见
:doc:`训练指标 <../../reference/metrics>`。

需要 rollout 视频时，在 env 配置中开启：

.. code:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

以下实验使用 Agibot G2 双臂轮式机器人，GenieSim 仿真运行在 RTX 4090 上。

下表比较 SFT 初始检查点、单任务 PPO 与 GRPO，以及分别使用 4、6、10 个任务混训的 PPO 模型，所有检查点采用统一评测设置。

.. list-table::
   :class: geniesim-results-table
   :header-rows: 1
   :widths: 34 11 11 11 11 11 11

   * - subtask name
     - SFT
     - 单任务 PPO
     - 单任务 GRPO
     - 4 任务 PPO
     - 6 任务 PPO
     - 10 任务 PPO
   * - ``pick_block_color``
     - 71.00%
     - 96.00%
     - 94.00%
     - 97.00%\ :sup:`※`
     - 84.00%\ :sup:`※`
     - 95.00%\ :sup:`※`
   * - ``pick_block_size``
     - 60.00%
     - 92.00%
     - —
     - 92.00%\ :sup:`※`
     - 88.00%\ :sup:`※`
     - 92.00%\ :sup:`※`
   * - ``pick_billiards_color``
     - 66.00%
     - 93.00%
     - —
     - 91.00%\ :sup:`※`
     - 82.00%\ :sup:`※`
     - 89.00%\ :sup:`※`
   * - ``pick_block_shape``
     - 32.00%
     - 45.00%
     - —
     - 37.00%\ :sup:`※`
     - 37.00%\ :sup:`※`
     - 41.00%\ :sup:`※`
   * - ``pick_specific_object``
     - 64.00%
     - 85.00%
     - 70.00%
     - 75.00%
     - 81.00%\ :sup:`※`
     - 83.00%\ :sup:`※`
   * - ``pick_object_type``
     - 50.00%
     - 65.00%
     - —
     - 64.00%
     - 63.00%\ :sup:`※`
     - 64.00%\ :sup:`※`
   * - ``pick_follow_logic_or``
     - 56.00%
     - 80.00%
     - —
     - 76.00%
     - 71.00%
     - 78.00%\ :sup:`※`
   * - ``pick_block_number``
     - 50.47%
     - 63.55%
     - —
     - 58.88%
     - 57.01%
     - 61.68%\ :sup:`※`
   * - ``pick_common_sense``
     - 47.00%
     - 65.00%
     - —
     - 53.00%
     - 51.00%
     - 53.00%\ :sup:`※`
   * - ``straighten_object``
     - 16.00%
     - —
     - —
     - 14.00%
     - 8.00%
     - 18.00%\ :sup:`※`

:sup:`※` 表示该任务包含在对应多任务模型的训练组合中。

对应的训练参考配置均位于 ``examples/embodiment/config/``：

- 单任务 PPO：``geniesim_pick_block_color_ppo_openpi_pi05.yaml``；训练其他任务时调整任务选择。
- 单任务 GRPO：``geniesim_pick_block_color_grpo_openpi_pi05.yaml``；训练其他任务时调整任务选择。
- 4 任务 PPO：``geniesim_multitask4_ppo_openpi_pi05.yaml``。
- 6 任务 PPO：``geniesim_multitask6_ppo_openpi_pi05.yaml``。
- 10 任务 PPO：``geniesim_multitask10_ppo_openpi_pi05.yaml``。

附录
----------------------------------------

.. _geniesim-parameters:

泛化参数说明
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 18 47

   * - 字段名称
     - 默认设置
     - 说明
   * - ``env.train.seed``
     - ``0``
     - 训练随机种子；非负值启用 reset 随机扰动。
   * - ``env.eval.seed``
     - ``-1``
     - 关闭评测时的 reset 随机扰动。
   * - ``override_cfg.randomization``
     - ``null``
     - 在 env.train 或 env.eval 下配置；null 沿用仿真端默认值。下列泛化字段均写在该配置中。
   * - ``use_task_generalization``
     - ``true``
     - 继承任务默认设置；false 关闭该来源，显式配置值优先。
   * - ``use_benchmark_generalization``
     - ``true``
     - 继承基准设置；false 关闭该来源，显式配置值优先。
   * - ``object_xy_radius``
     - ``0.03``
     - 物体 XY 位置扰动幅度，单位 m；设为 0 关闭。
   * - ``init_joint_thresh``
     - ``0.05``
     - 关节初始位置扰动范围为 ±0.05 rad；设为 0 关闭。
   * - ``init_base_xy_thresh``
     - ``[0.05, 0.05]``
     - 底盘 X、Y 各扰动 ±0.05 m；null 关闭。任务自带逐实例底盘位姿时跳过默认扰动。
   * - ``render.lights.enable``
     - ``false``
     - 灯光扰动开关；true 开启。
   * - ``render.material.enable``
     - ``false``
     - 材质扰动开关；true 开启。
   * - ``render.background.enable``
     - ``false``
     - 背景变化开关；true 开启。
   * - ``render.camera_pose.enable``
     - ``false``
     - 相机位姿扰动开关；true 开启，并设置 render.camera_pose.threshold 中的非零阈值。
   * - ``joint_pd.enable``
     - ``false``
     - 关节 PD 参数扰动开关；true 开启，并设置 joint_pd.kp_scale 与 joint_pd.kd_scale 的倍率范围。

.. _geniesim-training-tasks:

训练任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. rubric:: 指令理解

.. list-table::
   :header-rows: 1
   :widths: 5 35 28 32

   * - 序号
     - subtask name
     - task name
     - 说明
   * - 1
     - ``pick_block_color``
     - ``table_task_1_g2_op``
     - 抓取指定颜色的方块。
   * - 2
     - ``pick_billiards_color``
     - ``table_task_1_g2_op``
     - 抓取指定颜色的台球。
   * - 3
     - ``pick_specific_object``
     - ``table_task_0_g2_op``
     - 抓取指令指定的物体。
   * - 4
     - ``pick_block_size``
     - ``table_task_1_g2_op``
     - 抓取指定尺寸的方块。
   * - 5
     - ``pick_object_type``
     - ``table_task_0_g2_op``
     - 抓取指定类别的物体。
   * - 6
     - ``pick_follow_logic_or``
     - ``table_task_0_g2_op``
     - 抓取满足指令中任一条件的物体。
   * - 7
     - ``pick_block_number``
     - ``table_task_0_g2_op``
     - 抓取带有指定数字的方块。
   * - 8
     - ``pick_common_sense``
     - ``table_task_0_g2_op``
     - 根据常识描述选择并抓取物体。
   * - 9
     - ``pick_block_shape``
     - ``table_task_0_g2_op``
     - 抓取指定形状的方块。
   * - 10
     - ``straighten_object``
     - ``table_task_0_g2_op``
     - 将桌面上的物体摆正。

.. rubric:: 空间理解

.. list-table::
   :header-rows: 1
   :widths: 5 35 28 32

   * - 序号
     - subtask name
     - task name
     - 说明
   * - 11
     - ``pick_object_absolute_position``
     - ``table_task_2_g2_op``
     - 抓取位于指定绝对位置的物体。
   * - 12
     - ``pick_object_relative_position``
     - ``table_task_2_g2_op``
     - 根据物体之间的相对位置选择并抓取目标物体。
   * - 13
     - ``place_object_relative_position``
     - ``table_task_2_g2_op``
     - 将物体放到相对于另一物体的指定位置。
   * - 14
     - ``place_beverage_to_anothers_position``
     - ``table_task_2_g2_op``
     - 将饮料放到另一物体所在的位置。
   * - 15
     - ``sort_cubes_by_size``
     - ``table_task_2_g2_op``
     - 将方块按尺寸排序。
   * - 16
     - ``sort_number``
     - ``table_task_2_g2_op``
     - 将带数字的物体按数字顺序排列。
   * - 17
     - ``stack_bowls``
     - ``table_task_2_g2_op``
     - 将碗叠放在一起。
   * - 18
     - ``stack_three_building_blocks``
     - ``table_task_2_g2_op``
     - 将三个积木堆叠起来。

.. rubric:: 操作技能

.. list-table::
   :header-rows: 1
   :widths: 5 35 28 32

   * - 序号
     - subtask name
     - task name
     - 说明
   * - 19
     - ``open_door``
     - ``home_g2_op``
     - 打开门。
   * - 20
     - ``pour_workpiece``
     - ``laboratory_06_g2_op``
     - 倾倒工件。
   * - 21
     - ``take_wrong_item_shelf``
     - ``market_01_g2_op``
     - 取出货架上放错位置的物品。
   * - 22
     - ``hold_pot``
     - ``kitchen_02_g2_op``
     - 用双臂端起锅具。
   * - 23
     - ``place_block_into_box``
     - ``table_task_2_g2_op``
     - 将方块放入盒中。
   * - 24
     - ``stock_and_straighten_shelf``
     - ``market_00_g2_op``
     - 向货架补货并整理物品。
   * - 25
     - ``clean_the_desktop``
     - ``study_room_05_g2_op``
     - 清理桌面。
