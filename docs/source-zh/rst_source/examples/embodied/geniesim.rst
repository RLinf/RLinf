基于 GenieSim 的强化学习训练
========================================

.. |pi| replace:: π₀.₅

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/genie_sim_system_overview.webp
   :align: center
   :width: 100%
   :alt: Genie Sim 平台与功能概览。

   Genie Sim 系统概览（图片来源：`GenieSim <https://github.com/AgibotTech/genie_sim>`__）。

本文档介绍 `GenieSim <https://github.com/AgibotTech/genie_sim>`__ 官方的后训练流程，用 RLinf 训练和评测视觉语言动作（VLA）模型。Genie Sim 是智元机器人的具身智能仿真平台，把场景重建、资产生成、数据采集、自动评测和模型后训练放在同一套工具链里，用来训练、验证和迭代具身模型。任务覆盖指令理解、空间理解和操作技能。物体、场景和指令都会变化，用来看模型能不能泛化。RLinf 可以用 PPO 和 GRPO 微调这些模型，单任务和多任务都能训，目标是提高双臂操作能力。

GenieSim 仿真和 RLinf 训练分在两个 GPU 节点上跑。跑 RLinf 的是训练节点，跑仿真的是仿真节点。仿真节点用 NVIDIA RTX 系列显卡。

概览
----------------------------------------

用 PPO 和 GRPO 提高由自然语言引导的双臂操作。相对 SFT 检查点，10 任务 PPO 把按颜色抓方块的成功率从 71% 提到 95%，按尺寸抓方块从 60% 提到 92%。

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

| **操作顺序：** 安装 → 启动 GenieSim 服务 → 启动 ``run_embodiment.sh`` → 看 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 一台能连上的 GenieSim 服务 · 一个 VLA 检查点。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

这 25 个任务覆盖指令理解、空间理解和操作技能，可以用下面的单任务或多任务配方来训练。全部任务名见文末 :ref:`任务附录 <geniesim-training-tasks>`。

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

下面的 G2 实验用的是 OpenPI |pi|。模型看到的是三路相机图像、双臂和夹爪的状态，以及一句自然语言任务指令。:ref:`任务附录 <geniesim-training-tasks>` 按能力把这 25 个任务分了类。

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 规格
   * - 观测
     - 三路 640×480 的 RGB 图像。送进模型之前，按原来的宽高比缩放，再填充到 224×224。本体状态是 14 个手臂关节位置，外加 2 个夹爪开合角度。
   * - 动作
     - 16 维连续动作：14 个手臂关节的目标位置，外加 2 个夹爪目标。模型预测的是增量，每次输出 10 步，组成一个动作块。
   * - 奖励
     - 仿真器维护一个 0 到 1 的任务完成度。能分成先后阶段的任务，完成度等于已完成阶段数除以总阶段数；分不成阶段的任务只有 0（没成功）和 1（成功）。每一步的奖励是这一步的完成度减去上一步的完成度。以 ``pick_block_color`` 为例，先对准、再抓起，两段各让完成度增加 0.5，所以各给 0.5 的奖励，一整局加起来是 1。
   * - 任务指令
     - 使用 ``task_descriptions`` 里的完整自然语言指令。

模型的输入和输出都是 32 维，其中有效的状态和动作各占 16 维。仿真器状态里多出来的通道会被掩掉，模型输出也只保留机器人控制要用的那 16 维。状态和动作按检查点里的 ``assets/norm_stats.json`` 做归一化。

安装
----------------------------------------

先在训练节点上、任何容器之外，执行仓库的准备命令。

.. include:: _setup_common.rst
   :end-before: 然后，使用下列两种方式之一准备依赖

下面两种装法选一种。两种都会装上 GenieSim ``v3.5.0`` 里的 ``rlinf-geniesim-client``，``install.sh`` 会自动克隆对应源码。本机已经有 GenieSim 源码时，把 ``GENIESIM_PATH`` 指到那个目录。前置依赖和 GPU 驱动见 :doc:`安装 </rst_source/start/installation>`。

**自定义环境**

在训练节点的仓库根目录安装 OpenPI 和 GenieSim 客户端，然后激活环境：

.. code:: bash

   # 中国大陆用户可追加 --use-mirror。
   bash requirements/install.sh embodied --model openpi --env geniesim
   source .venv/bin/activate

**Docker 镜像**

在训练节点的仓库根目录构建镜像：

.. code:: bash

   docker build -f docker/Dockerfile \
     --build-arg BUILD_TARGET=embodied-geniesim \
     --build-arg NO_MIRROR=1 \
     -t rlinf:embodied-geniesim .

镜像基于 NVIDIA CUDA 12.8，里面有 OpenPI 训练环境和 GenieSim 客户端。连上 GenieSim 服务就可以开始训练。

还是在训练节点的 RLinf 仓库根目录，启动容器并挂载仓库：

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 32g \
      --network host \
      --name rlinf \
      -v "$(pwd):/workspace/RLinf" \
      --workdir /workspace/RLinf \
      rlinf:embodied-geniesim

容器启动后位于 ``/workspace/RLinf``。在容器里激活 OpenPI：

.. code:: bash

   source switch_env openpi

下载模型
----------------------------------------

下载 GenieSim 用的 OpenPI |pi| SFT 检查点 `RLinf-Pi05-GenieSim-G2-Instruction-SFT <https://huggingface.co/RLinf/RLinf-Pi05-GenieSim-G2-Instruction-SFT>`__，作为 PPO 或 GRPO 的起点。

这个检查点用 `GenieSim3.0-Dataset 的 instruction_and_robust 任务集 <https://huggingface.co/datasets/agibot-world/GenieSim3.0-Dataset/tree/main/task_suite/instruction_and_robust>`__ 里的 5,000 条轨迹做监督微调，训了 45,000 步，action chunk 长度是 10。

在已经激活的 OpenPI 环境里下载。用 Docker 时，在训练节点的 RLinf 容器内、``/workspace/RLinf`` 执行；用自定义环境时，在训练节点的 RLinf 仓库根目录执行。已经有检查点的话，把 ``GENIESIM_CHECKPOINT_PATH`` 设成当前环境能读到的目录，然后直接看「运行」。

.. code:: bash

   pip install huggingface-hub

   # 中国大陆用户可设置镜像：
   # export HF_ENDPOINT=https://hf-mirror.com

   export GENIESIM_CHECKPOINT_PATH="$PWD/checkpoints/geniesim_sft"
   hf download RLinf/RLinf-Pi05-GenieSim-G2-Instruction-SFT \
      --local-dir "$GENIESIM_CHECKPOINT_PATH"

示例把检查点放在 RLinf 仓库下面。要换目录，下载前改 ``GENIESIM_CHECKPOINT_PATH``。训练和评测都从这个变量读模型路径，也可以在 YAML 里直接写成 RLinf 进程能访问的路径：

.. include:: _model_path.rst

运行
----------------------------------------

在仿真节点上，按 `GenieSim post-training server 说明 <https://github.com/AgibotTech/genie_sim#39-post-training-server>`__ 准备 ``geniesim_assets``，并构建基于 Isaac Sim 6.0.1 的镜像。GenieSim 用 ``v3.5.0``，和已经装好的客户端是同一个版本。

**启动仿真 server**

在仿真节点上、容器外的 GenieSim 仓库根目录，启动并进入 server 容器：

.. code:: bash

   geniesim docker --profile post-train up --headless
   geniesim docker --profile post-train into

进入 ``post_train_server`` 容器后，在 ``/workspace`` 启动 server 并查看状态。下面的数量、起始端口和 GPU 都是例子，按实际的 RLinf 配置改：

.. code:: bash

   geniesim post-train server start --count 2 --base-port 5601 --devices 0
   geniesim post-train server status --count 2 --base-port 5601

这个例子在 GPU 0 上启动单任务配方需要的两个 server。各配方要多少 server、多少 GPU，见下表。仿真 GPU 是 RTX 4090。``--devices`` 把连续的端口分到同一张 GPU 上，对应配方里每个 env 进程负责的那一组端点：

.. list-table::
   :header-rows: 1
   :widths: 22 10 16 12 12 12 16

   * - 配方
     - server 数
     - 端口
     - 每个 server 的环境数
     - 仿真 GPU 数
     - 训练 GPU 数
     - ``--devices``
   * - 单任务 PPO / GRPO
     - 2
     - 5601–5602
     - 16
     - 1
     - 1
     - ``0``
   * - 4 任务 PPO
     - 4
     - 5601–5604
     - 16
     - 2
     - 2
     - ``0,1``
   * - 6 任务 PPO
     - 6
     - 5601–5606
     - 12
     - 2
     - 2
     - ``0,1``
   * - 10 任务 PPO
     - 12
     - 5601–5612
     - 14
     - 4
     - 4
     - ``0,1,2,3``

启动某个配方的训练 server 时，``--count`` 填表里的 server 数，``--base-port`` 用 5601，``--devices`` 填表里的值。

在仿真节点上另开一个终端，不要进容器，在任意目录查看这台机器的 IP 和正在监听的 TCP 端口：

.. code:: bash

   hostname -I
   ss -ltn

把下文的 ``sim-host`` 换成训练节点能够访问的仿真节点 IP，端口用 server 实际在听的端口。详细参数见 `GenieSim server 说明 <https://github.com/AgibotTech/genie_sim/blob/HEAD/source/rlinf_geniesim/README.md>`__。

下面的 RLinf 命令都在训练节点上、已激活的 OpenPI 环境里、RLinf 仓库根目录下执行。Docker 用户用容器内的 ``/workspace/RLinf``，自定义环境用训练节点上的仓库目录。检查点路径沿用上面设好的 ``GENIESIM_CHECKPOINT_PATH``。

.. note::

   在 RTX 4090 上，渲染宽度用默认的 640 时，建议每张 GPU 最多跑 42 个并行环境；把 ``env.train.override_cfg.render_width`` 设成 360 时，建议每张 GPU 最多跑 64 个。同一张 GPU 上的几个 server 合起来不能超过这个数。

   ``env.train.override_cfg.images_at_chunk_end_only`` 默认是开的。打开之后，RLinf 和仿真器按整个 action chunk 交互，并且只渲染这个 chunk 的最后一帧。

**训练**

运行单任务 PPO：

.. code:: bash

   export GENIESIM_HOST_0=sim-host
   bash examples/embodiment/run_embodiment.sh geniesim_pick_block_color_ppo_openpi_pi05

GRPO 和多任务配方列在 `可视化与结果`_。每个配方的 ``env.train.override_cfg.addresses`` 写着要用的 server 地址。换地址或端口时，改这个列表。

训练时默认不做验证。把 ``runner.val_check_interval`` 设成正数就会打开。打开之后，还要把 ``GENIESIM_EVAL_HOST`` 设成验证用的仿真节点，并按 ``env.eval.override_cfg.addresses`` 里的端口再启动一批 server，数量和训练 server 相同。

**评测**

先设好仿真 server 的地址，再跑单独的评测。下面的地址只是例子，换成实际的 IP 和端口：

.. code:: bash

   export GENIESIM_EVAL_SERVER_ADDR=tcp://sim-host:5601
   bash evaluations/run_eval.sh geniesim_pick_block_color_eval

``GENIESIM_EVAL_SERVER_ADDR`` 对应 ``evaluations/geniesim/geniesim_pick_block_color_eval.yaml`` 里的 ``env.eval.override_cfg.address``，值里要带上实际端口。更多选项见 :doc:`评测 CLI <../../evaluations/reference/cli>`。

可视化与结果
----------------------------------------

在 RLinf 仓库根目录启动 TensorBoard：

.. code:: bash

   tensorboard --logdir ../results --port 6006

最值得看的指标是 ``env/success_once``。全部会记下来的指标见
:doc:`训练指标 <../../reference/metrics>`。

要保存 rollout 视频时，在 env 配置里打开：

.. code:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

下面这些实验用的是智元 G2 双臂轮式机器人，GenieSim 跑在 RTX 4090 上。

表里比较的是 SFT 起点、单任务 PPO、单任务 GRPO，以及分别用 4 个、6 个、10 个任务混在一起训练的 PPO。所有检查点用同一套评测。

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
     - 85.00%
     - 92.00%\ :sup:`※`
     - 88.00%\ :sup:`※`
     - 92.00%\ :sup:`※`
   * - ``pick_billiards_color``
     - 66.00%
     - 93.00%
     - 91.00%
     - 91.00%\ :sup:`※`
     - 82.00%\ :sup:`※`
     - 89.00%\ :sup:`※`
   * - ``pick_block_shape``
     - 32.00%
     - 45.00%
     - 37.00%
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
     - 60.00%
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

:sup:`※` 表示这个任务包含在对应多任务模型的训练任务里。

表里的每个成功率，对子任务的每个测试用例只算一次。一个测试用例是一个场景实例配一条指令。单独评测会跑 128 条轨迹，比一个子任务的测试用例多，所以有的用例会跑两遍。写进表里之前，重复的那一遍已经去掉。训练或评测过程中记下来的成功率还含着这些重复，所以可能和表里的数差一点。

对应的训练配置都在 ``examples/embodiment/config/``：

- 单任务 PPO：``geniesim_pick_block_color_ppo_openpi_pi05.yaml``。换任务时改任务选择。
- 单任务 GRPO：``geniesim_pick_block_color_grpo_openpi_pi05.yaml``。换任务时改任务选择。
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
     - 训练用的随机种子。设成非负数时，reset 会加随机扰动。
   * - ``env.eval.seed``
     - ``-1``
     - 评测时关闭 reset 的随机扰动。
   * - ``override_cfg.randomization``
     - ``null``
     - 写在 env.train 或 env.eval 下面。null 表示沿用仿真器的默认值。下面的泛化字段都写在这项里。
   * - ``use_task_generalization``
     - ``true``
     - 沿用任务自带的设置。false 表示不用这一层。单独写明的值优先。
   * - ``use_benchmark_generalization``
     - ``true``
     - 沿用基准测试的设置。false 表示不用这一层。单独写明的值优先。
   * - ``object_xy_radius``
     - ``0.03``
     - 物体在 XY 方向上的位置扰动，单位是米。设为 0 即关闭。
   * - ``init_joint_thresh``
     - ``0.05``
     - 关节初始位置在 ±0.05 弧度内扰动。设为 0 即关闭。
   * - ``init_base_xy_thresh``
     - ``[0.05, 0.05]``
     - 底盘的 X、Y 各在 ±0.05 米内扰动。null 表示关闭。任务如果已经按实例给了底盘位姿，就不再加这层默认扰动。
   * - ``render.lights.enable``
     - ``false``
     - 是否扰动灯光。true 为开启。
   * - ``render.material.enable``
     - ``false``
     - 是否扰动材质。true 为开启。
   * - ``render.background.enable``
     - ``false``
     - 是否更换背景。true 为开启。
   * - ``render.camera_pose.enable``
     - ``false``
     - 是否扰动相机位姿。设为 true 开启，并在 render.camera_pose.threshold 里填非零阈值。
   * - ``joint_pd.enable``
     - ``false``
     - 是否扰动关节 PD 参数。设为 true 开启，并在 joint_pd.kp_scale 和 joint_pd.kd_scale 里填倍率范围。

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
