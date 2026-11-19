基于Behavior评测平台的强化学习训练
========================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/behavior.jpg
   :align: center
   :width: 90%

   BEHAVIOR 基准（图片来源：`BEHAVIOR <https://behavior.stanford.edu>`__）。

`2026 BEHAVIOR 挑战 <https://behavior.stanford.edu/challenge/index.html>`__ 使用 OmniGibson 3.9.3 和 NVIDIA Isaac Sim 5.1 模拟家居任务。RLinf 的示例用双臂移动机器人 R1 Pro 训练视觉语言动作模型（VLA）。

概览
----------------------------------------

在 RLinf 环境中准备 2026 仿真资产，并为目标任务选择 PPO 训练或独立评测配置。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      OpenVLA-OFT · π₀ / π₀.₅

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      2026 挑战 · 100 个任务

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 支持光追的 GPU

| **你将完成：** 核对仿真版本 → 下载 2026 仿真资产与模型 → 选择 RLinf 配置。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · Python 3.11 · OmniGibson 3.9.3 · Isaac Sim 5.1。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 说明
   * - 任务
     - 官方挑战包含 100 个家居任务。本分支的任务说明表收录了 50 个名称，从中选一个填入 ``omni_config.task.activity_name`` 即可。默认任务为 ``turning_on_radio``\ 。
   * - 机器人
     - RLinf 示例和官方默认配置都使用双臂移动机器人 R1 Pro。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 说明
   * - RLinf 观测
     - 头部 RGB 图像（720×720）、左右腕部 RGB 图像（各 480×480）、机器人本体状态和任务指令。
   * - 官方评测观测
     - 默认模式提供三路 224×224 RGB 和本体状态。全分辨率模式提供头部 720×720、两腕各 480×480 的 RGB 和深度图，以及本体状态。详见 `官方评测说明 <https://behavior.stanford.edu/challenge/evaluation.html>`_\ 。
   * - 动作 (Action)
     - R1 Pro 的 23 维连续控制量包括底盘的 3 维速度、躯干的 4 个关节目标、每条手臂的 7 个关节目标，以及每侧夹爪的 1 个开合命令。


安装
----------------------------------------

.. warning::

   使用配有 RT Core 的 NVIDIA GPU。操作系统、驱动和显存要求见 `Isaac Sim 5.1 系统要求 <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_\ 。

.. include:: _setup_common.rst
   :end-before: 然后，使用下列两种方式之一

**选项 1：Docker 镜像** — BEHAVIOR 3.9.3 镜像待发布。

**选项 2：自定义环境** — RLinf 的 ``--env behavior`` 安装入口尚待适配 3.9.3。已有对应版本的 RLinf 环境时，激活它并核对版本。以下命令假设环境位于仓库的 ``.venv`` 目录。

.. code-block:: bash

   source .venv/bin/activate
   python -c "from importlib.metadata import version; print(version('omnigibson'), version('isaacsim'))"

输出为 OmniGibson 3.9.3 和 Isaac Sim 5.1.x 时，可使用下节的 SDK 下载命令。尚未安装 SDK 时，也可先下载下节列出的三个压缩包。


下载资产
----------------------------------------

首次下载时，为 2026 仿真资产选择独立的根目录。已有完整资产时，可直接使用现有目录。每次启动 RLinf 前，在同一终端导出 ``OMNIGIBSON_DATA_PATH`` 环境变量。

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data

在已激活的 RLinf 环境中运行以下命令，可下载或补齐 3.9.3 所需的资产。

.. code-block:: bash

   mkdir -p "$OMNIGIBSON_DATA_PATH"
   python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
   python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets()"
   python -c "from omnigibson.utils.asset_utils import download_2026_challenge_task_instances; download_2026_challenge_task_instances()"

**这些命令会：**

1. 将机器人模型包 ``omnigibson-robot-assets-3.8.2.zip`` 解压到 ``omnigibson-robot-assets/`` 目录。
2. 将场景与物体包 ``behavior-1k-assets-3.9.0.zip`` 解压到 ``behavior-1k-assets/`` 目录，并将密钥 ``omnigibson.key`` 保存到资产根目录。
3. 将任务实例包 ``2026-challenge-task-instances.zip`` 解压到 ``2026-challenge-task-instances/`` 目录。其中包含初始场景布局和 100 项任务清单。

如需提前准备资产，可从 `官方仿真资产仓库 <https://huggingface.co/datasets/behavior-1k/zipped-datasets/tree/main>`_ 下载上述三个压缩包，分别解压到对应目录。安装 SDK 后，运行上方调用 ``download_behavior_1k_assets()`` 的命令即可获取密钥。


下载模型
----------------------------------------

RLinf 于 2025 年发布了以下两个 BEHAVIOR checkpoint。选择模型后，用以下任一方式下载。

.. code-block:: bash

   # 方法 1：git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-Behavior
   git clone https://huggingface.co/RLinf/RLinf-Pi0-Behavior

   # 方法 2：huggingface-hub（国内用户可设置 HF_ENDPOINT=https://hf-mirror.com）
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-Behavior --local-dir RLinf-OpenVLAOFT-Behavior
   hf download RLinf/RLinf-Pi0-Behavior --local-dir RLinf-Pi0-Behavior

.. include:: _model_path.rst


运行
----------------------------------------

.. warning::

   将 BEHAVIOR env worker 放在从 0 开始的 GPU 上。IsaacSim 在 env worker 从更靠后的 GPU rank 启动时可能卡住。

每个配方都是 ``examples/embodiment/config/`` 下的一个 YAML 配置：

.. list-table::
   :header-rows: 1
   :widths: 34 26 40

   * - 模型 / 用途
     - 算法
     - 配置
   * - OpenVLA-OFT
     - PPO
     - ``behavior_ppo_openvlaoft.yaml``
   * - π₀
     - PPO
     - ``behavior_ppo_openpi.yaml``
   * - π₀.₅
     - PPO
     - ``behavior_ppo_openpi_pi05.yaml``

RLinf 的 3.9.3 环境适配完成后，可用原有入口启动训练：

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data
   bash examples/embodiment/run_embodiment.sh behavior_ppo_openvlaoft

**这个命令会：**

1. 加载 ``examples/embodiment/config/behavior_ppo_openvlaoft.yaml`` 及共享环境配置 ``examples/embodiment/config/env/behavior_r1pro.yaml``。
2. 按组件放置配置启动 actor、rollout 和 BEHAVIOR env 的 Ray worker。
3. 运行 PPO 训练，并把日志和检查点写入 ``runner.logger.log_path``。

.. admonition:: 进一步配置
   :class: note

   - BEHAVIOR 吞吐调优 → 先增加 env GPU 数量，再调 ``env.num_env_subprocess`` 和 ``env.train.total_num_envs``。
   - 按可用 GPU 显存和所选场景的显存占用设置仿真子进程数。
   - 缓存任务实例 → 使用 ``rlinf/envs/sim/behavior/instance_generator.py`` 和 ``examples/embodiment/config/env/behavior_r1pro.yaml`` 生成。
   - 组件放置和吞吐调优 → :doc:`组件放置 <../../concepts/placement>` 与 :doc:`执行模式 <../../concepts/execution_modes>`
   - 指标定义和日志后端 → :doc:`训练指标 <../../reference/metrics>`

.. warning::

   OpenVLA-OFT 和 π₀ 在 BEHAVIOR 3.9.3 上的训练成功率待验证。

独立评测
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

原则上，任何在 BEHAVIOR 上有非零成功率、并且已经转换为 PyTorch
格式的 ``pi05`` checkpoint 都可以使用该配置评估。我们只将
OpenPI-Comet 作为示例来源：

- https://huggingface.co/sunshk/openpi_comet/tree/main

下载后，可参考以下仓库将权重转换为 PyTorch 格式：

- https://github.com/mli0603/openpi-comet

感谢 OpenPI-Comet 作者开源模型和工具，这有助于 RLinf 中的可复现评估。

转换完成后，按如下方式更新 ``behavior_openpi_pi05_eval.yaml``：

1. 将 ``rollout.model.model_path`` 设置为转换后的模型目录。
2. 如需更长轨迹，提高 ``env.eval`` 的 ``max_episode_steps`` 与 ``max_steps_per_rollout_epoch``，例如设置为 ``4096``。

.. code-block:: yaml

   env:
     eval:
       max_episode_steps: 4096
       max_steps_per_rollout_epoch: 4096

请先按本页说明准备 2026 仿真资产。:doc:`BEHAVIOR-1K 评测指南 <../../evaluations/guides/behavior>` 说明如何设置模型路径、使用 ``behavior_openpi_pi05_eval`` 启动独立评测、读取成功率指标 ``eval/success_once`` 以及查看日志和视频。

--------------

**5. 评估 JAX 对齐的 Pi0.5 checkpoint**

BEHAVIOR 评估使用 ``model_type: openpi``。若权重来自 :doc:`sft_openpi` 或 OpenPI checkpoint 转换器，评估配置为：

- ``evaluations/behavior/behavior_openpi_pi05_eval.yaml``

该配置以纯评估模式运行（``runner.only_eval: True``），并消费转换后的
checkpoint，即由 OpenPI checkpoint 转换器
（``ckpt_convertor.openpi`` 的 ``openpi_pytorch_to_openpi`` /
``sft_to_openpi``）产出的权重。
将模型路径以 ``/path/to/...`` 占位符的形式直接写在配置中：

- ``rollout.model.model_path``：评估 checkpoint。

归一化统计从转换后 checkpoint 中打包的
``physical-intelligence/behavior/norm_stats.json`` 读取。

RLinf 环境完成 3.9.3 适配后，可用以下命令运行独立评测。

.. code:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data
   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval

.. note::

   评估时流匹配动作头使用非确定性（随机）采样噪声，因此重复运行之间，单次轨迹与
   成功次数会有轻微差异。


配置参考
----------------------------------------

BEHAVIOR 环境由 ``examples/embodiment/config/env/behavior_r1pro.yaml`` 驱动。RLinf 先加载
OmniGibson 的基础配置（``base_config_name``），再应用 ``omni_config`` 覆盖项（见
``rlinf/envs/sim/behavior/utils.py`` 中的 ``setup_omni_cfg``）。下表中的字段控制 reset 行为、场景
加载、仿真器频率与吞吐，大多有合理默认值，仅在自定义任务或调优性能时需要修改。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 配置项
     - 含义
   * - ``base_config_name``
     - 在 ``omni_config`` 覆盖前加载的 OmniGibson 基础配置（如 ``r1pro_behavior``）。
   * - ``omni_config.task.type`` / ``omni_config.scene.type``
     - 显式保留 ``BehaviorTask`` / ``InteractiveTraversableScene``，以确保覆盖后选中预期的上游
       OmniGibson 类。
   * - ``omni_config.task.activity_name``
     - 任务名称，例如 ``turning_on_radio``\ 。同一任务的初始布局由 ``omni_config.task.activity_instance_id`` 指定。
   * - ``omni_config.task.instance_resample_mode``
     - reset 时的实例切换：``disabled``\ （加载固定的 ``activity_instance_id``）、``offline``
       （启动时扫描一次 ``activity_instance_dir``，每次 reset 采样一个缓存实例——
       ``*_template.json`` 走较重的场景重载路径，``*_template-tro_state.json`` 走较轻的原地路径）、
       或 ``online``\ （需 ``online_object_sampling: True`` 且 ``use_presampled_robot_pose: False``）。
   * - ``omni_config.task.activity_instance_dir``
     - 缓存实例 JSON 目录（``*_template.json`` / ``*_template-tro_state.json``），供 ``offline``
       模式以及 ``disabled`` 模式下的固定 id 加载使用。
   * - ``omni_config.task.instance_file_format``
     - 缓存实例格式：``template``\ （完整重载）或 ``tro_state``\ （仅任务相关、轻量）。RLinf 接受不含
       ``robot_poses`` 的 ``tro_state`` 文件，此时会清除过期的缓存机器人位姿元数据，reset 回退到
       任务默认机器人位姿。
   * - ``omni_config.scene.partial_scene_load``
     - 为 ``true`` 时自动按 ``activity_name`` 填充 ``scene.load_room_types``\ （减少启动时间与内存），
       需要 ``activity_name`` 与 ``scene_model``。为 ``false``/省略时需显式设置 ``load_room_types``。
   * - ``omni_config.camera.head_resolution`` / ``omni_config.camera.wrist_resolution``
     - 头部 / 腕部相机分辨率（默认 720×720 / 480×480，应用到 R1Pro 传感器）。
   * - ``omni_config.env.action_frequency`` / ``rendering_frequency`` / ``physics_frequency``
     - RLinf 的 R1 Pro 配置将动作、渲染和物理步进频率分别设为 60、60 和 120 Hz。越高越慢。
   * - ``omni_config.env.automatic_reset``
     - 保持 ``False``——reset 由 RLinf 训练/评估循环显式控制。
   * - ``omni_config.env.flatten_obs_space`` / ``flatten_action_space``
     - 保持 ``False`` 以保留结构化的观测 / 动作空间。
   * - ``omni_config.macro.use_gpu_dynamics``
     - ``False`` 通常更快；仅在需要粒子 / 流体时启用。
   * - ``omni_config.macro.enable_object_states``
     - 保持 ``True``——``BehaviorTask`` 依赖物体状态。
   * - ``omni_config.macro.enable_transition_rules``
     - ``True`` 启用基于转移规则的状态变化（如切割、烹饪）。
   * - ``omni_config.macro.use_numpy_controller_backend``
     - ``True`` 使用 numpy 控制器后端，在单进程 / 中等并行下通常更快。
   * - ``skip_intermediate_renders``
     - 为 ``True`` 时跳过动作 chunk 内中间观测的采集（显著提升环境速度）。此时保存的视频只显示策略在
       chunk 边界观测到的帧。
   * - ``num_env_subprocess``
     - 将 ``num_envs`` 拆分到多个子进程，每个子进程承载自己的 Isaac/OmniGibson 仿真（见
       ``BehaviorProcess``）。默认 ``1``。**约束：** ``num_envs`` 必须能被 ``num_env_subprocess``
       整除。提高该值可缓解环境步进瓶颈，但会成倍增加进程数与内存。

生成缓存任务实例
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``rlinf/envs/sim/behavior/instance_generator.py`` 直接从 ``behavior_r1pro.yaml`` 生成
``*_template.json`` 与 ``*_template-tro_state.json`` 文件（它读取 ``scene_model``、
``activity_name``、``activity_definition_id``、机器人配置与房间加载设置，并临时切换为在线物体采样）。

2026 资产已包含官方评测实例。以下命令用于生成额外的训练布局。3.9.3 的默认输出目录仍待适配。使用以下命令前，先在 ``behavior_r1pro.yaml`` 中将 ``omni_config.task.activity_instance_dir`` 设为训练布局的输出目录。

.. code-block:: bash

   cd /path/to/RLinf

   python rlinf/envs/sim/behavior/instance_generator.py \
     --config examples/embodiment/config/env/behavior_r1pro.yaml \
     --output-format template \
     --start-idx 1 --end-idx 50

   python rlinf/envs/sim/behavior/instance_generator.py \
     --config examples/embodiment/config/env/behavior_r1pro.yaml \
     --output-format tro_state \
     --start-idx 1 --end-idx 50

生成的文件名遵循
``<scene_model>_task_<activity_name>_<activity_definition_id>_<activity_instance_id>_template(.json|-tro_state.json)``，
因此 ``--start-idx`` / ``--end-idx`` 决定 ``activity_instance_id`` 范围。``tro_state`` 输出仅在任务
元数据提供时包含顶层 ``robot_poses``，否则省略该键，reset 回退到任务默认机器人位姿。BEHAVIOR-1K 上游的
``multiply_b1k_tasks.py`` 仍可使用，但推荐 RLinf 的生成器，因为它直接读取 RLinf YAML 并保留
``activity_definition_id``。


可视化与结果
----------------------------------------

启动 TensorBoard 实时观察训练：

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

最值得关注的指标是 **``env/success_once``** —— 任务成功率。每个日志指标的含义见
:doc:`训练指标 <../../reference/metrics>`。

如需保存评估视频，在配置中启用：

.. code-block:: yaml

   env:
      eval:
         video_cfg:
            save_video: True
            video_base_dir: ${runner.logger.log_path}/video/eval


对于 BEHAVIOR 实验，我们受到了
`Behavior-1K baselines <https://github.com/StanfordVL/b1k-baselines.git>`_ 的启发，
仅进行了少量修改。感谢作者发布开源代码。
