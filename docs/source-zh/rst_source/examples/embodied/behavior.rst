2026 BEHAVIOR 挑战的强化学习训练
========================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/behavior.jpg
   :align: center
   :width: 90%

   BEHAVIOR 基准（图片来源：`BEHAVIOR <https://behavior.stanford.edu>`__）。

`2026 BEHAVIOR 挑战 <https://behavior.stanford.edu/challenge/index.html>`__ 使用 OmniGibson 和 NVIDIA Isaac Sim 模拟家居任务。RLinf 的示例用双臂移动机器人 R1 Pro 训练视觉语言动作模型（VLA）。

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
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · Python 3.11 · RLinf 安装的 BEHAVIOR-1K v3.9.4 分支 · Isaac Sim 5.1。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 说明
   * - 任务
     - 任务说明表收录了全部 100 个挑战任务。选择要评测的任务，将其名称填入 ``omni_config.task.activity_name`` 即可。默认任务为 ``turning_on_radio``\ 。
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
     - R1 Pro 的 23 维连续控制量包括底盘的 3 个速度指令（前后移动、左右横移和转向）、躯干 4 个关节的目标位置、左右臂各 7 个关节的目标位置，以及左右夹爪各 1 个开合指令。


安装
----------------------------------------

.. warning::

   使用配有 RT Core 的 NVIDIA GPU。操作系统、驱动和显存要求见 `Isaac Sim 5.1 系统要求 <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_\ 。

.. include:: _setup_common.rst
   :end-before: 然后，使用下列两种方式之一

**选项 1：Docker 镜像** — 用于 BEHAVIOR-1K v3.9.4 分支的 RLinf Docker 镜像待发布。

**选项 2：自定义环境**

在 RLinf 仓库根目录运行以下命令。RLinf、OpenPI 和仿真器共用一个 Python 3.11.14 环境。脚本会安装 Torch 2.7.0，以及所需版本的 LeRobot、Hugging Face Hub 和 Warp。

.. code-block:: bash

   unset ISAAC_PATH EXP_PATH CARB_APP_PATH
   bash requirements/install.sh embodied --model openpi --env behavior
   source .venv/bin/activate
   python -c "from importlib.metadata import version; print(version('omnigibson'), version('isaacsim'))"
   git -C "${BEHAVIOR_PATH:-$VIRTUAL_ENV/BEHAVIOR-1K}" rev-parse HEAD

v3.9.4 源码中的 OmniGibson 软件包版本号仍为 3.9.3。按默认方式安装后，Git 提交号应与 ``requirements/install.sh`` 中指定的提交一致，Isaac Sim 版本应为 5.1.x。


下载资产
----------------------------------------

首次下载时，为 2026 仿真资产选择独立的根目录。已有完整资产时，可直接使用现有目录。每次启动 RLinf 前，在同一终端导出 ``OMNIGIBSON_DATA_PATH`` 环境变量。

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data

在已激活的 RLinf 环境中运行以下命令，可下载 2026 挑战所需的仿真资产。首次获取场景资产密钥时，OmniGibson 会提示你接受数据集许可协议。

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

PPO 训练沿用以下入口。

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
   - 组件放置和吞吐调优 → :doc:`组件放置 <../../concepts/placement>` 与 :doc:`执行模式 <../../concepts/execution_modes>`
   - 指标定义和日志后端 → :doc:`训练指标 <../../reference/metrics>`

.. warning::

   OpenVLA-OFT 和 π₀ 在 BEHAVIOR-1K v3.9.4 分支上的训练成功率待验证。

独立评测
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

原则上，任何在 BEHAVIOR 上有非零成功率、并且已经转换为 PyTorch
格式的 ``pi05`` checkpoint 都可以使用该配置评估。我们只将
OpenPI-Comet 作为示例来源：

- https://huggingface.co/sunshk/openpi_comet/tree/main

下载后，可参考以下仓库将权重转换为 PyTorch 格式：

- https://github.com/mli0603/openpi-comet

感谢 OpenPI-Comet 作者开源模型和工具，这有助于 RLinf 中的可复现评估。

将 ``rollout.model.model_path`` 设为转换后的模型目录。评测默认使用所选任务的官方超时。步数配置、成功率指标、日志和视频见 :doc:`BEHAVIOR-1K 评测指南 <../../evaluations/guides/behavior>`。

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

在同一环境中按 :doc:`BEHAVIOR-1K 评测指南 <../../evaluations/guides/behavior>` 中的命令运行该模型。

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
     - 固定布局用 ``disabled``\ ，从缓存随机取布局用 ``offline``\ 。用 ``online`` 重新采样时，每个仿真进程只能运行一个环境。各模式所需配置见 :ref:`behavior-eval-config`\ 。
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
     - 设为 ``True`` 后，仿真器只在每个动作块实际执行的最后一步渲染并读取观测。
   * - ``num_env_subprocess``
     - 每个环境 worker 启动的仿真进程数，默认为 1。详见 :ref:`不同流水线阶段共用仿真进程时的限制 <behavior-eval-config>`。

2026 仿真资产已包含官方评测实例。选择实例的方法见 :doc:`BEHAVIOR 评测指南 <../../evaluations/guides/behavior>`。生成额外训练布局的工具需要 OmniGibson 3.7.2。


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
