基于Behavior评测平台的强化学习训练
========================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/behavior.jpg
   :align: center
   :width: 90%

   BEHAVIOR 基准（图片来源：`BEHAVIOR <https://behavior.stanford.edu>`__）。

`BEHAVIOR <https://behavior.stanford.edu>`__ 是一个基于 NVIDIA IsaacSim / OmniGibson 的日常
家居活动基准。双臂 R1 Pro 机器人执行长程操作任务；RLinf 借助它对视觉-语言-动作（VLA）策略进行
强化学习微调。

概览
----------------------------------------

在 BEHAVIOR 家居任务上用 PPO 对 VLA 进行强化学习微调。

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

| **你将完成：** 安装 IsaacSim 依赖 → 下载资产与基座模型 → 运行 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · Python 3.11 · RLinf 安装的 BEHAVIOR-1K v3.9.4 分支 · Isaac Sim 5.1 与 2026 BEHAVIOR 资产· 基座检查点（见下文步骤）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 说明
   * - 任务
     - 来自 2026 BEHAVIOR 挑战的 100 个家居操作任务（通过 ``omni_config.task.activity_name`` 选择）。
   * - 机器人
     - IsaacSim / OmniGibson 上的双臂 R1 Pro。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 说明
   * - 观测 (Observation)
     - 头部相机 RGB（720×720）以及左右腕部 RealSense RGB（480×480）。
   * - 官方评测观测
     - 默认模式提供三路 224×224 RGB 和本体状态。全分辨率模式提供头部 720×720、两腕各 480×480 的 RGB 和深度图，以及本体状态。详见 `官方评测说明 <https://behavior.stanford.edu/challenge/evaluation.html>`_\ 。
   * - 动作 (Action)
     - 23 维连续动作：3 自由度底盘（x, y, rz）、4 自由度躯干、2×7 自由度手臂、2×1 自由度平行夹爪。


安装
----------------------------------------

.. warning::

   安装前先检查 IsaacSim 软硬件要求：

   - https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html
   - https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html

   Linux 需要 NVIDIA driver 580.65.06 或更高版本。Isaac Sim 5.1 不支持 A100、H100 等没有光线追踪能力的 GPU。
   优先使用 RTX 30/40 系列或更新的 GPU，以获得更稳定的视觉效果和训练体验。

.. include:: _setup_common.rst
   :end-before: 然后，使用下列两种方式之一

**选项 1：Docker 镜像** — 用于 BEHAVIOR-1K v3.9.4 分支的 RLinf Docker 镜像待发布。

**选项 2：自定义环境** — 安装 ``--env behavior`` 依赖组合：

.. code-block:: bash

   # 国内用户可以添加 --use-mirror 加速下载。
   unset ISAAC_PATH EXP_PATH CARB_APP_PATH
   bash requirements/install.sh embodied --model openvla-oft --env behavior
   # 或安装 OpenPI 环境：
   # bash requirements/install.sh embodied --model openpi --env behavior
   source .venv/bin/activate


.. _behavior-sim-assets:

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

RLinf 于 2025 年发布了以下检查点。下载对应模型族的检查点（任选一种方式）：

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

使用 ``run_embodiment.sh`` 启动一个配置：

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
   - 按 GPU 显存和所选场景的占用调节 subprocess 数量。
   - 缓存任务实例 → 使用官方 2026 实例，选择方法见 :doc:`BEHAVIOR-1K 评测指南 <../../evaluations/guides/behavior>`。
   - 组件放置和吞吐调优 → :doc:`组件放置 <../../concepts/placement>` 与 :doc:`执行模式 <../../concepts/execution_modes>`
   - 指标定义和日志后端 → :doc:`训练指标 <../../reference/metrics>`

.. warning::

   已知问题：在当前 BEHAVIOR 设置下，OpenVLA-OFT / π₀ 的训练成功率
   （``env/success_once``）可能保持为 0。该问题会在后续版本修复。

独立评测
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

收音机评测已使用 `π₀.₅ PT50 权重 <https://huggingface.co/RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32>`_ 验证。下载和运行步骤见 :doc:`BEHAVIOR-1K 评测指南 <../../evaluations/guides/behavior>`\ 。

原则上，任何在 BEHAVIOR 上有非零成功率、并且已经转换为 PyTorch
格式的 ``pi05`` checkpoint 都可以使用该配置评估。我们只将
OpenPI-Comet 作为示例来源：

- https://huggingface.co/sunshk/openpi_comet/tree/main

下载后，可参考以下仓库将权重转换为 PyTorch 格式：

- https://github.com/mli0603/openpi-comet

感谢 OpenPI-Comet 作者开源模型和工具，这有助于 RLinf 中的可复现评估。

转换完成后，按如下方式更新 ``behavior_openpi_pi05_eval.yaml``：

1. 将 ``rollout.model.model_path`` 设置为转换后的模型目录。
2. 将 ``env.eval`` 的 ``max_episode_steps`` 与 ``max_steps_per_rollout_epoch`` 保持为 ``null``，使用所选任务的官方超时。手动填写整数可自定义回合上限和采样预算。

.. code-block:: yaml

   env:
     eval:
       max_episode_steps: null
       max_steps_per_rollout_epoch: null

独立评估请走 :doc:`BEHAVIOR-1K 评测指南 <../../evaluations/guides/behavior>`。
该指南负责 ``OMNIGIBSON_DATA_PATH`` 设置、
``behavior_openpi_pi05_eval`` 启动命令和结果解读。

--------------

**5. 评估 JAX 对齐的 Pi0.5 checkpoint**

现有 :doc:`SFT 示例 <sft_openpi>` 使用 2025 数据集。
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
     - 任务名称，例如 ``turning_on_radio``。任务说明表覆盖全部 100 个挑战任务。
   * - ``omni_config.task.instance_resample_mode``
     - reset 时的实例切换：``disabled``\ （加载固定的 ``activity_instance_id``）、``offline``
       （启动时扫描一次 ``activity_instance_dir``，每次 reset 采样一个缓存实例——
       ``*_template.json`` 走较重的场景重载路径，``*_template-tro_state.json`` 走较轻的原地路径）、
       或 ``online``\ （需 ``online_object_sampling: True``、``use_presampled_robot_pose: False``，且每个仿真进程只有一个环境）。
   * - ``omni_config.task.activity_instance_dir``
     - 缓存实例 JSON 目录（``*_template.json`` / ``*_template-tro_state.json``），供 ``offline``
       模式以及 ``disabled`` 模式下的固定 id 加载使用。
   * - ``omni_config.task.instance_file_format``
     - 缓存实例格式：``template``\ （完整重载）或 ``tro_state``\ （仅任务相关、轻量）。RLinf 接受不含
       ``robot_poses`` 的 ``tro_state`` 文件，此时会清除过期的缓存机器人位姿元数据，reset 恢复场景的初始机器人位姿。
   * - ``omni_config.scene.partial_scene_load``
     - 设为 ``true`` 时，仅加载当前任务需要的房间。
   * - ``omni_config.camera.head_resolution`` / ``omni_config.camera.wrist_resolution``
     - 头部 / 腕部相机分辨率（默认 720×720 / 480×480，应用到 R1Pro 传感器）。
   * - ``omni_config.env.action_frequency`` / ``rendering_frequency`` / ``physics_frequency``
     - 动作 / 渲染 / 物理步进频率（默认 60 / 60 / 120 Hz）。越高越慢。
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
     - 为 ``True`` 时跳过动作 chunk 内中间帧的渲染和观测采集。此时保存的视频只显示策略在 chunk 边界观测到的帧。
   * - ``num_env_subprocess``
     - 将 ``num_envs`` 拆分到多个子进程，每个子进程承载自己的 Isaac/OmniGibson 仿真（见
       ``BehaviorProcess``）。默认 ``1``。**约束：** ``num_envs`` 必须能被 ``num_env_subprocess``
       整除。提高该值可缓解环境步进瓶颈，但会成倍增加进程数与内存。不同流水线阶段共用进程时的限制见 :ref:`behavior-eval-config`。

生成缓存任务实例
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2026 仿真资产已包含官方评测实例，选择方法见 :doc:`BEHAVIOR-1K 评测指南 <../../evaluations/guides/behavior>`。现有 ``rlinf/envs/sim/behavior/instance_generator.py`` 需要 OmniGibson 3.7.2，用于生成额外训练布局。


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
