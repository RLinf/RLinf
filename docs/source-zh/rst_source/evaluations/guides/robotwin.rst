RoboTwin 评测
=============

RoboTwin 是双臂操作仿真平台，提供放置、调整、点击等多种桌面操作任务。RLinf 支持在 RoboTwin 上并行评测 VLA 策略，并输出 ``eval/success_once`` 等指标。

相关训练文档：:doc:`../../examples/embodied/robotwin`

环境准备
--------

**安装依赖**

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla-oft --env robotwin
   source .venv/bin/activate

支持的模型包括 ``openvla-oft``、``openpi``、``lingbotvla``，安装时替换 ``--model`` 参数即可。

**RoboTwin 仓库与 Assets**

评测前需克隆 RLinf 适配分支并下载仿真资产（详见训练文档）：

.. code-block:: bash

   # 1. 克隆 RoboTwin 仓库（须使用 RLinf_support 分支）
   git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support
   cd RoboTwin

   # 2. 下载并解压 Assets
   bash script/_download_assets.sh

下载完成后，将评测 YAML 中的 ``env.eval.assets_path`` 指向解压后的 assets 目录。

**环境变量**

每次评测前须设置：

.. code-block:: bash

   export ROBOTWIN_PATH=/path/to/RoboTwin
   export ROBOT_PLATFORM=ALOHA

``run_eval.sh`` 会将 ``ROBOTWIN_PATH`` 加入 ``PYTHONPATH``；环境初始化时还会将 ``assets_path`` 写入 ``ASSETS_PATH``。

**Docker（可选）**

也可使用官方 Docker 镜像 ``rlinf/rlinf:agentic-rlinf0.3-robotwin`` 运行评测，镜像已包含 RoboTwin 依赖与兼容性补丁。进入容器后按模型类型切换环境：

- OpenVLA-OFT：``source switch_env openvla-oft``
- OpenPI（π\ :sub:`0`\ / π\ :sub:`0.5`\ ）：``source switch_env OpenPI``

示例配置
--------

``evaluations/robotwin/`` 目录下已有以下示例：

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - 配置文件
     - 任务
     - 模型
   * - ``robotwin_place_empty_cup_openvlaoft_eval.yaml``
     - place_empty_cup
     - OpenVLA-OFT
   * - ``robotwin_place_empty_cup_openpi_eval.yaml``
     - place_empty_cup
     - π₀
   * - ``robotwin_adjust_bottle_openpi_eval.yaml``
     - adjust_bottle
     - π₀
   * - ``robotwin_adjust_bottle_openpi_pi05_eval.yaml``
     - adjust_bottle
     - π₀.₅
   * - ``robotwin_place_shoe_lingbotvla_eval.yaml``
     - place_shoe
     - LingBotVLA
   * - ``robotwin_click_bell_lingbotvla_eval.yaml``
     - click_bell
     - LingBotVLA

若 ``evaluations/robotwin/<config>.yaml`` 不存在，``run_eval.sh`` 会回退到 ``examples/embodiment/config/`` 下同名配置（需设置 ``runner.only_eval: True`` 与 ``runner.task_type: embodied_eval``）。``rlinf/envs/robotwin/seeds/eval_seeds.json`` 中另有 **22 个任务** 的评测种子，其余任务可从训练配置派生评测 YAML（见 :doc:`../reference/configuration`）。

完整评测流程
------------

**Step 1：激活环境并设置路径**

.. code-block:: bash

   source .venv/bin/activate
   export ROBOTWIN_PATH=/path/to/RoboTwin
   export ROBOT_PLATFORM=ALOHA

**Step 2：准备模型**

推荐预训练权重示例：

- OpenVLA-OFT：`RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup>`_
- π\ :sub:`0`：`RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle <https://huggingface.co/RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle>`_
- π\ :sub:`0.5`：`RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle <https://huggingface.co/RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle>`_

下载命令见训练文档「模型下载」一节。

**Step 3：编辑配置**

复制或编辑目标 YAML，至少修改 ``rollout.model.model_path`` 与 ``env.eval.assets_path``。通用 ``env.eval`` 字段见 :doc:`../reference/configuration` 中的 :ref:`env-eval-fields`；RoboTwin 评测协议与模型差异见下文 :ref:`robotwin-eval-config`。

**Step 4：启动评测**

.. code-block:: bash

   bash evaluations/run_eval.sh robotwin robotwin_place_empty_cup_openvlaoft_eval \
     rollout.model.model_path=/path/to/model \
     env.eval.assets_path=/path/to/robotwin_assets

**Step 5：查看结果**

终端输出 ``eval/success_once``；日志与视频见 :doc:`../reference/results`。

.. _robotwin-eval-config:

评测配置详解
------------

RoboTwin 评测对 ``eval_seeds.json`` 中每个任务的 **success seed** 各跑一条轨迹，并汇总 ``eval/success_once`` （轨迹中至少成功一次的比例）。以下字段均在 ``env.eval`` 下配置，共同决定 **并行规模**、**单条轨迹长度** 与 **测试集覆盖范围**。

评测协议概述
~~~~~~~~~~~~

RoboTwin 评测使用预筛选的 **success seeds** 作为每条轨迹的随机种子，以固定初始场景与语言指令。种子列表位于 ``rlinf/envs/robotwin/seeds/eval_seeds.json``，按 ``task_name`` 索引；当前文件覆盖 **22 个任务** （150–320 条种子不等）。

在 ``RoboTwinEnv`` 中：

- 启动时从 ``seeds_path`` 加载对应任务的 ``success_seeds``，经全局 shuffle 后按 worker 切分，保证各 env worker 分到不重叠的子集；
- 每条轨迹由分配到的 **seed** 唯一确定初始场景与语言指令；
- ``is_eval: True`` 时，``auto_reset`` 触发后会为已完成的环境分配下一个 seed（当 ``use_fixed_reset_state_ids: False`` 时）。

默认示例配置（``total_num_envs: 128``、``rollout_epoch: 1``、``use_fixed_reset_state_ids: True``）下，每个并行环境在一轮内只评测一条固定种子的轨迹。以 8 GPU（``component_placement: 0-7``）为例，一轮约评测 128 条轨迹，不一定覆盖该任务的全部 seeds。

各示例任务 seeds 数量与步数上限
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 22 22 28

   * - 任务
     - seeds 总数
     - 示例 ``max_episode_steps``
     - 示例配置
   * - ``adjust_bottle``
     - 150
     - 200
     - ``robotwin_adjust_bottle_openpi_eval``
   * - ``place_empty_cup``
     - 260
     - 200
     - ``robotwin_place_empty_cup_openvlaoft_eval``
   * - ``click_bell``
     - 150
     - 400
     - ``robotwin_click_bell_lingbotvla_eval``
   * - ``place_shoe``
     - 320
     - 400
     - ``robotwin_place_shoe_lingbotvla_eval``

``max_episode_steps`` 应与训练配置及 ``task_config.step_lim`` 保持一致；LingBotVLA 示例任务通常使用 400 步，OpenVLA-OFT / OpenPI 示例多为 200 步。

``env.eval`` 各字段的详细说明见 :doc:`../reference/configuration` 中的 :ref:`env-eval-fields`。

模型专属配置
~~~~~~~~~~~~

不同 VLA 在 RoboTwin 上使用不同的机器人 embodiment、相机与域随机化设置，评测时必须与训练协议一致，否则结果不可比：

OpenVLA-OFT（demo_randomized 协议）

- 沿用 env preset 默认：``task_config.embodiment: [piper, piper, 0.6]``
- ``center_crop: True``；模型侧设置 ``rollout.model.center_crop: True``
- 域随机化保持开启（训练 preset 默认）
- ``rollout.model.num_action_chunks: 25``；``unnorm_key`` 须与 SFT 一致，如 ``place_empty_cup_1k``
- ``rollout.model.implement_version: "official"``

OpenPI（π\ :sub:`0`\ / π\ :sub:`0.5`\ ，demo_clean 协议）

- ``task_config.embodiment: [aloha-agilex]``
- ``center_crop: False``
- ``task_config.camera.collect_wrist_camera: true``
- ``task_config.domain_randomization`` 全部关闭：``random_background``、``cluttered_table``、``random_light`` 等设为 ``false``
- ``rollout.model.num_action_chunks: 50``
- ``rollout.model.openpi.config_name``：``pi0_aloha_robotwin`` 或 ``pi05_aloha_robotwin``
- 建议 ``env.enable_offload: True``、``rollout.enable_offload: True`` 以降低显存占用

LingBotVLA

- 除 ``rollout.model.model_path`` 外，还须配置 ``tokenizer_path`` 与 ``rollout.model.lingbotvla.config_path``
- ``rollout.model.num_action_chunks: 50``；``max_episode_steps: 400`` （``click_bell``、``place_shoe`` 等）
- ``use_custom_reward: False`` （须关闭自定义奖励）

覆盖完整测试集
~~~~~~~~~~~~~~

设任务 seeds 总数为 ``S``，并行环境数为 ``E``，单条轨迹步数上限为 ``T`` （即 ``max_episode_steps``）。

**方式一：大并行**

.. code-block:: yaml

   env:
     eval:
       total_num_envs: 260        # S，以 place_empty_cup 为例
       max_episode_steps: 200
       max_steps_per_rollout_epoch: 200   # 等于 max_episode_steps
       use_fixed_reset_state_ids: True
       rollout_epoch: 1

**方式二：动态种子与自动重置（资源受限时推荐）**

.. code-block:: yaml

   env:
     eval:
       total_num_envs: 128
       max_episode_steps: 200
       # N = ceil(S / E)；place_empty_cup: ceil(260/128) = 3
       max_steps_per_rollout_epoch: 600   # N * max_episode_steps = 3 * 200
       auto_reset: True
       ignore_terminations: True
       use_fixed_reset_state_ids: False   # 允许 auto_reset 时切换 seed
       is_eval: True
       rollout_epoch: 1

**多轮平均**

.. code-block:: yaml

   env:
     eval:
       rollout_epoch: 2
       use_fixed_reset_state_ids: False   # rollout_epoch > 1 时须设为 False

注意事项
~~~~~~~~

- ``max_steps_per_rollout_epoch`` 必须能被 ``rollout.model.num_action_chunks`` 整除，否则启动时会校验失败。
- ``env.eval.seeds_path`` 默认指向 ``eval_seeds.json``；自定义种子文件须包含对应 ``task_name`` 的 ``success_seeds`` 列表。
- OpenVLA-OFT 在 **demo_randomized** 下训练/评测，OpenPI 在 **demo_clean** 下训练/评测，混用域随机化设置会导致指标不可比。
- 以下任务尚未支持：``place_fan``、``open_laptop``、``place_object_scale``、``put_object_cabinet``。

进阶用法
--------

**调整并行规模**

.. code-block:: bash

   bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_eval \
     env.eval.total_num_envs=64 \
     rollout.model.model_path=/path/to/model

**从训练配置派生其他任务**

``eval_seeds.json`` 中另有 ``beat_block_hammer``、``handover_block``、``lift_pot`` 等任务的种子。可复制 ``evaluations/robotwin/`` 中结构相近的 YAML，将 ``defaults`` 中的 env preset 改为 ``env/robotwin_<task>@env.eval`` （对应 ``examples/embodiment/config/env/`` 下的 preset），并调整 ``rollout.model`` 与任务相关的 ``unnorm_key`` / ``openpi.config_name``。

**加载 RL  checkpoint**

.. code-block:: bash

   bash evaluations/run_eval.sh robotwin robotwin_place_empty_cup_openvlaoft_eval \
     runner.ckpt_path=/path/to/checkpoint.pt

常见问题
--------

- **ROBOTWIN_PATH 未设置：** ``run_eval.sh`` 会将其加入 ``PYTHONPATH``，但必须指向有效的 RoboTwin 仓库根目录（``RLinf_support`` 分支）。
- **assets_path 错误：** 环境通过 ``ASSETS_PATH`` 加载仿真资产；路径无效会导致启动失败或场景缺失。
- **机器人平台：** 通过 ``ROBOT_PLATFORM=ALOHA`` 选择平台变体。
- **GPU 显存不足：** 在 YAML 中设置 ``env.enable_offload: True`` 与 ``rollout.enable_offload: True``；或减小 ``env.eval.total_num_envs``。
- **评测覆盖范围：** 见上文 :ref:`robotwin-eval-config`；默认 128 并行 + ``use_fixed_reset_state_ids: True`` 只覆盖部分 seeds。
- **渲染问题：** 若 headless 环境报错，可尝试 ``export MUJOCO_GL=osmesa`` 与 ``export PYOPENGL_PLATFORM=osmesa`` （``run_eval.sh`` 默认已设置）。
