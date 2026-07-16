LIBERO 评测
===========

LIBERO 是基于 robosuite（MuJoCo）的机器人操作仿真基准，涵盖 Spatial、Object、Goal、Long 等任务套件。RLinf 支持在 LIBERO 上并行评测 VLA 策略并输出任务级成功率。

相关训练文档：:doc:`../../examples/embodied/libero`、:ref:`LIBERO-Pro 与 LIBERO-Plus <zh-liberopro-plus-benchmark>`

环境准备
--------

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env libero
   source .venv/bin/activate

使用 ``--env libero`` 时，安装脚本会将 LIBERO clone 到 ``.venv/libero``（若已设置 ``LIBERO_PATH`` 则复用已有目录），并在 ``.venv/bin/activate`` 中将其加入 ``PYTHONPATH``。

支持的模型包括 ``openpi``、``openvla-oft``、``starvla``、``dreamzero``，安装时替换 ``--model`` 参数即可。

示例配置
--------

``evaluations/libero/`` 目录下已有以下示例：

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - 配置文件
     - 任务套件
     - 模型
   * - ``libero_spatial_openpi_pi05_eval.yaml``
     - Spatial
     - π₀.₅
   * - ``libero_spatial_starvla_eval.yaml``
     - Spatial
     - StarVLA
   * - ``libero_spatial_dreamzero_eval.yaml``
     - Spatial
     - DreamZero
   * - ``libero_object_openpi_pi05_eval.yaml``
     - Object
     - π₀.₅
   * - ``libero_object_openvlaoft_eval.yaml``
     - Object
     - OpenVLA-OFT
   * - ``libero_goal_openpi_eval.yaml``
     - Goal
     - π₀
   * - ``libero_goal_openvlaoft_eval.yaml``
     - Goal
     - OpenVLA-OFT
   * - ``libero_10_openpi_pi05_eval.yaml``
     - Long (libero_10)
     - π₀.₅
   * - ``libero_10_openvlaoft_eval.yaml``
     - Long (libero_10)
     - OpenVLA-OFT

完整评测流程
------------

**Step 1：激活环境**

.. code-block:: bash

   source .venv/bin/activate

**Step 2：编辑配置**

复制或编辑目标 YAML，至少修改 ``rollout.model.model_path``。``env.eval`` 字段说明见 :doc:`../reference/configuration` 中的 :ref:`env-eval-fields`；LIBERO 评测协议与套件参数见下文 :ref:`libero-eval-config`。

**Step 3：启动评测**

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model

**Step 4：查看结果**

终端输出 ``eval/success_once``；日志见 :doc:`../reference/results`。

.. _libero-eval-config:

评测配置详解
------------

LIBERO 评测会对任务套件中每个 ``(task_id, trial_id)`` 组合各跑一条轨迹，并汇总 ``eval/success_once`` （轨迹中至少成功一次的比例）。以下字段均在 ``env.eval`` 下配置，共同决定 **并行规模**、**单条轨迹长度** 与 **测试集覆盖范围**。

评测协议概述
~~~~~~~~~~~~

LIBERO 官方为每个任务提供一组固定的初始状态（``task_suite.get_task_init_states(task_id)``，来自 ``.pruned_init`` 文件），
`官方仓库 <https://github.com/Lifelong-Robot-Learning/LIBERO>`_ 的标准评测套件为 **LIBERO-Spatial**、**LIBERO-Object**、**LIBERO-Goal** 与 **LIBERO-Long** （``libero_10``），各含 10 个任务、每任务约 50 个初始状态，完整评测一个套件约 **500** 条轨迹。
RLinf 的 ``evaluations/libero/`` 示例覆盖上述四个 ``task_suite_name``：``libero_spatial``、``libero_object``、``libero_goal``、``libero_10``。

在 RLinf 的 ``LiberoEnv`` 中，每条评测轨迹由 ``(task_id, trial_id)`` 唯一确定：

- ``task_id``：当前 ``task_suite_name`` 下的任务索引（``0 … n_tasks-1``），决定语言指令与 BDDL 场景；
- ``trial_id``：该任务下的初始状态索引，通过 ``get_task_init_states(task_id)[trial_id]`` 加载 MuJoCo 初始位形。

环境内部将各任务的 trial 依次拼接为全局 ``reset_state_id``，再由 ``reset_state_id`` 反解出 ``task_id`` 与 ``trial_id``。
评测模式（``is_eval=True``）下按交错顺序遍历全部 ``reset_state_id``——``(task0, trial0), (task1, trial0), …, (task0, trial1), …``——以保证并行环境下各任务 trial 均匀推进；``auto_reset`` 触发时会按序分配下一个 ``reset_state_id``。

一轮 ``rollout_epoch`` 应尽量覆盖套件内全部 ``(task_id, trial_id)`` 组合。实现上有两种方式：

1. **大并行**：令 ``total_num_envs`` ≥ 总 init state 数，``max_steps_per_rollout_epoch = max_episode_steps``，每个并行环境只跑一条轨迹（见 ``libero_spatial_openpi_pi05_eval.yaml``）。
2. **自动重置**：在 ``auto_reset=True`` 时，episode 结束后环境立即加载下一个 init state；将 ``max_steps_per_rollout_epoch`` 设为 ``max_episode_steps`` 的 **N** 倍，则每轮可评测约 ``N × total_num_envs`` 条轨迹（见 ``libero_spatial_dreamzero_eval.yaml``）。

各套件 ``max_episode_steps`` 参考值
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

各套件最长训练 demo 步数决定了步数上限的参考下限；RLinf 评测 YAML 中的 ``max_episode_steps`` 因模型动作频率不同而有所差异，应不低于参考下限，并与训练配置保持一致：

.. list-table::
   :header-rows: 1
   :widths: 22 22 28 28

   * - 套件
     - 参考下限
     - RLinf 示例值
     - 示例配置
   * - ``libero_spatial``
     - 220
     - 240 / 480
     - ``libero_spatial_openpi_pi05_eval`` / ``libero_spatial_dreamzero_eval``
   * - ``libero_object``
     - 280
     - 280 / 512
     - ``libero_object_openpi_pi05_eval`` / ``libero_object_openvlaoft_eval``
   * - ``libero_goal``
     - 300
     - 320 / 512
     - ``libero_goal_openpi_eval`` / ``libero_goal_openvlaoft_eval``
   * - ``libero_10``
     - 520
     - 520
     - ``libero_10_openpi_pi05_eval`` / ``libero_10_openvlaoft_eval``

``env.eval`` 各字段的详细说明见 :doc:`../reference/configuration` 中的 :ref:`env-eval-fields`。

覆盖完整测试集
~~~~~~~~~~~~~~

设套件总 init state 数为 ``S``，并行环境数为 ``E``，单条轨迹步数上限为 ``T`` （即 ``max_episode_steps``）。

方式一：大并行（``auto_reset`` 可有可无）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   env:
     eval:
       total_num_envs: 500        # S，Spatial / Object / Goal / Long
       max_episode_steps: 240
       max_steps_per_rollout_epoch: 240   # 等于 max_episode_steps
       auto_reset: True           # 可选；E >= S 时影响不大
       rollout_epoch: 1

**方式二：自动重置（资源受限时推荐）**

.. code-block:: yaml

   env:
     eval:
       total_num_envs: 128
       max_episode_steps: 480
       # N = ceil(S / E)；Spatial: ceil(500/128) = 4
       max_steps_per_rollout_epoch: 1920   # N * max_episode_steps = 4 * 480
       auto_reset: True
       ignore_terminations: True
       use_fixed_reset_state_ids: True
       use_ordered_reset_state_ids: True
       rollout_epoch: 1

每轮 ``rollout_epoch`` 评测轨迹数约为 ``N × total_num_envs`` （``N = max_steps_per_rollout_epoch / max_episode_steps``）。例如 Spatial（``S=500``）在 ``E=128`` 时需 ``N = ceil(500/128) = 4``。

**多轮平均**

.. code-block:: yaml

   env:
     eval:
       rollout_epoch: 2           # 相同种子下跑两轮，指标取平均

注意事项
~~~~~~~~

- ``max_steps_per_rollout_epoch`` 必须能被 ``rollout.model.num_action_chunks`` 整除，否则启动时会校验失败。
- 多个 env worker 的种子带有固定偏移 ``seed + rank × stage_num + stage_id``，保证各 worker 分到不同的 init state 子集。
- 终端输出的 ``eval/success_once`` 为所有已完成轨迹的成功率；启用 ``auto_reset`` 时，仅在新 episode 完成时计入指标，避免重复统计。

进阶用法
--------

**LIBERO-PRO**

.. code-block:: bash

   export LIBERO_TYPE=pro
   export LIBERO_PERTURBATION=all
   bash evaluations/run_eval.sh libero libero_10_openvlaoft_eval

**LIBERO-PLUS**

.. code-block:: bash

   export LIBERO_TYPE=plus
   export LIBERO_SUFFIX=all
   bash evaluations/run_eval.sh libero libero_10_openvlaoft_eval

**调整并行规模**

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval \
     env.eval.total_num_envs=64 \
     rollout.model.model_path=/path/to/model

常见问题
--------

- **渲染问题：** 若 headless 环境报错，尝试 ``export MUJOCO_GL=osmesa`` 与 ``export PYOPENGL_PLATFORM=osmesa`` （``run_eval.sh`` 默认已设置）。
- **评测覆盖范围：** 见上文 :ref:`libero-eval-config`；核心是 ``total_num_envs``、``auto_reset`` 与 ``max_steps_per_rollout_epoch`` 三者的配合。
