RoboUniverse-10K 评估
=====================

.. figure:: https://raw.githubusercontent.com/Einsia/RoboUniverse-10K/main/assets/task-taxonomy.png
   :align: center
   :width: 90%
   :alt: RoboUniverse-10K 任务分类

   RoboUniverse-10K 在 Franka Panda 和 AgileX 双臂 Piper 上的任务分类（来源：`Einsia/RoboUniverse-10K <https://github.com/Einsia/RoboUniverse-10K>`__）。

在 RoboUniverse-10K 的 10,807 个 Isaac Sim 操作任务上评估 π₀.₅ / OpenPI policy。

概览
----

使用两路 RGB 图像、关节状态和自然语言指令评估具身 policy。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀.₅ / OpenPI

   .. grid-item-card:: 算法
      :text-align: center

      仅评估

   .. grid-item-card:: 任务
      :text-align: center

      10,807 个任务（``heldout500`` 含 500 个）

   .. grid-item-card:: 硬件
      :text-align: center

      Linux x86_64 · NVIDIA RTX GPU（Isaac Sim 5.1.0）

| **你将完成：** 安装 → 下载数据集 → 准备模型 → 配置 EULA 与路径 → 启动评估 → 查看 ``eval/success_once``\ 。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 阅读 `NVIDIA Isaac Sim 许可协议 <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/NVIDIA_Omniverse_License_Agreement.html>`_。

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 机器人平台
     - 任务数量
     - 环境配置
     - 动作空间
   * - Franka Panda
     - 共 4,132 个（``heldout500`` 含 200 个）
     - ``env/robouniverse_franka``
     - 9 维关节位置目标（7 维机械臂 + 2 维夹爪）
   * - AgileX 双臂 Piper
     - 共 6,675 个（``heldout500`` 含 300 个）
     - ``env/robouniverse_agilex``
     - 16 维关节位置目标（双臂各 6 维 + 双夹爪各 2 维）

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 字段
     - 格式
   * - Observation
     - 外部相机与腕部相机两路 640×480 RGB 图像、实测关节位置（Franka 为 9 维，AgileX 为 16 维）以及可选关节速度。
   * - Action
     - 关节位置目标，按各任务 ``env_spec.json`` 运行（Franka 为 20 Hz / 0.05 s，AgileX 为 10 Hz / 0.1 s），并自动裁剪到关节限位内。
   * - Reward
     - 回合结束给出稀疏奖励（成功为 1，失败为 0）。
   * - Prompt
     - 每个任务自带的自然语言指令。

安装
----

在已安装 Isaac Sim 5.1.0 的 Linux x86_64 主机或 ``nvcr.io/nvidia/isaac-sim:5.1.0`` 容器中运行 RLinf。

.. include:: _setup_common.rst
   :end-before: 然后，使用下列两种方式之一准备依赖

安装支持 OpenPI 与 RoboUniverse-10K 的 RLinf 环境：

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env robouniverse
   source .venv/bin/activate

下载数据集
----------

从 Hugging Face 下载任务索引和 ``heldout500`` 划分文件到 ``/path/to/RoboUniverse-10K``\ ：

.. code-block:: bash

   hf download Einsia/RoboUniverse-10K catalog.csv splits/heldout500.csv \
     --repo-type dataset --revision ae170e8093dd6061765c14f2fb6a209432517d85 \
     --local-dir /path/to/RoboUniverse-10K

将首个 Franka ``heldout500`` 冒烟测试任务或全部 ``tasks/`` 目录下的任务压缩包下载到同一目录：

.. code-block:: bash

   # 下载首个 Franka heldout500 任务（冒烟测试）
   hf download Einsia/RoboUniverse-10K tasks/franka/000/000095_rotate_twist.zip \
     --repo-type dataset --revision ae170e8093dd6061765c14f2fb6a209432517d85 \
     --local-dir /path/to/RoboUniverse-10K

   # 或下载 catalog.csv 中的全部任务压缩包
   hf download Einsia/RoboUniverse-10K --include "tasks/**" \
     --repo-type dataset --revision ae170e8093dd6061765c14f2fb6a209432517d85 \
     --local-dir /path/to/RoboUniverse-10K

准备模型
--------

提供针对目标机器人关节空间微调好的 PyTorch OpenPI π₀.₅ checkpoint 及其归一化统计文件：

- **Franka Panda（9 维动作）：** 将权重放到 ``/path/to/openpi_robouniverse_franka_checkpoint``\ ，``norm_stats.json`` 放在 ``<model_path>/robouniverse_franka/norm_stats.json``\ （对应 ``openpi.config_name: pi05_robouniverse_franka``\ ）。
- **AgileX 双臂 Piper（16 维动作）：** 将权重放到 ``/path/to/openpi_robouniverse_agilex_checkpoint``\ ，``norm_stats.json`` 放在 ``<model_path>/robouniverse_agilex/norm_stats.json``\ （对应 ``openpi.config_name: pi05_robouniverse_agilex``\ ）。

如果 ``norm_stats.json`` 位于其他位置，可通过 ``rollout.model.openpi_data.norm_stats_path=/path/to/norm_stats.json`` 覆盖。通用的 OpenPI SFT 训练与权重转换流程见 :doc:`OpenPI 监督微调 </rst_source/examples/embodied/sft_openpi>`。

运行
----

编辑 ``evaluations/robouniverse/robouniverse_franka_eval_openpi.yaml``\ ，填写数据集目录、checkpoint 路径、Isaac Sim Python 路径（容器内为 ``/isaac-sim/python.sh``\ ，主机安装填写对应 ``python.sh`` 路径）并同意 NVIDIA 许可协议：

.. code-block:: yaml

   env:
     eval:
       init_params:
         data_root: /path/to/RoboUniverse-10K
         sim_python: /isaac-sim/python.sh
         accept_isaac_eula: true

   rollout:
     model:
       model_path: /path/to/openpi_robouniverse_franka_checkpoint

启动 Franka 评估：

.. code-block:: bash

   bash evaluations/run_eval.sh robouniverse robouniverse_franka_eval_openpi

默认配置会从 ``heldout500`` 评估 1 个回合（``rollout_epoch: 1``\ 、``total_num_envs: 1``\ ）。可通过覆盖 ``env.eval.rollout_epoch`` 或 ``env.eval.total_num_envs`` 扩展评估回合数，或指定具体任务列表：

.. code-block:: bash

   bash evaluations/run_eval.sh robouniverse robouniverse_franka_eval_openpi \
     'env.eval.init_params.cases=["000095_rotate_twist"]'

评估 AgileX 双臂任务时，在 ``evaluations/robouniverse/robouniverse_agilex_eval_openpi.yaml`` 中填写 ``data_root``\ 、``sim_python``\ 、``accept_isaac_eula: true`` 和 ``model_path`` 后运行：

.. code-block:: bash

   bash evaluations/run_eval.sh robouniverse robouniverse_agilex_eval_openpi

.. warning::

   每个模拟器 worker 至少分配 8 核 CPU 和 24 GiB 内存。核心数不足会导致 Isaac Sim 冷启动超时。

可视化与结果
------------

在 ``logs/<timestamp>-<config_name>/`` 和 TensorBoard 中查看评估结果：

- ``eval/success_once`` 和 ``eval/return`` 对应 :doc:`训练指标 </rst_source/reference/metrics>` 中的 ``env/success_once`` 与 ``env/return``\ 。
- ``eval/infra_error`` 统计因模拟器或任务评测器报错而提前中止的回合占比，与 policy 任务失败分开统计。

日志后端与输出配置见 :doc:`训练可视化 </rst_source/guides/logger>`。
