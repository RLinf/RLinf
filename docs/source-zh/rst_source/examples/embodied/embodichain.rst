EmbodiChain 强化学习
====================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/embodichain.gif
   :align: center
   :width: 90%

   EmbodiChain（图片来源：`EmbodiChain <https://github.com/DexForce/EmbodiChain>`__）。

`EmbodiChain <https://github.com/DexForce/EmbodiChain>`__ 是一个通过 Gym 风格接口暴露 RL 任务的具身智能实验室框架。你将使用 RLinf 在 EmbodiChain CartPole 任务上，通过 PPO 训练 MLP actor-critic。

概览
----------------------------------------

在 EmbodiChain CartPole 上训练基于状态的 MLP policy。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      MLP

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      CartPole

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 4 GPUs

| **你将完成：** 安装 → 启动 ``run_embodiment.sh`` → 观察 rollout reward。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · EmbodiChain 包与任务资源。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 任务
     - 描述
   * - CartPole
     - 使用 ``embodichain_tasks/configs/tasks/classic_control/cart_pole/env.json`` 中的状态观测平衡 pole。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 规格
   * - 观测
     - 由 ``state_keys: ["qpos", "qvel", "qf"]`` 构造的单个 ``states`` 张量。
   * - 动作
     - ``policy_setup: cartpole-delta-qpos`` 对应的 2 维连续动作。
   * - 奖励
     - EmbodiChain Gym config 中定义的任务奖励。
   * - 提示词
     - 不使用；这是低维状态控制配方。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: MLP

      **1. 安装环境**

      安装 EmbodiChain 0.3.0 及其依赖。安装脚本从 ``http://pyp.open3dv.site:2345/simple/dexsim-engine/`` 获取 ``dexsim-engine==0.5.0``，其余包通过配置的 Python 包索引解析（机器需要能够访问厂商服务器的 2345 端口）。

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env embodichain && source .venv/bin/activate

      默认使用已安装包中的配置；如需指向本地 EmbodiChain 源码目录，可设置 ``export EMBODICHAIN_PATH=/path/to/EmbodiChain``。

      **2. 下载任务资源（可选）**

      不需要检查点，MLP policy 从头开始训练。如果运行时提示缺少任务资源，在同一个 Python 环境中下载：

      .. code-block:: bash

         export EMBODICHAIN_DATA_ROOT="$PWD/data"
         python -m embodichain.data download --name CartPole
         python -m embodichain.data download --name SimResources

      **3. 启动训练**

      .. list-table::
         :header-rows: 1
         :widths: 28 46 26

         * - 配方
           - 配置
           - 命令后缀
         * - MLP + PPO
           - ``examples/embodiment/config/embodichain_ppo_cart_pole.yaml``
           - ``embodichain_ppo_cart_pole``

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh embodichain_ppo_cart_pole

.. warning::

   EmbodiChain 的 ``dexsim`` 依赖需要 ``libpython3.xx.so``。如果在 UV Python 布局下遇到 ``libpython3.11.so`` 运行时错误，请使用 Conda 环境并重新运行 ``bash requirements/install.sh embodied --env embodichain --no-root``。

.. note::

   将此配方迁移到其他 EmbodiChain 任务时，请保持 ``actor.model.obs_dim``、``actor.model.action_dim`` 和 ``actor.model.policy_setup`` 与任务配置一致。

可视化与结果
----------------------------------------

运行 ``tensorboard --logdir ./logs --port 6006`` 查看 rollout 奖励曲线（默认配置使用 W&B，可通过 ``runner.logger.logger_backends='["tensorboard"]'`` 切换到 TensorBoard），通过 ``env.eval.video_cfg.save_video=True`` 开启评测视频录制，完整指标和日志后端配置见 :doc:`训练指标 <../../reference/metrics>`。

.. dropdown:: 评测与 CI

   EmbodiChain CartPole 也被 embodied e2e 配置覆盖，位于 ``tests/e2e_tests/embodied/``。仅在需要非默认 checkout 时设置 ``EMBODICHAIN_PATH``。
