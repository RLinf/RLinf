基于 Genesis 的强化学习训练
========================================

.. figure:: https://raw.githubusercontent.com/YilingQiao/Genesis/readme-assets/videos/HeroShot_Final.png
   :align: center
   :width: 90%

   Genesis（图片来源：`Genesis <https://genesis-world.readthedocs.io/>`__）。

`Genesis <https://genesis-world.readthedocs.io/>`__ 是面向机器人任务的 GPU 加速多物理场
仿真器。你将使用 RLinf 在 Franka cube-pick 任务上，通过 PPO 训练 MLP 或 CNN policy。

概览
----------------------------------------

训练 Franka Panda 策略在 Genesis 中抓取方块。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      MLP · CNN

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      CubePick

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 1 GPU

| **你将完成：** 安装 → 可选下载 ResNet → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 安装步骤中的 Genesis 依赖。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 任务
     - 描述
   * - ``cube_pick``
     - 控制 Franka Panda 机械臂抓取并抬起方块。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 规格
   * - 观测
     - MLP 使用 16 维状态；CNN 使用 256×256 RGB 加 16 维状态。
   * - 动作
     - 9 维连续动作：7 个 Franka 机械臂关节位置和 2 个夹爪位置。
   * - 奖励
     - 稠密 approach reward 和 grasp-success bonus。
   * - 提示词
     - 不使用；这是低维/CNN policy 控制配方。

安装
----------------------------------------

.. include:: _setup_common.rst

**Docker 镜像**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 32g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-genesis

   # 国内用户可使用：
   # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-genesis

**自定义环境**

安装 Genesis 依赖：

.. code:: bash

   # 国内用户可添加 --use-mirror。
   bash requirements/install.sh embodied --env genesis
   source .venv/bin/activate

下载模型
----------------------------------------

MLP + PPO 配方可跳过本节。CNN + PPO 配方需要下载 ResNet 检查点：

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-ResNet10-pretrained

   # 或使用 huggingface-hub：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-ResNet10-pretrained --local-dir RLinf-ResNet10-pretrained

然后在 ``examples/embodiment/config/genesis_cubepick_ppo_cnn.yaml`` 中为 rollout 和 actor
设置相同的检查点路径：

.. code-block:: yaml

   rollout:
      model:
         model_path: /path/to/RLinf-ResNet10-pretrained
   actor:
      model:
         model_path: /path/to/RLinf-ResNet10-pretrained

运行
----------------------------------------

选择一个配方并启动训练：

.. list-table::
   :header-rows: 1
   :widths: 26 48 26

   * - 配方
     - 配置
     - 命令后缀
   * - MLP + PPO
     - ``examples/embodiment/config/genesis_cubepick_ppo_mlp.yaml``
     - ``genesis_cubepick_ppo_mlp``
   * - CNN + PPO
     - ``examples/embodiment/config/genesis_cubepick_ppo_cnn.yaml``
     - ``genesis_cubepick_ppo_cnn``

.. code:: bash

   bash examples/embodiment/run_embodiment.sh genesis_cubepick_ppo_mlp
   bash examples/embodiment/run_embodiment.sh genesis_cubepick_ppo_cnn

这条命令会：

1. 使用选定的 Hydra 配置启动 embodied 训练入口。
2. 为 actor、rollout 和 Genesis env 组件创建 Ray worker。
3. 运行 PPO rollout，计算 cube-pick 奖励，并更新选定策略。

.. note::

   两个配置默认都运行在 GPU ``0``。请根据硬件调整
   ``cluster.component_placement``、``env.train.total_num_envs`` 和 batch size。

可视化与结果
----------------------------------------

在 RLinf 仓库根目录启动 TensorBoard：

.. code:: bash

   tensorboard --logdir ../results --port 6006

关键指标是 ``env/success_once``。完整指标说明见
:doc:`训练指标 <../../reference/metrics>`。

如需视频，请在环境配置中启用 video：

.. code:: yaml

   env:
     eval:
       video_cfg:
         save_video: True
         video_base_dir: ${runner.logger.log_path}/video/eval

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 配方
     - 结果描述
   * - MLP + PPO
     - 使用默认 ``genesis_cubepick_ppo_mlp`` 参数时，``env/success_once`` 可达到约 80%。
