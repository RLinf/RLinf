:orphan:

LingBot-VA 在 RoboTwin 上的评测
==================================

本文档介绍如何在 RLinf 中对官方 LingBot-VA 的 RoboTwin checkpoint 执行评测。

当前接入面向 RoboTwin evaluation，并保留了官方 LingBot-VA 的推理流程，包括 16D end-effector action、``add_init_pose``、key-frame 更新，以及 chunk 之间的 ``compute_kv_cache``。

概览
----

LingBot-VA 当前在 RLinf 中仅支持 **RoboTwin 评测**。

当前接入具有以下特征：

* **环境**：通过 RLinf ``RoboTwinEnv`` 执行 RoboTwin 2.0 任务。
* **观测**：头部 RGB 图像、左右腕部 RGB 图像，以及本体状态。
* **动作空间**：16 维 end-effector 动作。
* **规划后端**：``curobo``。
* **执行类型**：``action_type: ee``。
* **当前支持**：当前支持通过 RLinf rollout worker 执行 evaluation。SFT 入口也已接入。

已验证配置固定为：

* ``runner.only_eval=True``
* ``total_num_envs`` 可按任务和可用硬件资源配置。
* ``enable_offload: false``

依赖安装
--------

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装 LingBot-VA 与 RoboTwin 支持
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash requirements/install.sh embodied --venv .venv-lingbotva --model lingbotva --env robotwin
   source .venv-lingbotva/bin/activate

请为 LingBot-VA 单独创建并使用一个 venv。安装脚本会把官方 LingBot-VA 仓库克隆到 ``.venv-lingbotva/lingbot-va``，
并自动把 ``LINGBOT_VA_REPO_PATH`` 写入 ``.venv-lingbotva/bin/activate``。

3. 准备 RoboTwin 资产
~~~~~~~~~~~~~~~~~~~~~

请使用 RoboTwin 的 ``RLinf_support`` 分支，并下载对应资产：

.. code-block:: bash

   git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support
   cd RoboTwin
   bash script/_download_assets.sh

4. 下载 LingBot-VA checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash requirements/embodied/download_assets.sh --dir /path/to/assets --assets lingbotva

随后将 ``LINGBOT_VA_MODEL_PATH`` 指向：

.. code-block:: bash

   /path/to/assets/.cache/lingbotva/lingbot-va-posttrain-robotwin

这个路径应当指向由 ``download_assets.sh`` 下载得到的 **checkpoint 根目录**。
具体来说，RLinf 期望下面这个文件存在：

.. code-block:: bash

   ${LINGBOT_VA_MODEL_PATH}/transformer/config.json

5. 评测前检查 ``attn_mode``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

评测前请打开 ``<model>/transformer/config.json``，确认 ``attn_mode`` 为 ``"torch"`` 或 ``"flashattn"``。

配置文件
--------

当前 RoboTwin 评测配置包括：

* ``examples/embodiment/config/robotwin_click_bell_eval_lingbotva.yaml``
* ``examples/embodiment/config/robotwin_place_empty_cup_eval_lingbotva.yaml``

这些配置保持了已验证的设置：

* ``planner_backend: curobo``
* ``action_type: ee``
* ``action_dim: 16``
* ``num_action_chunks: 32``
* ``ignore_terminations: false``
* ``enable_offload: false``

环境变量
--------

.. code-block:: bash

   export ROBOTWIN_PATH=/path/to/RoboTwin
   export LINGBOT_VA_MODEL_PATH=/path/to/assets/.cache/lingbotva/lingbot-va-posttrain-robotwin
   export LINGBOT_VA_REPO_PATH=/path/to/RLinf/.venv-lingbotva/lingbot-va
   export ROBOT_PLATFORM=ALOHA

这些变量的含义如下：

* ``ROBOTWIN_PATH``：本地 RoboTwin 仓库根目录。
* ``LINGBOT_VA_MODEL_PATH``：由
  ``requirements/embodied/download_assets.sh`` 下载得到的 LingBot-VA checkpoint 根目录。
* ``LINGBOT_VA_REPO_PATH``：RLinf 运行时使用的官方 LingBot-VA 仓库根目录。
  如果你是通过
  ``bash requirements/install.sh embodied --venv .venv-lingbotva --model lingbotva --env robotwin``
  安装的，通常它就是 ``<你的_RLinf_仓库>/.venv-lingbotva/lingbot-va``。

可选覆盖项：

.. code-block:: bash

   # 仅在直接调用 eval_embodied_agent.py 时需要手动设置。
   export REPO_PATH=/path/to/RLinf

启动命令
--------

推荐命令：

.. code-block:: bash

   bash examples/embodiment/eval_embodiment.sh robotwin_click_bell_eval_lingbotva

如果你直接调用 ``eval_embodied_agent.py``，请先设置 ``REPO_PATH``，然后运行：

.. code-block:: bash

   python examples/embodiment/eval_embodied_agent.py \
     --config-path examples/embodiment/config \
     --config-name robotwin_click_bell_eval_lingbotva

评测结果与可视化
----------------

当前已验证的任务包括：

* ``click_bell``
* ``place_empty_cup``

如果启用了 TensorBoard，可以使用：

.. code-block:: bash

   tensorboard --logdir ../results --port 6006

如果启用了评测视频记录，视频会保存在：

.. code-block:: bash

   <runner.logger.log_path>/video/eval

说明
----

* ``center_crop`` 需要按任务显式配置。
* ``max_episode_steps`` 与 ``max_steps_per_rollout_epoch`` 需要和 RLinf 的 chunk rollout 调度保持一致。
* 当前接入保留了官方 LingBot-VA 的 chunk/key-frame/KV-cache 评测节奏。
