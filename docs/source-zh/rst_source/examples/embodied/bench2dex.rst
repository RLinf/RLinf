Bench2Dex
=========

Bench2Dex 支持在并行环境中训练和评测 π₀.₅ policy。本文以 Task22 和 Task64 为例，涵盖四个评测通道。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀.₅

   .. grid-item-card:: 算法
      :text-align: center

      PPO · 异步 PPO · GRPO

   .. grid-item-card:: 任务
      :text-align: center

      Task22 · Task64

   .. grid-item-card:: 硬件
      :text-align: center

      Task22：8 张 NVIDIA A800

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 场景
     - 机器人
     - 动作维度
   * - ``22_tool_box_loading``
     - ``multi_panda_with_orca``
     - 40
   * - ``64_sports_ball_cup_sort``
     - ``multi_panda_with_allegro``
     - 32

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 字段
     - 说明
   * - 观测
     - 双目和腕部 RGB 图像，以及关节位置。
   * - 动作
     - 各关节的目标绝对位置。
   * - 奖励
     - 成功为 1，其余为 0。
   * - Prompt
     - 场景中的任务描述。

安装
----

安装 Bench2Dex 和 OpenPI 依赖，然后激活环境：

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env bench2dex \
     --python 3.11.14 --platform nvidia --venv .venv-bench2dex-openpi
   source .venv-bench2dex-openpi/bin/activate

下载资产
--------

从 `Assets <https://huggingface.co/datasets/Bench2Dex/Assets>`_ 下载资产，从 `teleopdata <https://huggingface.co/datasets/Bench2Dex/teleopdata>`_ 下载所选任务的场景锚点。解压后，设置路径：

.. code-block:: bash

   export BENCH2DEX_ASSETS_PATH="<asset_parent>"
   export BENCH2DEX_ANCHOR_DIR="<task_anchor_directory>"

``<asset_parent>`` 是包含 ``dex2bench_dataset/`` 等已解压资产文件夹的父目录。``<task_anchor_directory>`` 是所选任务的 HDF5 锚点目录。

下载模型
--------

从 `Bench2Dex/policy_ckpt <https://huggingface.co/Bench2Dex/policy_ckpt>`_ 下载对应任务的 π₀.₅ checkpoint，再转换为 PyTorch 格式。Task22 使用以下命令：

.. code-block:: bash

   export BENCH2DEX_PI05_PATH="<converted_checkpoint_directory>"
   JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
     --input-model "<official_step_directory>" \
     --input-norm-stats "<official_step_directory>/assets/dex2bench/norm_stats.json" \
     --output-model "$BENCH2DEX_PI05_PATH" \
     --output-norm-stats "$BENCH2DEX_PI05_PATH/assets/dex2bench/norm_stats.json" \
     --action-dim 40 --action-horizon 20 --max-token-len 280

将 ``<official_step_directory>`` 替换为下载的 step 目录，``<converted_checkpoint_directory>`` 替换为输出目录。Task64 使用对应任务的 checkpoint，并将参数改为 ``--action-dim 32``\ 。

运行
----

训练 policy
~~~~~~~~~~~

Task22 的 PPO 配方需要预先完成 10 轮 critic 预热后保存的 ``full_weights.pt``。设置文件路径后启动训练。

.. code-block:: bash

   export BENCH2DEX_WARMUP10_PATH="<warmup10_full_weights.pt>"
   bash examples/embodiment/run_embodiment.sh bench2dex_task22_ppo_openpi_pi05

该配方在固定白底场景中训练 190 轮，每 10 轮评测一次，每 2 轮保存一次 checkpoint。

采样时，从前 5 个去噪步骤中随机选取一步执行 Flow-SDE 更新。

其他训练配置见下表，将命令最后的配置名替换为对应项。异步 PPO 使用 ``examples/embodiment/run_async.sh`` 启动。

.. list-table::
   :header-rows: 1

   * - 任务
     - 算法
     - 配置名
   * - Task22
     - 异步 PPO
     - ``bench2dex_task22_async_ppo_openpi_pi05``
   * - Task64
     - PPO
     - ``bench2dex_task64_ppo_openpi_pi05``
   * - Task64
     - 异步 PPO
     - ``bench2dex_task64_async_ppo_openpi_pi05``
   * - Task64
     - GRPO
     - ``bench2dex_task64_grpo_openpi_pi05``

运行其他 Bench2Dex 任务时，在环境配置中选择场景和机器人，并使用对应的 checkpoint、归一化统计量和场景锚点。动作维度取 policy 控制的关节数，``max_episode_steps`` 取任务的控制步数上限。

在 ``examples/embodiment/config/`` 下对应的 YAML 文件中修改训练参数。GPU 分配见 :doc:`Placement <../../concepts/placement>`，恢复训练见 :doc:`检查点恢复 <../../guides/resume>`\ 。

.. note::

   开启录像时，在对应环境配置中设置 ``skip_intermediate_renders=false``\ 。

评测 policy
~~~~~~~~~~~

在四个通道上评测 Task22，使用八卡，每个通道运行 32 回合：

.. code-block:: bash

   python -m evaluations.bench2dex.run_all_channels \
     bench2dex_task22_openpi_pi05_eval --output-root "<evaluation_output>" \
     --gpu-id 0,1,2,3,4,5,6,7 --num-envs 32 --num-episodes 32 \
     "rollout.model.model_path=<model_path>" \
     "+rollout.model.openpi_data.norm_stats_path=$BENCH2DEX_PI05_PATH/assets/dex2bench/norm_stats.json"

``<model_path>`` 可以是转换后的模型目录，也可以是训练保存的 ``global_step_N/actor/model_state_dict/full_weights.pt`` 文件。Task64 使用 ``bench2dex_task64_openpi_pi05_eval`` 配置及对应任务的模型和锚点。

.. list-table:: 评测通道
   :header-rows: 1

   * - 通道
     - 场景变化
   * - ``none``
     - 原始锚点场景。
   * - ``cov``
     - 物体位姿和桌面高度。
   * - ``inv``
     - 外观、杂物和相机设置。
   * - ``inv_cov``
     - 组合变化，并使用未见过的资产。

可视化与结果
------------

训练日志和 checkpoint 保存在 ``logs/`` 下。启动 TensorBoard 查看训练进度：

.. code-block:: bash

   tensorboard --logdir logs

训练成功率对应 ``env/success_once`` 指标。评测成功率汇总在 ``<evaluation_output>/success_rates.tsv`` 文件中，各通道的 ``per_episode.jsonl`` 文件记录逐回合结果。指标定义见 :doc:`训练指标 <../../reference/metrics>`，评测用法见 :doc:`评测 <../../evaluations/index>`\ 。
