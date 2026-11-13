Bench2Dex
=========

在 Bench2Dex Task22、Task64 上训练 π₀.₅ policy，并在四个泛化通道上评测成功率。

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

      NVIDIA GPU：训练示例使用 4 张，评测使用 1 张

任务
~~~~

选择要运行的任务，准备对应的模型和资产。

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

policy 根据图像、关节位置和任务描述预测动作。

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

准备好 Task22 模型和资产路径后，启动 PPO 训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh bench2dex_task22_ppo_openpi_pi05

Task22 PPO 在 10 个去噪步骤中，从前 5 步选取一步作为随机步骤。冻结的视觉编码器保持 eval 模式。在训练 YAML 中修改以下配置：

.. code-block:: yaml

   actor:
     model:
       openpi:
         stochastic_step_range: [0, 5]
         frozen_vision_mode: eval

``stochastic_step_range`` 的索引从 0 开始，区间为左闭右开的 ``[start, end)``\ 。设为 ``null`` 可取消这一区间限制。使用 ``frozen_vision_mode: eval`` 时，必须设置 ``train_expert_only: true``\ 。将 ``frozen_vision_mode`` 设为 ``follow_model``，视觉编码器就会跟随 policy 的训练模式。

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

Task64 配置需要对应任务的模型和锚点。在 ``examples/embodiment/config/`` 下对应的 YAML 文件中修改训练参数。GPU 分配见 :doc:`Placement <../../concepts/placement>`，恢复训练见 :doc:`检查点恢复 <../../guides/resume>`\ 。

.. note::

   开启录像时，在对应环境配置中设置 ``skip_intermediate_renders=false``\ 。

评测 policy
~~~~~~~~~~~

在四个通道上评测 Task22，每个通道运行 50 回合：

.. code-block:: bash

   JAX_PLATFORMS=cpu python -m evaluations.bench2dex.run_all_channels \
     bench2dex_task22_openpi_pi05_eval --output-root "<evaluation_output>" \
     "rollout.model.model_path=<model_path>" \
     "+rollout.model.openpi_data.norm_stats_path=$BENCH2DEX_PI05_PATH/assets/dex2bench/norm_stats.json"

将 ``<evaluation_output>`` 替换为结果目录。``<model_path>`` 可以是转换后的模型目录，也可以是训练保存的 ``global_step_N/actor/model_state_dict/full_weights.pt`` 文件。Task64 使用 ``bench2dex_task64_openpi_pi05_eval`` 配置及对应任务的模型和锚点。

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

用 ``--gpu-id`` 选择 GPU，``--num-envs`` 设置并行环境数。

可视化与结果
------------

训练日志和 checkpoint 保存在 ``logs/`` 下。启动 TensorBoard 查看训练进度：

.. code-block:: bash

   tensorboard --logdir logs

训练成功率对应 ``env/success_once`` 指标。评测成功率汇总在 ``<evaluation_output>/success_rates.tsv`` 文件中，各通道的 ``per_episode.jsonl`` 文件记录逐回合结果。指标定义见 :doc:`训练指标 <../../reference/metrics>`，评测用法见 :doc:`评测 <../../evaluations/index>`\ 。
