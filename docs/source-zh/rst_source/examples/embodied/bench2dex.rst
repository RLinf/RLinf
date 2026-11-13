Bench2Dex
=========

使用 RLinf 训练和评测 Bench2Dex Task22、Task64 的 π₀.₅ policy。两项任务支持同步和异步 PPO，Task64 还提供分组 GRPO 配方；训练后的权重可以按官方四通道协议评测。

概览
----

各场景在独立进程中运行，policy 接收批量观测。在配方中设置场景数和 GPU placement。训练采集完整回合，提前结束的场景用零奖励补齐剩余位置。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀.₅

   .. grid-item-card:: 算法
      :text-align: center

      同步 PPO · 异步 PPO · GRPO · 评测

   .. grid-item-card:: 任务
      :text-align: center

      Task22 · Task64

   .. grid-item-card:: 硬件
      :text-align: center

      支持 RGB 渲染的 NVIDIA GPU

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 场景
     - 机器人
     - 主动关节数
     - 控制步数预算
   * - ``22_tool_box_loading``
     - ``multi_panda_with_orca``
     - 40
     - 681
   * - ``64_sports_ball_cup_sort``
     - ``multi_panda_with_allegro``
     - 32
     - 665

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 字段
     - 约定
   * - 观测
     - 双目和腕部 RGB 图像、关节位置、任务描述。
   * - 动作
     - 按所选机器人主动关节顺序排列的绝对位置目标。
   * - 奖励
     - 首次稳定成功获得 1，其余为 0；阶段奖励默认为 0。
   * - Prompt
     - 所选场景中的任务描述。

安装
----

在独立虚拟环境中安装 OpenPI 和场景运行时。安装以下版本：Bench2Dex commit ``fd90dcc625b66c2e535a8b25bcf75bc81c7b080d``\ 、Isaac Sim 5.1.0、Isaac Lab v2.3.2、Python 3.11.14，以及 CUDA 12.8 对应的 torch 2.7.0。

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env bench2dex \
     --python 3.11.14 --platform nvidia --venv .venv-bench2dex-openpi
   source .venv-bench2dex-openpi/bin/activate

激活环境后，``BENCH2DEX_PATH`` 指向上游源码。pip 安装的 Isaac Sim 自行解析运行时路径。独立安装需要设置 ``ISAAC_PATH`` 变量；启动器据此补齐 ``EXP_PATH`` 和 ``CARB_APP_PATH`` 变量，并保留这两个变量的显式设置。

下载资产
--------

从 `Assets <https://huggingface.co/datasets/Bench2Dex/Assets>`_ 下载机器人与场景资产，从 `teleopdata <https://huggingface.co/datasets/Bench2Dex/teleopdata>`_ 下载所选任务的 replay-generalization 锚点。按 `上游文档 <https://bench2dex.github.io/doc/>`_ 解压后，设置资产父目录和锚点目录：

.. code-block:: bash

   export BENCH2DEX_ASSETS_PATH="<asset_parent>"
   export BENCH2DEX_ANCHOR_DIR="<task_anchor_directory>"

将占位符替换为实际目录。资产父目录应包含 ``dex2bench_dataset/`` 和场景引用的 ``Dex2Assets/`` 目录。锚点和 checkpoint 必须来自同一任务。

下载模型
--------

按上游文档下载所选任务的官方 checkpoint，再转换 JAX step 目录和匹配的归一化文件。下面的命令转换 Task22 的 40 维 policy：

.. code-block:: bash

   export BENCH2DEX_PI05_PATH="<converted_checkpoint_directory>"
   JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
     --input-model "<official_step_directory>" \
     --input-norm-stats "<official_step_directory>/assets/dex2bench/norm_stats.json" \
     --output-model "$BENCH2DEX_PI05_PATH" \
     --output-norm-stats "$BENCH2DEX_PI05_PATH/assets/dex2bench/norm_stats.json" \
     --action-dim 40 --action-horizon 20 --max-token-len 280

Task64 使用对应任务的 checkpoint 和锚点，并将转换参数改为 ``--action-dim 32``\ 。两项任务都使用 20 步 action chunk、四个 policy 视角和 280 个 prompt token。仿真还会初始化官方相机配置中的顶视角。

运行
----

使用 PPO 训练 Task22
~~~~~~~~~~~~~~~~~~~~

准备好 Task22 checkpoint 和锚点目录后，启动同步 PPO：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh bench2dex_task22_ppo_openpi_pi05

每轮用 16 个环境采集 8 批 rollout，共 128 条轨迹。每条轨迹安排 700 步动作，物理仿真最多执行 681 步。每轮对采集的数据训练 3 遍。actor 和 critic 的学习率分别为 ``2e-6`` 和 ``2e-5``\ 。

Task22 每完成 2 轮保存一次，每完成 10 轮评测 32 回合；恢复训练后沿用累计轮数。训练和周期评测都采用 ``none`` 通道。在 TensorBoard 查看 ``env/success_once`` 和 ``eval/success_once`` 指标，具体定义见 :doc:`训练指标 <../../reference/metrics>`\ 。

需要同时采样和更新模型时，启动异步配方：

.. code-block:: bash

   bash examples/embodiment/run_async.sh bench2dex_task22_async_ppo_openpi_pi05

训练配方默认启用 actor 和 rollout offload。同步训练在完整 rollout 后释放仿真进程，下一轮重新创建；异步训练保留训练场景。

.. warning::

   环境 offload 要求完整回合和 ``auto_reset=false`` 配置。异步、流水线和 decoupled 执行必须关闭训练环境 offload。

训练每执行完一个 action chunk 采集一次 policy 观测。需要录像、触觉数据、逐动作观测或外部 reward model 时，在对应环境中设置 ``skip_intermediate_renders=false``\ 。

固定场景与恢复训练
~~~~~~~~~~~~~~~~~~

要复用同一场景，将 ``env.train.init_params.anchor_hdf5`` 和 ``env.eval.init_params.anchor_hdf5`` 都指向同一个 HDF5 锚点，并保留 ``none`` 通道。场景几何和外观取自这个锚点，应选择包含目标场景的文件。

用 Python 入口恢复 Task22 训练：

.. code-block:: bash

   export EMBODIED_PATH="$PWD/examples/embodiment"
   PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
   python "$EMBODIED_PATH/train_embodied_agent.py" \
     --config-path "$EMBODIED_PATH/config" \
     --config-name bench2dex_task22_ppo_openpi_pi05 \
     runner.resume_dir="<checkpoint_directory>"

传入完整的 ``global_step_N`` 目录，恢复模型、optimizer 和 scheduler。异步恢复使用 ``train_async.py`` 入口及 ``bench2dex_task22_async_ppo_openpi_pi05`` 配方。

使用 PPO 或 GRPO 训练 Task64
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

切换为 Task64 checkpoint 和锚点，再启动对应的 PPO 配方：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh bench2dex_task64_ppo_openpi_pi05

异步配方名为 ``bench2dex_task64_async_ppo_openpi_pi05``\ 。Task64 训练使用 ``seen`` 资产划分，并随机采样所有场景变化维度。每条轨迹安排 680 步动作，最多执行 665 个控制步。用 ``runner.val_check_interval`` 和 ``runner.save_interval`` 配置周期评测与保存。

需要从同一初始场景比较四次独立尝试时，使用分组 GRPO：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh bench2dex_task64_grpo_openpi_pi05

16 个环境组成 4 组。组内共享锚点和场景种子，动作噪声相互独立。用 ``algorithm.group_size`` 调整组大小，每个环境和 actor shard 都必须接收完整分组。GRPO 根据组内回报计算相对优势，不使用 value head。

评测 checkpoint
~~~~~~~~~~~~~~~

按官方四通道协议评测 Task22，每个通道运行 50 回合：

.. code-block:: bash

   JAX_PLATFORMS=cpu python -m evaluations.bench2dex.run_all_channels \
     bench2dex_task22_openpi_pi05_eval --episodes-per-process 10

Task64 使用 ``bench2dex_task64_openpi_pi05_eval`` 配方及对应任务的资产和权重。启动器按表中的四个通道依次评测。用 ``--gpu-id`` 选择 GPU，``--num-envs`` 设置并行场景数，``--output-root`` 设置结果目录。

评测训练后的权重时，在命令后添加 ``rollout.model.model_path=<checkpoint_directory>/actor/model_state_dict/full_weights.pt`` 和 ``+rollout.model.openpi_data.norm_stats_path=<sft_normalization_file>`` 参数。归一化文件应与原任务匹配。

.. list-table:: 官方通道协议
   :header-rows: 1

   * - 通道
     - 场景准备
     - 种子命名空间
   * - ``none``
     - 恢复锚点场景。
     - ``100000000``
   * - ``cov``
     - 恢复锚点，改变物体位姿和桌高。
     - ``100000000``
   * - ``inv``
     - 恢复锚点，改变外观、杂物和相机。
     - ``100000000``
   * - ``inv_cov``
     - 使用 unseen 资产划分，随机采样全部七个场景变化维度。
     - ``200000000``

并行评测保留官方的回合顺序和随机种子。官方评测在 CPU 上按回合编号和 policy 查询编号生成 JAX 噪声，在 GPU 上用 PyTorch 推理，每个实际动作后采集相机观测。

可视化与结果
------------

四通道输出根目录中的 ``success_rates.tsv`` 汇总成功率。逐回合记录写入 ``per_episode.jsonl`` 文件，任务结果写入 ``per_task.json`` 文件，汇总写入 ``summary.json`` 文件；另保存 RLinf 日志和 TensorBoard events。并行场景的结果按回合编号合并，缺失或重复回合会报错。公共输出目录说明见 :doc:`评测 <../../evaluations/index>`\ 。
