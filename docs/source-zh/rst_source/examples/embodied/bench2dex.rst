Bench2Dex
=========

使用 RLinf 同步或异步 PPO 训练 Bench2Dex Task64 的 π₀.₅ policy，机器人为双 Panda＋Allegro。先准备场景资产和转换后的 checkpoint，再启动训练，并通过官方四个泛化通道评测 checkpoint。

概览
----

Task64 配置支持每个环境 worker 管理一个或多个独立场景进程。``Bench2DexVectorEnv`` 继承 RLinf 的 ``BaseVectorEnv``，调度完整 action chunk；rollout worker 接收批量观测。仿真环境注册为 ``env_type: bench2dex``。

每个 ``Bench2DexSubprocEnvWorker`` 通过局部 ``spawn`` context 启动一个场景，子进程中的 ``Bench2DexRuntime`` 创建并关闭该场景的 ``SimulationApp``。vector env 先向所有活动场景发送完整 chunk，再收集结果。各场景按顺序执行动作，每个实际执行的动作后采集观测；外层环境合并批量观测，并补齐提前结束的场景。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀.₅

   .. grid-item-card:: 算法
      :text-align: center

      同步 PPO · 异步 PPO · policy 评测

   .. grid-item-card:: 任务
      :text-align: center

      Task64 · 双 Panda＋Allegro

   .. grid-item-card:: 硬件
      :text-align: center

      NVIDIA GPU

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 场景
     - 机器人
     - 目标
   * - ``64_sports_ball_cup_sort``
     - ``multi_panda_with_allegro``
     - 将四个球分类放入两个杯子。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 字段
     - 约定
   * - 观测
     - 双目和两个腕部 RGB 图像、关节位置、任务描述。
   * - 动作
     - 按所选机器人关节顺序排列的主动关节绝对位置目标。
   * - 奖励
     - 首次达到稳定成功时为 1，其余为 0。
   * - Prompt
     - 所选场景 YAML 中的任务描述。

安装
----

在独立虚拟环境中安装 OpenPI 和场景运行时。配置固定使用 Bench2Dex ``fd90dcc625b66c2e535a8b25bcf75bc81c7b080d``、Isaac Sim 5.1.0、Isaac Lab v2.3.2、Python 3.11.14，以及 CUDA 12.8 对应的 torch 2.7.0。

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env bench2dex \
     --python 3.11.14 --platform nvidia --venv .venv-bench2dex-openpi
   source .venv-bench2dex-openpi/bin/activate

激活环境后，``BENCH2DEX_PATH`` 指向上游源码。pip 安装的 Isaac Sim 自行解析运行时路径；使用独立安装时，可明确设置 ``ISAAC_PATH``，启动脚本据此补齐 ``apps/`` 和 ``kit/`` 路径。

准备资产
--------

从 `Assets <https://huggingface.co/datasets/Bench2Dex/Assets>`_ 准备机器人与场景资产，从 `teleopdata <https://huggingface.co/datasets/Bench2Dex/teleopdata>`_ 准备 Task64 replay-generalization 锚点。按 `上游文档 <https://bench2dex.github.io/doc/>`_ 解压后设置目录：

.. code-block:: bash

   export BENCH2DEX_ASSETS_PATH=/path/to/bench2dex-data
   export BENCH2DEX_ANCHOR_DIR=/path/to/task64/replay-generalization

资产父目录包含 ``dex2bench_dataset/`` 和场景引用的 ``Dex2Assets/``。锚点通道恢复 HDF5 中的场景，再读取实时传感器观测。

下载模型
--------

按上游文档下载官方 Task64 π₀.₅ checkpoint，再将包含 ``params/`` 及匹配的 ``assets/dex2bench/norm_stats.json`` 的 JAX step 目录转换为 RLinf PyTorch 格式：

.. code-block:: bash

   export BENCH2DEX_PI05_PATH=/path/to/task64-pi05-rlinf
   JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
     --input-model /path/to/official-pi05-step \
     --input-norm-stats /path/to/official-pi05-step/assets/dex2bench/norm_stats.json \
     --output-model "$BENCH2DEX_PI05_PATH" \
     --output-norm-stats "$BENCH2DEX_PI05_PATH/assets/dex2bench/norm_stats.json" \
     --action-dim 32 --action-horizon 20 --max-token-len 280

评测配置读取转换后的 ``model.safetensors`` 和归一化文件。Task64 使用四个 policy 视角、32 维主动关节动作、20 步 action chunk 和 280 个 prompt tokens。对齐版评测创建包含顶视角的官方五相机配置，并在首次 policy 观测前采集预热帧。

启动
----

PPO 训练
~~~~~~~~

两份训练配方使用相同的 Task64 checkpoint 和归一化文件。训练在 ``seen`` 资产池中重采样全部场景轴，使用独立的种子序列；官方评测锚点和 ``unseen`` split 单独配置。

启动同步 PPO：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh bench2dex_task64_ppo_openpi_pi05

该命令使用 ``EmbodiedRunner``，依次完成 rollout 收集、actor 与 value head 更新、权重同步，再进入下一轮采样。模型采用 Flow-SDE 探索，在 GPU 上通过 PyTorch 生成噪声，训练 action expert 和 value head。

需要让采样与参数更新重叠时，启动异步配方：

.. code-block:: bash

   bash examples/embodiment/run_async.sh bench2dex_task64_async_ppo_openpi_pi05

该命令使用 ``AsyncPPOEmbodiedRunner`` 和已有的 ``decoupled_actor_critic`` loss。异步 YAML 继承同步配方，模型、场景分布、采样预算和优化器参数保持一致。``algorithm.staleness_threshold: 1`` 允许采样权重落后一个版本，``algorithm.behave_weight_threshold`` 设为 ``2.0``。

两份配方初始配置均为 16 个环境，每个训练 epoch 收集一次 rollout。每回合预算为 665 个控制步，调度为 34 个、每个 20 步的 action chunk。在 YAML 中通过 ``env.train.total_num_envs`` 和 ``cluster.component_placement`` 调整场景及 worker 数量。actor 初始 microbatch 为 1，global batch 为 136，将 544 个 chunk 样本分成四个 minibatch。

训练环境使用 ``auto_reset=false``。回合在 chunk 内结束时，保留实际执行动作的奖励；后续 chunk 保持终态观测、奖励为零，由 RLinf 已有的 done-based loss mask 排除。同步和异步 PPO 均使用 chunk-level reward 与 log-probability。

runner 按 ``runner.save_interval`` 保存 checkpoint。恢复同步训练时，通过 Python 入口设置 ``runner.resume_dir``：

.. code-block:: bash

   export EMBODIED_PATH="$PWD/examples/embodiment"
   PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
   python "$EMBODIED_PATH/train_embodied_agent.py" \
     --config-path "$EMBODIED_PATH/config" \
     --config-name bench2dex_task64_ppo_openpi_pi05 \
     runner.resume_dir=/path/to/checkpoints/global_step_40

恢复异步训练时，使用 ``train_async.py`` 和 ``bench2dex_task64_async_ppo_openpi_pi05``，传入相同的 ``runner.resume_dir`` 参数。两份配方均通过下面的独立入口评测 checkpoint。训练时关注 ``env/success_once``，其余指标定义见 :doc:`训练指标 <../../reference/metrics>`。

Checkpoint 评测
~~~~~~~~~~~~~~

每通道运行 50 回合，每 10 回合重启一次进程：

.. code-block:: bash

   JAX_PLATFORMS=cpu python -m evaluations.bench2dex.run_all_channels \
     bench2dex_task64_openpi_pi05_eval --episodes-per-process 10

评测训练产物时，通过模型配置选择训练权重，并指定原来的 SFT 归一化文件：

.. code-block:: bash

   JAX_PLATFORMS=cpu python -m evaluations.bench2dex.run_all_channels \
     bench2dex_task64_openpi_pi05_eval --episodes-per-process 10 \
     rollout.model.model_path=/path/to/checkpoints/global_step_40/actor/model_state_dict/full_weights.pt \
     +rollout.model.openpi_data.norm_stats_path="$BENCH2DEX_PI05_PATH/assets/dex2bench/norm_stats.json"

该命令通过 OpenPI 模型加载器恢复动作网络，用指定的 SFT 归一化文件处理观测和动作。``BENCH2DEX_PI05_PATH`` 保持指向转换后的 SFT 目录。

启动器按顺序运行 ``none``、``cov``、``inv``、``inv_cov``，仿真和推理共用 GPU 0。用 ``--gpu-id`` 选择其他卡，``--output-root`` 设置输出目录，``--channels`` 选择通道。``--num-episodes`` 设置每通道总回合数，``--episodes-per-process`` 限制每次启动的总回合数；设为零且环境数能整除总回合数时，整个通道在一次启动中完成。

在同一个环境 worker 内并发运行两个场景：

.. code-block:: bash

   JAX_PLATFORMS=cpu python -m evaluations.bench2dex.run_all_channels \
     bench2dex_task64_openpi_pi05_eval \
     --num-envs 2 --num-episodes 50 --episodes-per-process 20

该命令分三次启动，分别完成 20、20、10 回合，每个场景对应运行 10、10、5 回合。启动器将 ``--num-envs`` 写入 ``env.eval.total_num_envs``，再根据回合预算计算 ``env.eval.rollout_epoch``。末批按需减少环境数，保证恰好完成指定回合。场景 0 取第 1、3、5 回合，场景 1 取第 2、4、6 回合，保持官方 episode seed 和锚点顺序。

观测以环境数为首维，chunk 结果形状为 ``[N,C]``。``auto_reset=true`` 时，只在 chunk 结束后重置已经结束的场景，终态保存在 ``final_observation`` 和 ``final_info`` 中。评测配置使用 ``auto_reset=false``。

配置通过 ``rollout.model.openpi.noise_rng: jax``，按每个回合的 seed 和 policy 查询编号在 CPU 生成噪声；模型继续在 GPU 上用 PyTorch 推理。仿真初始化在创建 simulation context 前设置任务 seed，物理时间步采用原生的 ``0.0166666``。

.. list-table:: 官方通道协议
   :header-rows: 1

   * - 通道
     - 场景准备
     - 种子命名空间
   * - ``none``
     - 恢复锚点场景。
     - ``100000000``
   * - ``cov``
     - 恢复锚点，只重采样物体位姿和桌高。
     - ``100000000``
   * - ``inv``
     - 恢复锚点，只重采样背景、桌面材质、灯光、杂物和相机。
     - ``100000000``
   * - ``inv_cov``
     - 在 unseen split 上重采样全部七个轴。
     - ``200000000``

回合种子按 ``namespace + task_id * 100000 + episode_index - 1`` 计算，Task64 首回合为 ``106400000`` 或 ``206400000``。锚点按官方顺序随回合编号循环。Task64 的控制预算为 665 步，配置调度 680 个动作位置，到任务预算时停止执行。

切换任务或机器人时，一起调整 task、embodiment、checkpoint、锚点目录和步数预算。环境根据机器人规格读取关节维度。

日志与结果
----------

输出根目录的 ``success_rates.tsv`` 汇总四通道成功率。每个通道写入原生 ``per_episode.jsonl``、``per_task.json``、``summary.json``，以及 RLinf 日志和 TensorBoard events。reach-and-stop 回合按上游 stable-success 判据计成功；公共评测命令和输出目录说明见 :doc:`评测 <../../evaluations/index>`。

每个环境 worker 通过一个 writer 写入所管理场景的结果。多个环境 worker 使用独立的 ``shards/<id>/`` 目录，四通道启动器按 episode 编号合并，并检查缺失和重复回合。
