Bench2Dex
=========

使用 RLinf 评测 Bench2Dex Task64 的 π₀.₅ policy，机器人为双 Panda＋Allegro。先准备场景资产、replay 锚点和转换后的 checkpoint，再运行官方四个泛化通道。

概览
----

Task64 配置采用每个环境 worker 一个场景，仿真环境注册为 ``env_type: bench2dex``。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀.₅

   .. grid-item-card:: 算法
      :text-align: center

      Policy 评测

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

评测配置读取转换后的 ``model.safetensors`` 和归一化文件。Task64 使用四个 policy 视角、32 维主动关节动作、20 步 action chunk 和 280 个 prompt tokens。

启动评测
--------

每通道运行 50 回合，每 10 回合重启一次进程：

.. code-block:: bash

   python -m evaluations.bench2dex.run_all_channels \
     bench2dex_task64_openpi_pi05_eval --episodes-per-process 10

启动器按顺序运行 ``none``、``cov``、``inv``、``inv_cov``，仿真和推理共用 GPU 0。用 ``--gpu-id`` 选择其他卡，``--output-root`` 设置输出目录，``--channels`` 选择通道。``--episodes-per-process 0`` 表示单个通道在同一进程中连续运行，``--num-episodes`` 设置该通道总回合数。

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
