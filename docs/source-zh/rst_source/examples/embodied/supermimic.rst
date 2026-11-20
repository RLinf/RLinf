SuperMimic：用可微世界模型训练 G1 动作跟踪
==========================================

SuperMimic 让跟踪回报沿着学到的世界模型反向传播，以此训练 Unitree G1 的全身动作跟踪 policy。本页在 MuJoCo Warp 里带你完整跑一遍：安装依赖，准备 40 条 LAFAN1 片段，用 GenCoord 配方单卡训练，评测这次训练写出的 checkpoint，最后找到各项输出文件。算法细节及其与论文的差别见 :doc:`算法页 <../../reference/algorithms/supermimic>`。

.. toctree::
   :hidden:

   动作数据 <supermimic_motion>
   评测 <supermimic_eval>

概览
----

在 40 条 LAFAN1 片段上训练一个 G1 跟踪 policy，再用这些片段评测。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      SuperMimic：在世界模型里反向传播 32 步，外加 critic

   .. grid-item-card:: 模型
      :text-align: center

      世界模型、跟踪 policy、critic

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      MuJoCo Warp G1（mjlab）；LAFAN1 片段

   .. grid-item-card:: 训练
      :text-align: center

      单卡；单节点多卡

| **你将完成：** 安装 → 准备 LAFAN1 → 单卡训练 → 评测 checkpoint → 查看输出。
| **前置条件：** 一张 NVIDIA GPU；RLinf 仓库（:doc:`安装说明 <../../start/installation>`）；能访问 Hugging Face 以下载片段。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 说明
   * - 观测
     - 机器人当前状态，加上片段里接下来的 10 帧参考动作，每两帧相隔 5 个控制步。本页用的 GenCoord 以模拟器的广义坐标描述状态；MaxCoord 则用 14 个跟踪部位的位姿和速度描述。
   * - 动作
     - 29 个关节的设定点，频率 50 Hz，由 PD 控制器跟踪。训练时给动作加噪声，评测时原样执行。
   * - 奖励
     - 衡量机器人贴合片段的程度，每个控制步最多 0.1。摔倒即结束这一回合。

SuperMimic 工作原理
-------------------

SuperMimic 反复做三件事。先让所有环境在 MuJoCo Warp 里走一个控制步，存下状态转移。再从存下的转移里截取短窗口，拟合世界模型。最后让世界模型往前推 32 步，把跟踪回报沿世界模型反向传播来改进 policy，同时让 critic 拟合想象出的回报。回合更多地从片段里 policy 回报低的区段开始。公式见 :doc:`算法页 <../../reference/algorithms/supermimic>`。

安装
----

在 RLinf 仓库根目录单独建一个虚拟环境：

.. code-block:: bash

   bash requirements/install.sh embodied --env supermimic --venv .venv-supermimic
   source .venv-supermimic/bin/activate

这个虚拟环境固定了 torch 和 ``warp-lang`` 的版本，不要和其他 RLinf 虚拟环境混用。``docker/Dockerfile`` 也提供 ``embodied-supermimic-image`` 构建目标。AMD GPU 见 `在其他硬件后端上运行`_。

下文所有命令都在仓库根目录、激活这个虚拟环境后执行。

准备 LAFAN1
-----------

片段来自 Hugging Face 数据集 `lvhaidong/LAFAN1_Retargeting_Dataset <https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset>`_ 的 ``g1/`` 目录。数据集卡片没有声明许可证。这些重定向数据源自 Ubisoft La Forge 的 `LAFAN1 <https://github.com/ubisoft/ubisoft-laforge-animation-dataset>`_。LAFAN1 采用 CC BY-NC-ND 4.0 许可（非商业、禁止演绎），传 ``--accept-license`` 前请先读这些条款。

下载、校验并转换 40 条片段：

.. code-block:: bash

   python toolkits/supermimic/prepare_lafan1.py --out /data/lafan1_g1 --accept-license

脚本下载指定版本的数据，逐个核对文件大小和 SHA-256，写出转换后的片段和 ``manifest.json``。最后一行输出是：

.. code-block:: text

   [lafan1] wrote 40 clips, 2.45 h, manifest /data/lafan1_g1/manifest.json

训练会拒绝清单里没有或校验和对不上的片段，所以 ``/data/lafan1_g1`` 里的文件保持原样，不要改动。

在另一台机器上转换
~~~~~~~~~~~~~~~~~~

训练机器连不上 Hugging Face 时，换一台能连上的机器转换。那台机器需要一份 RLinf 代码，以及装有 torch、numpy 和 omegaconf 的 Python。转换还要用到 G1 的默认关节姿态，这只能在训练机器上导出。

1. 在训练机器上把姿态写到 ``g1_q0.pt``：

   .. code-block:: bash

      python toolkits/supermimic/prepare_lafan1.py --export-q0 g1_q0.pt

2. 把 ``g1_q0.pt`` 复制到转换机器。
3. 在转换机器的 RLinf 目录下，带上这个姿态下载并转换：

   .. code-block:: bash

      python toolkits/supermimic/prepare_lafan1.py --out /data/lafan1_g1 \
          --q0-file /path/to/g1_q0.pt --accept-license

4. 把整个 ``/data/lafan1_g1`` 目录复制到训练机器的同一路径。

运行
----

训练、续训和评测共用同一段设置代码和同一个运行目录，评测读到的就是这次训练写出的 checkpoint。

设置终端
~~~~~~~~

每打开一个新终端，都先执行下面的设置代码，再运行后面的命令：

.. code-block:: bash

   export PYTHONPATH=$PWD:$PYTHONPATH
   export MUJOCO_GL=disabled
   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   export RUN_DIR=/data/runs/supermimic_gencoord

* ``MUJOCO_GL=disabled`` 跳过 MuJoCo 的 OpenGL 渲染器，训练和评测都用不到它。
* ``SUPERMIMIC_MOTION_PATH`` 是 `准备 LAFAN1`_ 写出的片段目录。
* ``RUN_DIR`` 存放日志、checkpoint 和评测结果。

单卡训练
~~~~~~~~

GenCoord 配置在 0 号 GPU 上训练 6000 次更新：4096 个环境，每次更新的世界模型批次是 2048 个窗口、想象 rollout 4096 条，矩阵乘用 TF32，权重是 float32。要换一张 GPU，改 ``examples/embodiment/config/supermimic_lafan1_gencoord.yaml`` 里 ``cluster.component_placement`` 下的 ``actor,env,rollout: 0``。打开 CUDA 图回放，开始训练：

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_gencoord \
       runner.logger.log_path=$RUN_DIR actor.cuda_graph=True

最初几次更新后，日志打印 ``Recorded the SuperMimic update as a CUDA graph.``。之后每 100 次更新打印一张指标表，开头是 ``Global Step:``，后面跟更新次数和 ``/6000``。每 2000 次更新和最后一次更新后各写一个 checkpoint。

训练中断后，在新终端里先执行设置代码，再从最新的完整 checkpoint 续训：

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_gencoord \
       runner.logger.log_path=$RUN_DIR actor.cuda_graph=True \
       runner.resume_dir=auto

日志会在 ``Resuming from`` 后面给出所用的 checkpoint。如果出现 ``resume_dir=auto found no complete checkpoint``，训练就是从头开始的，请检查 ``RUN_DIR`` 和配置名。续训时 GPU 数要和原来相同。

评测 checkpoint
~~~~~~~~~~~~~~~

用第 6000 次更新的权重，把 40 条片段各从第一帧播放一遍：

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       runner.ckpt_path=$RUN_DIR/supermimic_lafan1_gencoord/checkpoints/global_step_6000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=$RUN_DIR/eval_6000

日志最后一行是 ``Wrote SuperMimic evaluation results to`` 加输出目录。评测结果写到 ``$RUN_DIR/eval_6000/``。``evaluations/run_eval.sh`` 把评测日志写到 ``logs/<时间戳>-supermimic_lafan1_eval/``。要评测第 2000 或 4000 次更新，把两个路径里的 ``6000`` 换掉即可。指标定义和站立推力恢复见 :doc:`supermimic_eval`。

查看输出
~~~~~~~~

训练文件和评测结果在 ``RUN_DIR`` 下，评测日志在 ``logs/`` 下：

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - ``$RUN_DIR`` 下的路径
     - 内容
   * - ``metrics.log``
     - 训练中打印的指标表
   * - ``tensorboard/``
     - 训练的 TensorBoard 事件
   * - ``supermimic_lafan1_gencoord/checkpoints/global_step_<N>/``
     - 一个 checkpoint，``runner.resume_dir=auto`` 从这里续训
   * - ``.../global_step_<N>/actor/model_state_dict/full_weights.pt``
     - 评测加载的权重
   * - ``eval_6000/summary.json``
     - 40 条片段的评测指标
   * - ``eval_6000/per_clip.csv``
     - 每条片段一行指标
   * - ``eval_6000/config.yaml``
     - 评测实际使用的完整配置

可视化与结果
------------

用 TensorBoard 查看训练过程：

.. code-block:: bash

   tensorboard --logdir $RUN_DIR/tensorboard

横轴是更新次数。``env/sim/reward_mean`` 是模拟器中每步的奖励，会逐渐升向上限 0.1。``env/sim/terminal_rate`` 是每步摔倒的环境比例，会降到接近 0。``train/world_model/state_loss`` 反映世界模型预测下一状态的准确程度。通用指标的含义见 :doc:`../../reference/metrics`。

评测结果看 ``$RUN_DIR/eval_6000/summary.json`` 里的 ``normalized_return``：每条片段实际拿到的奖励除以最多能拿到的奖励，再对 40 条取平均。``fail_rate`` 是机器人摔倒的片段比例，``body_pos_err`` 是跟踪部位的平均位置误差，单位米。

按这套流程在一张 NVIDIA A800 上用 seed 0 跑一次的结果：

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - 更新次数
     - ``normalized_return``
     - ``fail_rate``
     - ``body_pos_err`` （米）
   * - 6000
     - 0.850
     - 0
     - 0.025

40 条片段都播到了最后一帧。论文在这 40 条片段上报告的归一化回报是 0.82。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/supermimic_gencoord_curves.png
   :alt: SuperMimic GenCoord 训练曲线
   :width: 100%

   同一次运行的训练曲线。

与官方代码对比
~~~~~~~~~~~~~~

GenCoord 配方在 NVIDIA A800 上每次更新的耗时，官方代码在同一批机器上运行。RLinf 各行设置 ``actor.cuda_graph=True``。bf16 各行再设置 ``actor.fsdp_config.amp_autocast.enabled=True``，网络矩阵乘用 bf16。

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - 运行
     - GPU 数
     - 矩阵乘
     - 秒/次更新
   * - 官方
     - 1
     - TF32
     - 0.515
   * - RLinf
     - 1
     - TF32
     - 0.603
   * - RLinf
     - 1
     - bf16
     - 0.515
   * - RLinf
     - 2
     - TF32
     - 0.486
   * - RLinf
     - 2
     - bf16
     - 0.449
   * - RLinf
     - 4
     - TF32
     - 0.421
   * - RLinf
     - 4
     - bf16
     - 0.407

RLinf 的时间取自独占 GPU 的一次运行：预热 200 次更新后，对接下来 1000 次更新的 ``time/step`` 取平均。计时包含采集、更新，以及把权重复制给 policy，不含写日志。官方的时间是一次 MaxCoord 运行里前后两个 checkpoint 的时间差。单卡 TF32 下，RLinf 每次更新比官方代码长约 17%：每个控制步里，环境和 policy 都要交换观测与动作；每次更新后，权重还要复制给 policy。

其他设置
--------

多卡训练
~~~~~~~~

环境数、世界模型批次和想象 rollout 数都是所有 GPU 的总和，多加 GPU 只是把同一次更新分摊出去。换一个新的 ``RUN_DIR``，在训练命令末尾列出要用的 GPU。两张卡时：

.. code-block:: bash

   export RUN_DIR=/data/runs/supermimic_gencoord_2gpu
   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_gencoord \
       runner.logger.log_path=$RUN_DIR actor.cuda_graph=True \
       '~cluster.component_placement' \
       '+cluster.component_placement={actor:0-1,env:0-1,rollout:0-1}'

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - 每次更新
     - 总量
     - 两卡时每卡
     - 四卡时每卡
   * - 环境
     - 4096
     - 2048
     - 1024
   * - 每个世界模型批次的窗口（共 10 批）
     - 2048
     - 1024
     - 512
   * - 想象 rollout
     - 4096
     - 2048
     - 1024

四张卡时写 ``0-3``。学习率和更新次数不变。续训时带上同样的 GPU 列表。评测沿用 `评测 checkpoint`_ 里的命令，在 0 号 GPU 上运行。

训练 MaxCoord
~~~~~~~~~~~~~

换一个新的 ``RUN_DIR``，把训练命令里的 ``--config-name`` 改成 ``supermimic_lafan1_maxcoord``。checkpoint 在 ``$RUN_DIR/supermimic_lafan1_maxcoord/checkpoints`` 下，评测命令要加上 ``actor.model.representation=maxcoord``。

用其他片段训练见 :doc:`supermimic_motion`。

.. _supermimic-hardware:

在其他硬件后端上运行
--------------------

AMD ROCm
~~~~~~~~

在 AMD GPU 上，MuJoCo Warp 运行在 ROCm Warp 之上，面向 MI300X（``gfx942``）。先准备 Linux x86_64、带 ``hipcc`` 的 ROCm 7.2 SDK 和 C++ 编译工具链，可以参考 `ROCm Warp 的 gfx942 Docker 构建步骤 <https://github.com/ROCm/warp/blob/36ed75aa65f837a78b20d1a3a1ac08cfe1fc7b71/docker/rocm/Dockerfile>`_ 搭环境。在仓库根目录安装：

.. code-block:: bash

   export ROCM_PATH=/opt/rocm  # 按实际 SDK 安装路径修改
   bash requirements/install.sh --platform amd --rocm 7.2 --no-flash-attn \
       embodied --env supermimic --venv .venv-supermimic
   source .venv-supermimic/bin/activate

脚本从 AMD 的 wheel 索引安装 torch 2.9.1，需要能读取 flat 索引的新版 uv。它按上面链接里的提交构建 ROCm Warp ``1.13.0+rocm.0``，再安装 mjlab 1.2.0、MuJoCo 3.8.1 和 mujoco-warp 3.8.1。``SUPERMIMIC_WARP_PATH`` 指定 Warp 源码目录（默认 ``../warp-rocm``），``SUPERMIMIC_HIP_ARCH`` 指定编译目标（默认 ``gfx942``）。

之后按 `准备 LAFAN1`_ 和 `运行`_ 操作即可。AMD 上还没有跑通过完整训练。
