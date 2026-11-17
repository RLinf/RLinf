SuperMimic：用可微世界模型训练 G1 动作跟踪
==========================================

本指南在 MuJoCo Warp 里复现 SuperMimic 在 Unitree G1 全身动作跟踪上的已报告结果：安装环境，准备 40 条 LAFAN1 片段，用 GenCoord 配方在一张 GPU 上训练 6000 次更新，中断后续训，评测最后的 checkpoint，再和 `可视化与结果`_ 里的数字对照。各步骤共用同一个运行目录，评测的就是这次训练写出的 checkpoint。算法以及和论文的差别见 :doc:`../../reference/algorithms/supermimic`；其他动作来源见 :doc:`supermimic_motion`；评测指标和推力恢复见 :doc:`supermimic_eval`。

.. toctree::
   :hidden:

   动作数据 <supermimic_motion>
   评测 <supermimic_eval>

除非另有说明，本页所有命令都在 RLinf 仓库根目录执行，并且已经激活下面安装的 SuperMimic 环境。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      SuperMimic：在可微世界模型里反向传播 32 步，外加 GAE critic

   .. grid-item-card:: 模型
      :text-align: center

      世界模型、动作 policy、critic；隐藏层用 Muon，其余参数用 fused Adam

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      MuJoCo Warp G1（mjlab）；LAFAN1，实验性的 BONES-SEED

   .. grid-item-card:: 训练
      :text-align: center

      每张 GPU 上一组 env、rollout 和 FSDP actor worker

| **你将完成：** 安装 -> 准备 LAFAN1 -> 训练 6000 次更新 -> 需要时续训 -> 评测第 6000 次更新 -> 对照已报告的表格。
| **前置条件：** 一张 NVIDIA GPU（已报告的运行用 A800），或实验性的 AMD MI300X 路径；RLinf 仓库（:doc:`安装说明 <../../start/installation>`）；能访问 Hugging Face 以下载片段。

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 项目
     - 内容
   * - 模拟器
     - 通过 `mjlab <https://github.com/mujocolab/mjlab>`_\ （NVIDIA 固定 commit ``8bb2bc9``，AMD 使用 1.2.0）使用 MuJoCo Warp。G1 有 29 个自由度，控制频率 50 Hz，每个控制步是 4 个 5 ms 物理步。
   * - 任务
     - 跟踪重定向到 G1 的参考动作片段。
   * - 状态
     - ``gencoord`` 是 71 维仿真状态，编码为 68 维。``maxcoord`` 是 14 个跟踪部位的位姿和速度，编码为 214 维。checkpoint 只能加载到训练时的那种表示。
   * - 动作
     - 29 个关节设定点，再由低增益 PD 控制器跟踪。训练时加标准差 0.1 的动作噪声，评测直接执行 policy 动作。
   * - Goal
     - 从当前参考帧起，每隔 5 个控制步取一帧，共 10 帧，超出片段末尾就停在最后一帧。

SuperMimic 工作原理
-------------------

SuperMimic 反复做三件事。先让所有环境在 MuJoCo Warp 里走一个控制步，把转移存下来。数据攒够后，用这些转移里的短窗口拟合世界模型。再让世界模型往前展开 32 步，把跟踪回报沿着它反向传播来改进 policy，并让 critic 拟合想象出的回报。policy 回报低的片段区段，回合会更频繁地从那里重新开始。在 RLinf 里，每张 GPU 上有一个仿真 worker、一个提供 policy 动作的 worker 和一个 FSDP 训练 worker。公式以及和官方代码的差别见 :doc:`算法页 <../../reference/algorithms/supermimic>`。

安装
----

NVIDIA 路径通过 MuJoCo Warp 运行仿真。在 RLinf 仓库根目录用安装脚本单独建一个虚拟环境：

.. code-block:: bash

   cd /path/to/RLinf
   bash requirements/install.sh embodied --env supermimic --venv .venv-supermimic
   source .venv-supermimic/bin/activate

之后每打开一个新终端，都要先回到仓库根目录并执行 ``source .venv-supermimic/bin/activate``。

这个环境使用 torch 2.14.0。torch 是 CUDA 12 构建时，脚本固定安装 ``warp-lang`` 1.18.0+cu12。这些依赖会升级 torch，请和其他 RLinf 环境分开安装。``docker/Dockerfile`` 也提供 ``embodied-supermimic-image`` 目标。

AMD MI300X（实验性）
~~~~~~~~~~~~~~~~~~~~

AMD 路径使用 ROCm Warp。先准备 Linux x86_64、ROCm 7.2 SDK（含 ``hipcc``）和 C++ 编译工具链；可参考 `ROCm Warp 的 gfx942 Docker 构建步骤 <https://github.com/ROCm/warp/blob/36ed75aa65f837a78b20d1a3a1ac08cfe1fc7b71/docker/rocm/Dockerfile>`_ 准备环境。在仓库根目录执行：

.. code-block:: bash

   export ROCM_PATH=/opt/rocm  # 按实际 SDK 安装路径修改
   bash requirements/install.sh --platform amd --rocm 7.2 --no-flash-attn \
       embodied --env supermimic --venv .venv-supermimic
   source .venv-supermimic/bin/activate

脚本从 AMD Radeon 的 wheel 索引安装 torch 2.9.1 和 torchvision 0.24.0（需要支持 flat 索引的新版 uv），在上面链接的提交上构建 ROCm Warp ``1.13.0+rocm.0``，再安装 mjlab 1.2.0、MuJoCo 3.8.1 和 mujoco-warp 3.8.1。``SUPERMIMIC_WARP_PATH`` 指定 Warp 源码目录（默认是仓库旁的 ``../warp-rocm``），``SUPERMIMIC_HIP_ARCH`` 指定编译目标（默认 ``gfx942``）。之后的步骤和下文相同。

这个安装脚本还没有跑过完整训练。

准备 LAFAN1
-----------

片段来自 Hugging Face 数据集 `lvhaidong/LAFAN1_Retargeting_Dataset <https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset>`_ 的 ``g1/`` 目录。脚本按固定修订版 ``ce1572906efe6157840e8474d5a0d7aa87481e74`` 下载 40 个 CSV，逐个核对大小和 SHA-256，转换后写出 ``manifest.json``。数据集卡片没有声明许可证；这些重定向数据来自 Ubisoft La Forge 的 `LAFAN1 <https://github.com/ubisoft/ubisoft-laforge-animation-dataset>`_，后者使用 CC BY-NC-ND 4.0（非商业、禁止演绎）。传 ``--accept-license`` 之前请先读这些条款。

在训练机器上执行：

.. code-block:: bash

   python toolkits/supermimic/prepare_lafan1.py --out /data/lafan1_g1 --accept-license

最后一行输出是 ``wrote 40 clips, 2.45 h``。目录里有 ``clip_000.pt`` 到 ``clip_039.pt``、``manifest.json``，以及 ``csv/`` 下的原始 CSV。有清单时，加载器会拒绝 SHA-256 对不上或清单里没有的片段，所以不要改名、增删或修改准备好的文件。

在另一台机器上转换
~~~~~~~~~~~~~~~~~~

训练机器连不上 Hugging Face 时，可以在能连上的机器上转换。那台机器需要一份 RLinf 代码，以及装有 torch、numpy 和 omegaconf 的 Python，不需要 mjlab。转换器要从 mjlab 读取 G1 的默认关节姿态，所以先在训练机器上把它导出来。

1. 在训练机器上激活 SuperMimic 环境，把姿态写到 ``g1_q0.pt``：

   .. code-block:: bash

      python -c "import torch; from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state; torch.save(asset_default_state()[7:36].clone(), 'g1_q0.pt')"

2. 把 ``g1_q0.pt`` 复制到转换机器。
3. 在转换机器的 RLinf 目录下，带上这个姿态下载并转换：

   .. code-block:: bash

      python toolkits/supermimic/prepare_lafan1.py --out /data/lafan1_g1 \
          --q0-file /path/to/g1_q0.pt --accept-license

4. 把整个 ``/data/lafan1_g1`` 目录（包括 ``manifest.json``）复制到训练机器的同一路径。

运行
----

每打开一个新终端，都先在仓库根目录执行下面这段，再训练、续训或评测。后文的所有路径都由 ``RUN_DIR`` 推出：

.. code-block:: bash

   export PYTHONPATH=$PWD:$PYTHONPATH
   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   export MUJOCO_GL=disabled
   export RUN_DIR=/data/runs/supermimic_gencoord
   GPU_PLACEMENT=('~cluster.component_placement'
       '+cluster.component_placement={actor:0,env:0,rollout:0}')
   TRAIN_ARGS=(--config-name supermimic_lafan1_gencoord
       runner.max_epochs=6000
       actor.cuda_graph=True
       runner.logger.log_path=$RUN_DIR
       "${GPU_PLACEMENT[@]}")

``GPU_PLACEMENT`` 把 env、rollout 和 actor worker 都放到 0 号 GPU，训练和评测都用它；要换卡就把 ``0`` 改成对应序号。不加它时，训练和评测都会用上机器上的所有 GPU。

训练
~~~~

已报告的运行是单卡 GenCoord，使用配置默认值：4096 个环境，每次更新的世界模型批次是 2048 个窗口，想象 rollout 是 4096 条。``TRAIN_ARGS`` 在此之上设定 6000 次更新和 CUDA 图回放。矩阵乘用 TF32，权重保持 float32。

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py "${TRAIN_ARGS[@]}"

在一张 A800 上训练略超过一小时。最初几次更新之后，日志会出现 ``Recorded the SuperMimic update as a CUDA graph.``；之后每 100 次更新打印一张指标表，开头是 ``Global Step:``，后面跟当前更新数和 ``/6000``。每 2000 次更新写一个 checkpoint，位置是 ``$RUN_DIR/supermimic_lafan1_gencoord/checkpoints/global_step_<N>``。

续训
~~~~

训练中断后，如果是新终端，先执行 `运行`_ 里的准备代码，再在训练命令后加上 ``runner.resume_dir=auto`` 重新运行：

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py "${TRAIN_ARGS[@]}" \
       runner.resume_dir=auto

``auto`` 会在 ``$RUN_DIR/supermimic_lafan1_gencoord/checkpoints`` 下选最新的完整 checkpoint。checkpoint 里有 ``runner_state.json`` 才算完整，这个文件最后写出。checkpoint 还包括 ``actor/`` 下分片的网络、优化器和随机数状态，``env/`` 下的重置统计，以及评测用的 ``actor/model_state_dict/full_weights.pt``。请确认日志里有 ``Resuming from`` 和预期的 ``global_step_<N>``。如果出现 ``resume_dir=auto found no complete checkpoint``，说明训练从头开始了，请检查 ``RUN_DIR`` 和配置名。replay 不随 checkpoint 保存，续训后要先采集几秒才会做下一次更新。续训时请保持 GPU 数不变。

评测
~~~~

在 40 条片段上评测最终权重。每条片段从第 0 帧播放一遍，直接执行 policy 动作：

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       runner.ckpt_path=$RUN_DIR/supermimic_lafan1_gencoord/checkpoints/global_step_6000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=$RUN_DIR/eval_6000 \
       "${GPU_PLACEMENT[@]}"

评测约 7 分钟。``$RUN_DIR/eval_6000/summary.json`` 里有下表报告的 ``normalized_return``、``fail_rate`` 和 ``body_pos_err``；``per_clip.csv`` 每条片段一行。指标定义见 :doc:`supermimic_eval`。

多卡训练
~~~~~~~~

环境数、世界模型批量和想象批量都是所有 GPU 的总和，增加 GPU 只是把同一次更新分到多张卡上。在 `运行`_ 的准备代码里换一个新的 ``RUN_DIR``，并在 ``GPU_PLACEMENT`` 里列出这些 GPU。两张卡时改成下面两项，重新执行整段准备代码，再运行训练命令：

.. code-block:: bash

   export RUN_DIR=/data/runs/supermimic_gencoord_2gpu
   GPU_PLACEMENT=('~cluster.component_placement'
       '+cluster.component_placement={actor:0-1,env:0-1,rollout:0-1}')

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

四张卡时把 placement 里的 ``0-1`` 改成 ``0-3``。每张 GPU 上各有一个 env、一个 rollout 和一个 actor worker。学习率和更新次数不随 GPU 数变化。续训时先执行改过的准备代码，再运行续训命令。要像 `评测`_ 那样单卡评测，把 ``GPU_PLACEMENT`` 改回 0 号 GPU，``RUN_DIR`` 保持新的值。多节点和 Slurm 启动还没有用这个配方验证过。

MaxCoord
~~~~~~~~

要训练 MaxCoord 表示，换一个新的 ``RUN_DIR``，把 ``--config-name`` 改成 ``supermimic_lafan1_maxcoord``。它的 checkpoint 在 ``$RUN_DIR/supermimic_lafan1_maxcoord/checkpoints`` 下，评测时要加 ``actor.model.representation=maxcoord``。MaxCoord 目前没有已报告的数字。

配置
----

两份训练配置只差状态表示，评测配置继承 GenCoord：

* 训练：``examples/embodiment/config/supermimic_lafan1_gencoord.yaml`` 和 ``supermimic_lafan1_maxcoord.yaml``；
* 评测：``evaluations/supermimic/supermimic_lafan1_eval.yaml`` 做动作跟踪，``supermimic_push_recovery.yaml`` 做站立推力恢复。

下表是最常改的键，都可以作为 Hydra 覆盖项传入。

.. list-table::
   :header-rows: 1
   :widths: 40 12 48

   * - 键
     - 默认
     - 含义
   * - ``runner.max_epochs``
     - 400000
     - 全局更新次数。已报告的运行设为 6000。
   * - ``runner.save_interval``
     - 2000
     - 每隔这么多次更新，以及最后一次更新后，写一个 checkpoint。
   * - ``env.train.total_num_envs``
     - 4096
     - 所有 GPU 上的环境总数，必须是 GPU 数的整数倍。
   * - ``env.train.num_clips``
     - 40
     - 加载的片段数。``null`` 加载目录里的全部片段。
   * - ``algorithm.replay_ratio``
     - 16
     - 世界模型平均重复使用每个采集到的控制步的次数。默认值下每个控制步对应 0.4 次更新，所以 6000 次更新约采集 15000 个控制步。
   * - ``algorithm.wm_window_batch``
     - 2048
     - 每个世界模型批次的窗口数，所有 GPU 的总和。
   * - ``algorithm.imagine_batch``
     - 4096
     - 每次更新的想象 rollout 条数，所有 GPU 的总和。
   * - ``actor.cuda_graph``
     - False
     - 前三次之后的每次更新作为一张 CUDA 图回放。已报告的运行设为 ``True``。要求学习率是常数。
   * - ``actor.fsdp_config.amp_autocast.enabled``
     - False
     - 网络矩阵乘改用 bf16，权重、损失和优化器状态仍是 float32。这不是论文的 TF32 设置。

可视化与结果
------------

TensorBoard 事件写在 ``$RUN_DIR/tensorboard``：

.. code-block:: bash

   tensorboard --logdir $RUN_DIR/tensorboard

横轴是更新次数。``env/sim/reward_mean`` 和 ``env/sim/terminal_rate`` 反映仿真里的跟踪情况，``train/actor/reward`` 是想象中的奖励，``train/world_model/state_loss`` 反映世界模型的拟合。通用指标的含义见 :doc:`../../reference/metrics`。

已报告的跟踪结果
~~~~~~~~~~~~~~~~

GenCoord，40 条 LAFAN1 片段，4096 个环境，种子 0，用 `运行`_ 里的命令训练，按 `评测`_ 的方式评测，GPU 是 NVIDIA A800。RLinf 的运行都打开 CUDA 图回放，bf16 的几行另设 ``actor.fsdp_config.amp_autocast.enabled=True``。官方一行是本集成的作者在同一集群上运行的官方 SuperMimic 代码。所有运行都评测第 6000 次更新，约 15000 个控制步。

.. list-table::
   :header-rows: 1
   :widths: 14 8 10 10 20 14 20

   * - 运行
     - GPU 数
     - 矩阵乘
     - 更新次数
     - ``normalized_return``
     - ``fail_rate``
     - ``body_pos_err``\ （m）
   * - 官方
     - 1
     - TF32
     - 6000
     - 0.830
     - 0.025
     - 0.027
   * - RLinf
     - 1
     - TF32
     - 6000
     - 0.846
     - 0
     - 0.024
   * - RLinf
     - 1
     - bf16
     - 6000
     - 0.857
     - 0
     - 0.023
   * - RLinf
     - 2
     - TF32
     - 6000
     - 0.843
     - 0.025
     - 0.023
   * - RLinf
     - 2
     - bf16
     - 6000
     - 0.833
     - 0.05
     - 0.023

* 这些运行早于当前版本。当前版本把三个网络拆成独立的 FSDP 单元，也改了 replay 窗口和 critic 小批的抽取方式，还没有在同等规模上重跑，所以上表数字对当前版本仍待验证。
* 每行是一个种子。``fail_rate`` 0.025 表示 40 条里摔倒 1 条。较早的 checkpoint 在不同运行之间差得更多：四次 RLinf 运行的 ``normalized_return`` 在第 2000 次更新时为 0.672–0.732，第 4000 次时为 0.765–0.794。
* 论文在这 40 条片段上报告的归一化回报是 0.82。

训练速度
~~~~~~~~

同一个 GenCoord 配方在 A800 上每次更新的秒数，都打开 CUDA 图回放。RLinf 的每次运行独占所用 GPU，共跑 1200 次更新，统计后 1000 次。``time/step`` 是日志里每次更新计时的均值，包含采集、更新和权重同步。墙钟是这 1000 次更新的总耗时除以 1000，还包含写日志等开销。这些运行和上表用的是同一个早期版本，当前版本的速度待测。

.. list-table::
   :header-rows: 1
   :widths: 16 10 12 22 22

   * - 运行
     - GPU 数
     - 矩阵乘
     - ``time/step``\ （秒）
     - 墙钟（秒）
   * - 官方
     - 1
     - TF32
     - --
     - 0.515
   * - RLinf
     - 1
     - TF32
     - 0.603
     - 0.637
   * - RLinf
     - 1
     - bf16
     - 0.515
     - 0.548
   * - RLinf
     - 2
     - TF32
     - 0.486
     - 0.521
   * - RLinf
     - 2
     - bf16
     - 0.449
     - 0.483
   * - RLinf
     - 4
     - TF32
     - 0.421
     - 0.456
   * - RLinf
     - 4
     - bf16
     - 0.407
     - 0.442

官方的时间由本集成的作者在同一集群上实测，取官方 MaxCoord 运行两个 checkpoint 之间的墙钟。RLinf 的时间里约有 4% 来自 rollout 一侧每分钟一次、约 2 秒的停顿。

范围
----

CI 在生成的站立片段上运行 ``tests/unit_tests/test_supermimic.py`` 里的单元测试、一次带续训和独立评测的短 GenCoord 训练，以及一次双卡的短 MaxCoord 训练。上面的表格评测的是 GenCoord 第 6000 次更新在 40 条 LAFAN1 片段上的结果。MaxCoord 的精度、超过 6000 次更新的运行、多节点、Slurm、BONES-SEED 训练，以及论文 §4.3 的 T-pose 恢复，目前都没有数字。
