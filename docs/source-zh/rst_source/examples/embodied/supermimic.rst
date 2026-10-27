SuperMimic：用可微世界模型训练 G1 动作跟踪
==========================================

本指南在 MuJoCo Warp 里用 SuperMimic 训练并评测 Unitree G1 的全身动作跟踪 policy：安装环境、准备参考动作、选择配置、在一张或多张 GPU 上训练、恢复训练、评测。算法、公式以及实现与论文的差异见 :doc:`../../reference/algorithms/supermimic`。

概览
----

* **仿真器：** 通过 `mjlab <https://github.com/mujocolab/mjlab>`_\ （commit ``8bb2bc9``）使用 MuJoCo Warp，G1 29 自由度，控制频率 50 Hz，每个控制步 4 个 5 ms 物理步。
* **任务：** 跟踪重定向到 G1 的参考动作片段（LAFAN1 或 BONES-SEED）；可以再加一个编码器跟踪速度命令。
* **观测：** 最近 10 帧的关节角（相对默认姿态）、关节速度、基座角速度、投影重力和上一动作，每帧 93 个数，带均匀传感器噪声。
* **动作：** 29 个关节设定点 :math:`q_{sp} = \alpha a + q_0`，由低增益 PD 跟随。
* **Goal：** 10 帧参考，每帧 68 个数（680 维），或 `配置`_ 里的变体。
* **训练：** 世界模型拟合、经过世界模型的 32 步反向传播、GAE critic，每张 GPU 一个进程完成全部工作。

安装
----

SuperMimic 需要 NVIDIA GPU（MuJoCo Warp）。用 RLinf 的安装脚本单独建一个虚拟环境：

.. code-block:: bash

   bash requirements/install.sh embodied --env supermimic --venv .venv-supermimic
   source .venv-supermimic/bin/activate

因为 mjlab 要求 ``torch>=2.14.0``，安装脚本只给这个环境选 torch 2.14.0，RLinf 全局的 torch 版本不变。装好的 torch 是 CUDA 12 构建时，脚本会再装 ``warp-lang`` 1.18.0 的 CUDA 12 构建（PyPI 上的 wheel 是 CUDA 13 构建）。SuperMimic 专用的包列在 ``requirements/embodied/envs/supermimic.txt``；用 ``pip`` 把这个文件装进已有的 RLinf 环境会升级那里的 torch，所以请让 SuperMimic 用独立环境。``docker/Dockerfile`` 里的 ``embodied-supermimic-image`` 目标执行同样的安装。

G1 的模型、关节顺序、执行器增益和站立关键帧都来自 mjlab。每个参考片段都带着转换时用的默认姿态 ``q0``，如果它和 mjlab 的默认关节姿态不同，仿真器会拒绝加载：Goal 和观测减去的是 ``q0``，而 PD 目标加的是 mjlab 的默认姿态。

准备参考动作
------------

动作数据不放进仓库。每个 ``clip_*.pt`` 包含 ``fps``\ （50，或 30 由加载器重采样）、29 个 G1 关节名、``q0``，以及 ``[T, 71]`` 的状态：根位置、根四元数（wxyz）、关节角、世界系根线速度、根坐标系角速度、关节速度。30 Hz 的片段会重采样到 50 Hz 控制网格，速度用中心差分重新计算。

**LAFAN1（论文的 40 条）。** 片段是 Hugging Face 数据集 ``lvhaidong/LAFAN1_Retargeting_Dataset`` 的 ``g1/`` 目录。脚本固定修订版 ``ce1572906efe6157840e8474d5a0d7aa87481e74``，逐个核对 CSV 的大小和 SHA-256，并写出 ``manifest.json``。数据集卡片没有声明许可证；这些重定向数据来自 Ubisoft La Forge 的 LAFAN1，后者使用 CC BY-NC-ND 4.0（非商业、禁止演绎）。传 ``--accept-license`` 之前请先读这些条款。

.. code-block:: bash

   python toolkits/supermimic/prepare_lafan1.py --out /data/lafan1_g1 --accept-license
   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1

40 条片段共 2.45 小时。有 ``manifest.json`` 时，加载器会拒绝 SHA-256 对不上或清单里没有的片段。

**BONES-SEED（实验性）。** 加载器要求片段已经是上述格式的 G1 重定向数据，并带一个供附录 E 名字过滤使用的 ``name``。RLinf 不提供从 BONES-SEED 发布格式转换的脚本，因为还没有拿真实数据核对过那个格式。可行性过滤需要逐帧几何量，下面的工具用 ``mj_geomDistance`` 算出来，写进每个片段的副本：

.. code-block:: bash

   python toolkits/supermimic/bones_feasibility.py --clips /data/bones_g1 \
       --out /data/bones_g1_filtered
   export SUPERMIMIC_BONES_PATH=/data/bones_g1_filtered

BONES-SEED 配置加载所有通过过滤的片段（``env.num_clips: null``），并在日志里打印每个过滤器去掉了多少。论文的 142,220 / 129,667 / 129,514 是它那个版本的数据；这条流程还没有处理过完整的 BONES-SEED。

配置
----

配置都在 ``examples/embodiment/config/``。批量都是全局量，多卡时每个 rank 分到相同的一份。

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 配置
     - 训练内容
   * - ``supermimic_lafan1``
     - LAFAN1 上的论文设定：680 维 Goal、93 维观测、4096 个环境，每次更新 10 × 2048 个世界模型窗口、4096 条想象 rollout。
   * - ``supermimic_lafan1_anchor``
     - 689 维 Goal：680 维 Goal 加上参考躯干在机器人躯干坐标系下的位姿（CuboiLeo 的锚点）。
   * - ``supermimic_lafan1_goal730``
     - zita-ch 的 730 维 Goal（接下来 10 帧，每帧 73 个数）。
   * - ``supermimic_lafan1_obs99``
     - 99 维观测：93 维再加根朝向相对参考的 Rot6D。
   * - ``supermimic_velocity``
     - 动作跟踪加速度命令编码器（附录 G），各占一半环境。
   * - ``supermimic_bones_seed``
     - 过滤后 BONES-SEED 上的论文设定。
   * - ``supermimic_bones_seed_2gpu``
     - BONES-SEED 上的论文双卡配方（见 `多卡训练`_）。
   * - ``supermimic_smoke``
     - 适合单张 8 GB GPU 的小规模运行：8 个环境、小批量、3 次更新，网络、``k``、``H`` 和奖励与论文设定相同。

有两个实验选项会改变终止头，默认关闭：``algorithm.termination_head: analytic`` 和 ``algorithm.termination_bias_init``，什么时候需要它们见算法页。``algorithm.seed`` 默认是 0，第 ``r`` 个 rank 用 ``seed + r``。

单卡训练
--------

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   bash examples/embodiment/run_supermimic.sh supermimic_lafan1

``run_supermimic.sh <配置名> [hydra 覆盖项...]`` 把配置名之后的参数都交给 Hydra；没有给 ``runner.logger.log_path`` 时，写到 ``logs/<时间>-<配置名>``。在小显存 GPU 上快速检查：

.. code-block:: bash

   bash examples/embodiment/run_supermimic.sh supermimic_smoke env.num_clips=40

只有 smoke 启动会在你没设置时把 ``RAY_memory_usage_threshold`` 设为 0.99，因为 8 GB 内存的机器离 Ray 的默认阈值很近；其他配置都保留 Ray 内存监控的原样。Ray 对象存储限制为 512 MiB（``cluster.object_store_memory_mb``），因为 replay 从不经过它；这个限制只在 RLinf 自己启动 Ray 时生效。

多卡训练
--------

``cluster.component_placement.supermimic`` 列出使用的 GPU，每张 GPU 一个 rank。各 rank 做同步数据并行：各自步进自己的环境、维护自己的 replay、采自己那份批量，在裁剪之前按样本数加权 all-reduce 梯度，所以每个 rank 做完全相同的更新。rank 0 广播初始权重并负责评测。``train/param_divergence`` 每 ``algorithm.param_check_interval`` 次更新检查一次各 rank 权重是否一致。

多用几张 GPU 有两种方式：

* **固定全局批量。** 配置不变，只加 GPU。全局批量、学习率和更新数据比都和单卡相同，每个 rank 分一份；相当于把单卡训练摊到多张卡上。

  .. code-block:: bash

     bash examples/embodiment/run_supermimic.sh supermimic_lafan1 \
         cluster.component_placement.supermimic=0-3

* **论文双卡配方。** 每张 GPU 保持单卡的规模，所以全局批量翻倍，梯度在两张卡之间取平均，学习率为 :math:`2\sqrt{2} \times 10^{-4}`\ （表 A.5）：

  .. code-block:: bash

     export SUPERMIMIC_BONES_PATH=/data/bones_g1_filtered
     bash examples/embodiment/run_supermimic.sh supermimic_bones_seed_2gpu

  LAFAN1 用同样的覆盖项：``cluster.component_placement.supermimic=0-1 env.num_envs=8192 algorithm.wm_window_batch=4096 algorithm.imagine_batch=8192 algorithm.lr=2.8284271e-4``。

GPU 上的集合通信用 NCCL，超时由 ``RLINF_TIMEOUT``\ （分钟）决定。如果某个 rank 出错，RLinf 的 worker group 会立即停掉 driver 和其他 rank，不会等到这个超时。

用 Slurm 跑多节点（实验性）
---------------------------

``examples/embodiment/slurm_supermimic.sh`` 在每个 Slurm 节点上启动一个 Ray 节点，在每次 ``ray start`` 之前导出唯一的 ``RLINF_NODE_RANK``，并在第一个节点上运行 driver，placement 覆盖所有 GPU。分区、账号和 GPU 数来自你的 ``sbatch`` 选项；``log_path`` 和数据必须放在所有节点都能读写的存储上。

.. code-block:: bash

   sbatch --nodes=2 --gpus-per-node=8 --partition=<partition> \
       examples/embodiment/slurm_supermimic.sh supermimic_lafan1 \
       env.motion_path=/shared/lafan1_g1 runner.logger.log_path=/shared/runs/lafan1

这个脚本只用打桩的 ``srun`` 和 ``ray`` 检查过，还没有在 Slurm 集群上运行。

检查点与恢复
------------

``runner.save_interval`` 每隔 N 次更新写一个 ``log_path/<experiment_name>/global_step_<N>``，最后一次更新总会写。检查点包括 ``supermimic.pt``\ （网络、优化器、学习率调度、计数器、重置分箱统计）、每个 rank 一个 ``rank_<r>.pt``\ （随机数状态），以及 ``checkpoint_meta.json``\ （文件大小和 SHA-256、world size、模型维度、Goal 窗口、格式版本）。文件先写到 ``.tmp`` 目录，所有 rank 写完后再改名，所以中断的保存不会留下半成品。

.. code-block:: bash

   bash examples/embodiment/run_supermimic.sh supermimic_lafan1 \
       runner.logger.log_path=/data/runs/lafan1 runner.resume_dir=auto

``runner.resume_dir=auto`` 选这次运行里最新的完整检查点；给路径则用那个检查点。格式、维度或 Goal 窗口不同的检查点会被拒绝。rank 数不同时，权重和优化器照常加载，随机数流从种子重新开始。replay 和仿真状态不保存，恢复后要先重新预热 replay，所以这是接着训练，不是精确复原。

评测
----

``evaluations/eval_supermimic.py`` 只从检查点加载网络，单卡运行，不需要 Ray。每个训练配置在 ``evaluations/supermimic/`` 都有对应的评测配置；给一个装着多个检查点的目录时选最新的那个。

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       eval.checkpoint=/data/runs/lafan1/supermimic_lafan1
   bash evaluations/run_eval.sh supermimic_velocity_eval \
       eval.checkpoint=/data/runs/velocity/supermimic_velocity

**动作跟踪** 把每条片段从第 0 帧播放一次，关闭传感器噪声和动作噪声，结果写到 ``per_clip.csv``。归一化回报是奖励之和除以 0.1 乘片段的计划长度，所以提前失败时没播放的部分仍留在分母里。

**速度跟踪** 用 ``eval.velocity`` 选择：

* ``g2``\ （速度评测配置的默认值）运行附录 G.2 阶跃测试：在 (vx, vy)、(vx, wz)、(vy, wz) 三个平面上各取 15 × 15 网格，去掉站立命令，共 672 条命令，每条 4 次试验，从 mjlab 的站立姿态开始执行 5 s。每次试验的值是最后 2 s 平均速度误差的范数除以命令范数；每条命令报告没有摔倒的试验的中位数；4 次都摔倒的命令没有数值（CSV 里是 NaN，``summary.json`` 里是 ``null``），计入 ``eval_velocity_g2/commands_all_trials_fell``。论文没有说明试验之间什么不同，本实现让仿真上下文和初始关节噪声各不相同。结果写到 ``velocity_g2.csv`` 和 ``eval_velocity_g2/*``。
* ``smoke`` 从参考起点开始保持 7 条固定命令，写出 ``velocity_smoke.csv`` 和 ``eval_velocity_smoke/*``。它只检查速度编码器能跑通，不是论文指标。

也可以用 ``runner.only_eval=true`` 走训练入口评测，此时用训练配置的环境数。设置了 ``runner.val_check_interval`` 时，训练每隔这么多次更新评测一次，最后一次更新后也会评测。

指标与日志
----------

日志走 RLinf 的 ``MetricLogger``\ （TensorBoard、WandB、SwanLab，都在 ``log_path`` 下）。多个 rank 时，计数求和，计时和显存取最大的 rank，共享计数取 rank 0，损失取平均；``runner.log_per_rank_metrics=true`` 会再写一份 ``rank_<r>/`` 前缀的指标。常用的键：

* ``wm/one_step_loss``、``wm/k_step_loss``：世界模型一步和八步之后的误差。
* ``imagination/survival``、``imagination/termination_prob``：32 步存活权重和平均终止概率。
* ``actor/grad_norm``\ （裁剪前）、``actor/encoder_grad_norm``、``actor/decoder_grad_norm``。
* ``critic/loss``、``critic/value_mean``、``critic/target_mean``。
* ``reset/bin_return_mean``、``reset/bin_return_min``：动作重置分箱的回报。
* ``train/nonfinite_skips``、``train/param_divergence``、``time/comm_s``。
* ``memory/torch_peak_allocated``、``memory/torch_peak_reserved``\ （PyTorch 分配器）、``memory/warp_mempool_peak``\ （Warp 内存池）、``memory/device_used``\ （驱动看到的整张 GPU）。这几个计数互相重叠，不能相加。

故障排查
--------

* **CUDA out of memory。** 报错会写出 rank 和它分到的规模。要保持批量就加 GPU，或者另用一个小规模配置；``toolkits/supermimic/profile_sim.py`` 可以测给定规模的显存。
* **Ray 因内存不足杀掉 worker。** 应该释放主机内存，而不是调高 ``RAY_memory_usage_threshold``；smoke 启动是唯一的例外。CPU 核多、内存少的机器上，Ray 会按核数预启动空闲 worker（每个约 50 MB），SuperMimic 用不到它们；设置 ``RAY_prestart_worker_first_driver=0`` 可以不启动这些进程，内存监控保持不变。
* **"No verified G1 reference clips"。** ``SUPERMIMIC_MOTION_PATH`` 或 ``env.motion_path`` 没有指向准备好的片段。
* **"reference clip q0 differs from the mjlab G1 default joint pose"。** 片段是用别的默认姿态转换的，需要重新转换。
* **检查点格式或 Goal 窗口被拒绝。** 检查点来自别的版本或配置，不能用当前配置恢复。
* **"linesearch iterations limit reached"。** 未训练的 policy 会触发 MuJoCo Warp 求解器的这条警告，不会中断训练。

性能与硬件
----------

``toolkits/supermimic/profile_sim.py`` 对一组环境数逐个起子进程，测采集吞吐、更新耗时、replay 大小和上面那几个显存计数：

.. code-block:: bash

   python toolkits/supermimic/profile_sim.py --motion-path /data/lafan1_g1 \
       --num-envs 256 1024 4096 --steps 30 --warmup 20 --update-iters 3 \
       --out profile.json

在 WSL2 下的一张 8 GB 笔记本 GPU 上：1024 个环境每秒采集约 3,900 个环境转移；用论文批量四分之一（1024 条想象 rollout、512 个窗口）做一次更新约 21 s，PyTorch 峰值 4.2 GB；论文批量下一次更新需要约 11 GB，超过这张卡的显存。这些数字来自受 kernel 启动开销限制的笔记本环境，不能用来推测数据中心 GPU 的表现；正式训练前请先在自己的硬件上测出每次更新的秒数。

已经运行过的内容
----------------

在一张 8 GB GPU 上用 MuJoCo Warp 和 LAFAN1 片段跑过：LAFAN1 各配置（680 维、689 维、730 维、99 维观测、速度）的小规模短训练，包括 driver 被强杀后的恢复，动作和速度评测，以及 G.2 阶跃测试。多卡的数值逻辑用两个 CPU（Gloo）进程测试过。尚未运行：论文规模训练、真实 GPU 上的 NCCL、多节点、Slurm、BONES-SEED 数据，以及足以证明 policy 学会跟踪的长时间训练。
