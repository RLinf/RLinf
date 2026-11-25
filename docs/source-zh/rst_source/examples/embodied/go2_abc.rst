用 PPO 训练 Go2 运动策略
===============================

用 RLinf PPO 在 Isaac Lab 中训练 Go2 的运动 policy，控制 12 个关节。任务 A 在平地跟踪机体速度指令，B 穿越崎岖地形，C 爬楼梯。每个 policy 读取 48 个本体状态和指令值，以及 187 个地形高度采样值。

**前置条件：** 按 :doc:`安装说明 </rst_source/start/installation>` 在配备 NVIDIA GPU 的机器上准备 RLinf。

安装与运行
----------

.. tab-set::

   .. tab-item:: Isaac Lab

      **1. 安装 Go2 运行环境**

      安装脚本会准备 Isaac Lab 2.3.2、Isaac Sim 5.1、评测所需的 RSL-RL 3.1.2，以及 `Go2 运动基准库 <https://github.com/viktoriageorrg04/bsc-thesis-multi-task-RL-env>`_ 中的三项仿真任务。

      .. code-block:: bash

         bash requirements/install.sh embodied --env go2_abc
         source .venv/bin/activate

      在已激活的环境中运行 ``python -c "import isaacsim"``，按终端提示确认 `NVIDIA 许可条款 <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html#running-isaac-sim>`_。提交批处理作业时，同意条款后可在作业环境中设置 ``OMNI_KIT_ACCEPT_EULA=YES``。

      **2. 准备初始权重**

      在 RLinf 仓库根目录下载楼梯任务的冻结初始权重。以下命令会先校验 SHA-256，再将权重放入训练数据目录。

      .. code-block:: bash

         python -m examples.embodiment.go2_abc.prepare_baseline C

      C 配置读取 ``GO2_ABC_DATA`` 目录下的 ``models/C/baseline.pt``，默认目录为 ``data/go2_abc``。将命令末尾的任务名换成 A 或 B，即可下载对应任务的初始权重。

      使用自己的权重时，将 ``actor.model.model_path`` 设为对应文件路径，权重需包含输入 235 维、输出 12 维的 actor 和配套的 critic。从头训练可将该项设为 ``null``。

      **3. 训练策略**

      从 RLinf 仓库根目录启动楼梯任务续训。

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh isaaclab_go2_c_ppo

      平地任务使用 ``isaaclab_go2_a_ppo``，越野任务使用 ``isaaclab_go2_b_ppo``。A 和 B 默认使用 2,048 个并行环境，C 使用 1,024 个。训练共更新 1,000 次，每 50 次保存一份 actor 权重。启动日志和运行目录位于 ``logs/`` 下。

      Go2 的 actor 和 critic 从第一步起一起训练，共用 ``actor.optim.lr`` 设置的学习率。默认训练种子为 0，在启动命令末尾添加 ``actor.seed=2``，即可将 actor、rollout 和训练环境的种子一并设为 2。

      如果使用本地 Isaac Sim 5.1 资产镜像，可直接用 Hydra 启动同一份训练配置。将下面的资产目录换成本机路径。

      .. code-block:: bash

         export GO2_ASSET_ROOT=/absolute/path/to/isaac_sim_5.1
         python -m examples.embodiment.train_embodied_agent \
           --config-path "$PWD/examples/embodiment/config" --config-name isaaclab_go2_c_ppo \
           "env.train.init_params.asset_root=$GO2_ASSET_ROOT" "runner.logger.log_path=$PWD/logs/go2-stairs"

评测训练好的策略
----------------

在 RLinf 仓库根目录下载 A、B、C 三项任务的评测脚本。

.. code-block:: bash

   (
     set -e
     runtime=../public-evaluator
     mkdir -p "$runtime/a40" "$runtime/benchmark/scripts"
     wrappers=https://raw.githubusercontent.com/QuincyQAQ/RLinf/b405f67382ce63af2401a2a9ff5b0b3c7f28ccb7/a40
     for module in evaluate_continuation evaluate_dual passage_metrics b1_numerical_boundary; do
       curl -fsSL "$wrappers/$module.py" -o "$runtime/a40/$module.py"
     done
     benchmark=https://raw.githubusercontent.com/viktoriageorrg04/bsc-thesis-multi-task-RL-env/743ac04d9543b04b7854f1100bfa0cc69c993106
     curl -fsSL "$benchmark/scripts/evaluate.py" -o "$runtime/benchmark/scripts/evaluate.py"
   )

在 Slurm 分配的 GPU 节点上，选择训练任务及其日志目录。以下命令使用前面楼梯任务的目录。通过标准脚本启动的训练，请选用 ``logs/`` 下对应的时间戳目录。训练种子按该权重的实际值填写，未知时可省略种子参数。

.. code-block:: bash

   TASK=C
   RUN_DIR=logs/go2-stairs
   CHECKPOINT="$RUN_DIR/${TASK}_rlinf/checkpoints/global_step_1000/actor/model_state_dict/full_weights.pt"
   python -m examples.embodiment.go2_abc.evaluate --task "$TASK" --train-seed 0 "$CHECKPOINT" "$RUN_DIR/eval"

评测平地时设为 ``TASK=A``，崎岖地面设为 ``TASK=B``。每份权重使用新的输出目录。本地资产镜像可通过 ``--asset-root "$GO2_ASSET_ROOT"`` 指定。若要核对 actor 和 critic 是否都经过更新，可用 ``--initial-checkpoint`` 传入此次训练实际使用的初始权重。

评测使用固定种子 23456 和 34567，每个种子在 256 个并行环境中采集至少 1,024 个回合。两个种子目录分别保存逐回合记录，合并后的速度跟踪成功数和成功率写入 ``summary.json``。任务 C 另统计楼梯通过数和通过率。

复现 RSL-RL PPO 对照组时，将固定版本的 `train.py <https://github.com/viktoriageorrg04/bsc-thesis-multi-task-RL-env/blob/743ac04d9543b04b7854f1100bfa0cc69c993106/scripts/train.py>`_ 放到 ``evaluate.py`` 所在目录。将 `续训源码目录 <https://github.com/QuincyQAQ/RLinf/tree/b405f67382ce63af2401a2a9ff5b0b3c7f28ccb7/a40>`_ 中的 ``train_continuation_rsl.py`` 放入 ``../public-evaluator/a40/``。任务 B 还需从同一源码目录复制 ``train_legacy_distribution.py``。

在 `训练配置 <https://github.com/QuincyQAQ/RLinf/tree/23f16691d4dede5dab949a5cbf7934f6021e7ddf/examples/embodiment/go2_abc/config>`_ 中选择对应任务的 ``*_ppo.json``，按文件中的参数和环境变量启动 ``train_continuation_rsl.py``。将冻结权重和资产镜像路径替换为本机路径，``--seed`` 设为 0 或 2。

将当前 RLinf 仓库加入 ``PYTHONPATH``，并在训练和评测命令末尾添加 ``env.rewards.base_height.func=rlinf.envs.sim.isaaclab.tasks.go2:terrain_base_height_l2``，使两种 PPO 使用相同的地形高度奖励。

评测时，用同一任务的 ``*_baseline.json`` 启动 ``evaluate_continuation.py``。将 ``--checkpoint`` 指向冻结输入，或指向完成 1,000 次更新后保存的 RSL-RL 权重（A、B 为 ``model_999.pt``，C 为 ``model_960.pt``）。

三个任务用种子 2 训练 RSL-RL 策略时，都要把训练目录名和评测报告标签的后缀从 ``_s0`` 改为 ``_s2``。例如，任务 A 训练时用 ``--seed 2 --run_name A_ppo_s2``，评测这份权重时用 ``--report_train_task A2_omni_s2``。

两次评测分别将 ``GO2_EVAL_SEED`` 设为 23456 和 34567，每次用 ``--output_dir`` 指定新的输出目录。

仿真评测结果
------------

每个任务使用一份基线权重。RSL-RL PPO 和 RLinf PPO 都从这份权重出发，分别用训练种子 0 和 2 续训 1,000 次更新。冻结基线只评测一次，两个训练种子共用这一结果。

表中成功率按评测种子 23456 和 34567 的全部回合合并计算，括号内为成功回合数与总回合数。任务 C 分别统计速度跟踪和楼梯通过，后者根据 Go2 在仿真中的运动轨迹判定。

.. list-table::
   :header-rows: 1

   * - 指标
     - 训练种子
     - 冻结基线
     - RSL-RL PPO
     - RLinf PPO (Huber)
   * - A 速度跟踪
     - 0
     - 91.47% (1887/2063)
     - 93.33% (1917/2054)
     - 96.78% (1984/2050)
   * - A 速度跟踪
     - 2
     - 91.47% (1887/2063)
     - 90.90% (1878/2066)
     - 98.29% (2013/2048)
   * - B 速度跟踪
     - 0
     - 99.85% (2045/2048)
     - 99.61% (2040/2048)
     - 99.85% (2045/2048)
   * - B 速度跟踪
     - 2
     - 99.85% (2045/2048)
     - 99.76% (2043/2048)
     - 99.80% (2044/2048)
   * - C 速度跟踪
     - 0
     - 99.85% (2045/2048)
     - 100.00% (2048/2048)
     - 100.00% (2048/2048)
   * - C 速度跟踪
     - 2
     - 99.85% (2045/2048)
     - 99.80% (2044/2048)
     - 99.90% (2046/2048)
   * - C 楼梯通过
     - 0
     - 21.39% (438/2048)
     - 44.82% (918/2048)
     - 56.69% (1161/2048)
   * - C 楼梯通过
     - 2
     - 21.39% (438/2048)
     - 33.50% (686/2048)
     - 62.06% (1271/2048)

在 CPU 上回放
-------------

使用步骤 2 下载的冻结权重，回放一帧合成传感器数据。输出记录时间戳、12 个目标关节角（rad）和 12 个电机力矩（N·m），均按 SDK 关节顺序排列。

.. code-block:: bash

   TASK=C
   REPLAY_DIR=$(mktemp -d)
   python -c 'import json; print(json.dumps(json.load(open("tests/fixtures/go2_control_reference.json"))["frame"]))' > "$REPLAY_DIR/frame.jsonl"
   python -m examples.embodiment.go2_abc.deploy --task "$TASK" --group baseline \
     --frames "$REPLAY_DIR/frame.jsonl" --output "$REPLAY_DIR/commands.jsonl"

要回放训练后的策略，将 ``TASK`` 设为对应任务，并将 ``--group baseline`` 换成 ``--checkpoint "$CHECKPOINT"``。按组别加载权重时可使用 ``--data`` 指定数据根目录。

回放自己的记录时，按 ``frame.jsonl`` 的格式提供关节和 IMU 数据、实测机身线速度及地形高度。所有时间戳使用同一单调时钟，每 5 ms 采样一次，policy 每四帧更新一次。机身与地面高度使用同一世界坐标系。高度网格随机身偏航角旋转，间距为 0.1 m，按 y 从 −0.5 到 0.5 m 排列各行，每行按 x 从 −0.8 到 0.8 m 排列。
