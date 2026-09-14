Psi0 + SIMPLE 评测与训练
========================

本页介绍如何通过 RLinf 在 SIMPLE 的八个 Teleop 任务上评测 Psi0，并在
OpenOven 上进行四卡 PPO 后训练。每个任务必须使用对应的 checkpoint 和 reset
states；这些资产不能跨任务复用。

适用范围与版本固定
------------------

安装脚本已固定
`Psi0 模型代码 <https://github.com/physical-superintelligence-lab/Psi0/tree/a32e57a3fabb8590c80677f9cd3d1fc3db60eb06>`_
和
`SIMPLE 环境代码 <https://github.com/physical-superintelligence-lab/SIMPLE/tree/5e3d6f84e85343e34e9bca8d157f0d7813231185>`_。
下面下载命令中的 ``--revision`` 分别固定模型权重和 reset 数据的仓库快照，
避免代码或资产更新影响复现，无需另行设置版本。

当前范围是固定 Psi0 发布中具有 task-specific checkpoint 的八个 Teleop 任务。
``G1WholebodyBendPickMP-v0`` 和 ``G1WholebodyTabletopGraspMP-v0`` 使用另一套
state、action 和 System-0 协议，暂不支持。

安装 Runtime
------------

.. code-block:: bash

   bash requirements/install.sh embodied \
     --venv .venv-psi0-simple \
     --model psi0 \
     --env simple
   source .venv-psi0-simple/bin/activate


选择任务与下载资产
------------------

下表给出
`固定 Psi0 checkpoint 发布 <https://huggingface.co/USC-PSI-Lab/psi-model/tree/d34a91932d25c45ef211582315b9224c7dc8ace9/psi0/simple-checkpoints>`_
中的八个可替换 Teleop 任务。``官方结果`` 是
`SIMPLE 固定版本结果表 <https://github.com/physical-superintelligence-lab/SIMPLE/blob/5e3d6f84e85343e34e9bca8d157f0d7813231185/README.md#-simulation-benchmarking-results>`_
中 Psi0 在 level 0/1/2 各 10 条轨迹上的成功次数。

.. list-table::
   :header-rows: 1
   :widths: 24 48 12 16

   * - ``SIMPLE_TASK``
     - run 名
     - horizon
     - 官方结果
   * - ``G1WholebodyXMovePickTeleop-v0``
     - ``g1wholebodyxmovepick-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2604022205``
     - 800
     - 10/10/6
   * - ``G1WholebodyHandoverTeleop-v0``
     - ``g1wholebodyhandover-v0.simple.flow1000.cosine.lr1.0e-04.b64.gpus4.2604071507``
     - 800
     - 7/7/10
   * - ``G1WholebodyLocomotionPickBetweenTablesTeleop-v0``
     - ``g1wholebodylocomotionpickbetweentablesteleop-v0.simple.flow1000.cosine.lr1.0e-04.b64.gpus4.2604081126``
     - 1200
     - 7/5/6
   * - ``G1WholebodyXMoveBendPickTeleop-v0``
     - ``g1wholebodyxmovebendpickteleop-v0.simple.flow1000.cosine.lr1.0e-04.b112.gpus7.2604100422``
     - 800
     - 10/9/9
   * - ``G1WholebodyCloseDoorTeleop-v0``
     - ``g1wholebodyclosedoorteleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605070100``
     - 1000
     - 10/10/10
   * - ``G1WholebodyOpenOvenTeleop-v0``
     - ``g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604``
     - 1000
     - 7/5/4
   * - ``G1WholebodyOpenFaucetTeleop-v0``
     - ``g1wholebodyopenfaucetteleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605081439``
     - 1000
     - 3/3/4
   * - ``G1WholebodyPickAndPlaceAndHugContainerTeleop-v0``
     - ``g1wholebodypickandplaceandhugcontainerteleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2604280201``
     - 1000
     - 7/6/3

任务专属资产
~~~~~~~~~~~~

以下命令以 OpenOven 为例。资产根目录和 ``HF_HOME`` 由所有任务共用，只需设置
一次。切换任务时，仅修改注释标出的 ``SIMPLE_TASK`` 和 run 名；两者必须来自上表
同一行，其余命令会据此生成对应路径。

.. code-block:: bash

   # Set once. Reuse for every SIMPLE task.
   export PSI0_ARTIFACT_ROOT=/mnt/public2/yangtingyuan/RLinf_yty/checkpoint/psi0-simple-artifacts
   export HF_HOME="${PSI0_ARTIFACT_ROOT}/hf-cache"

   # Replace these two values together for each task.
   export SIMPLE_TASK=G1WholebodyOpenOvenTeleop-v0  # Replace the task ID.
   # Replace only the run name after psi0/simple-checkpoints/.
   export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604

   mkdir -p "${PSI0_ARTIFACT_ROOT}"
   hf download USC-PSI-Lab/psi-model \
     "${PSI0_RUN_REL}/argv.txt" \
     "${PSI0_RUN_REL}/run_config.json" \
     "${PSI0_RUN_REL}/checkpoints/ckpt_40000/model.safetensors" \
     --revision d34a91932d25c45ef211582315b9224c7dc8ace9 \
     --local-dir "${PSI0_ARTIFACT_ROOT}/psi-model"

   hf download USC-PSI-Lab/psi-data "simple-eval/${SIMPLE_TASK}.zip" \
     --repo-type dataset \
     --revision 6eeff2d02fdaac5dd4f3e84244fc83d3fad2c203 \
     --local-dir "${PSI0_ARTIFACT_ROOT}/psi-data"

   export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"
   export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/${SIMPLE_TASK}"
   mkdir -p "${SIMPLE_RESET_DIR}"
   unzip -q "${PSI0_ARTIFACT_ROOT}/psi-data/simple-eval/${SIMPLE_TASK}.zip" \
     -d "${SIMPLE_RESET_DIR}"


公共 Qwen Metadata
~~~~~~~~~~~~~~~~~~

Psi0 的联合 checkpoint 已包含 System-2 的 ``vlm_model`` 权重和 System-1 的
``action_header`` 权重，不会另行下载 Qwen 模型权重。固定 loader 仍会通过模型 ID
读取 Qwen 配置、tokenizer 和 processor。以下公共 metadata 在同一 ``HF_HOME``
中只需下载一次，切换任务时不要重复执行：

.. code-block:: bash

   hf download Qwen/Qwen3-VL-2B-Instruct \
     chat_template.json config.json generation_config.json merges.txt \
     preprocessor_config.json tokenizer.json tokenizer_config.json \
     video_preprocessor_config.json vocab.json

上游 loader 默认解析 ``main``。仅按提交 SHA 下载不会建立缓存中的 ``main``
引用，离线加载时可能找不到已下载的快照，因此这里沿用 loader 的默认版本，写入
``HF_HOME`` 下的标准缓存，而非任意 ``--local-dir``。缓存完整后可离线复用。
这部分 metadata 与上面固定版本的资产不同：``main`` 可能更新，并非严格版本锁定；
若要锁定，还需让 loader 显式使用相同 revision。


运行 Eval
---------

仓库提供 CloseDoor、OpenFaucet 和 OpenOven 三份直接配置：

.. code-block:: text

   evaluations/simple/simple_closedoor_psi0_eval.yaml
   evaluations/simple/simple_openfaucet_psi0_eval.yaml
   evaluations/simple/simple_openoven_psi0_eval.yaml

OpenOven
~~~~~~~~

OpenOven 的任务、horizon、episode 起点和视频参数均由配置提供，
最小 smoke 只需覆盖本地资产路径：

.. code-block:: bash

   bash evaluations/run_eval.sh simple simple_openoven_psi0_eval \
     rollout.model.model_path="${PSI0_RUN_DIR}" \
     env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}"

完整的三个 DR level 各运行 10 条轨迹：

.. code-block:: bash

   for SIMPLE_DR_LEVEL in 0 1 2; do
     bash evaluations/run_eval.sh simple simple_openoven_psi0_eval \
       rollout.model.model_path="${PSI0_RUN_DIR}" \
       env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}" \
       env.eval.reset_dataset.dr_level="${SIMPLE_DR_LEVEL}" \
       env.eval.reset_dataset.num_episodes=10 \
       env.eval.rollout_epoch=10
   done

切换其他任务
~~~~~~~~~~~~

切换其他 Teleop 任务可以复用 OpenOven 配置。资产已下载并解压时无需重下，
但每次切换都要重新设置下面四个变量；已导出的路径不会随任务或 run 名自动更新。
以 XMovePick 为例：

.. code-block:: bash

   export SIMPLE_TASK=G1WholebodyXMovePickTeleop-v0
   export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyxmovepick-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2604022205
   export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"
   export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/${SIMPLE_TASK}"

   bash evaluations/run_eval.sh simple simple_openoven_psi0_eval \
     rollout.model.model_path="${PSI0_RUN_DIR}" \
     env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}" \
     env.eval.init_params.task_id="simple/${SIMPLE_TASK}" \
     env.eval.max_episode_steps=800 \
     env.eval.max_steps_per_rollout_epoch=816

每次切换任务都要同步更新上述四个变量，并按下表选择两项步数，保证 checkpoint、
reset 数据与 ``task_id`` 属于同一任务。三档完整测评的循环也需保留两项路径覆盖，
并补上 ``task_id`` 和两项步数。``max_steps_per_rollout_epoch`` 是不小于任务
horizon 的最小 24 的倍数。

.. list-table::
   :header-rows: 1
   :widths: 45 25 30

   * - 任务
     - ``max_episode_steps``
     - ``max_steps_per_rollout_epoch``
   * - XMovePick、Handover、XMoveBendPick
     - 800
     - 816
   * - CloseDoor、OpenOven、OpenFaucet、PickAndPlaceAndHugContainer
     - 1000
     - 1008
   * - LocomotionPickBetweenTables
     - 1200
     - 1200

所有八个任务都固定为 ``mujoco_isaac``、50 Hz render、200 Hz physics、最多
300 步 stabilization，以及 30/24/6 RTC。官方结果每档只有 10 条轨迹；应分别
报告三个 level，不能把单条 smoke 或总成功数当作严格复现结论。

运行 OpenOven PPO 后训练
------------------------

先确认 ``PSI0_RUN_DIR`` 和 ``SIMPLE_RESET_DIR`` 对应 OpenOven。
当前训练配置使用 DR0、600 步上限、四卡四环境；8 轮 rollout 共采集 32 条轨迹，
``global_batch_size=200``、``update_epoch=1``，每轮训练执行 4 次优化器更新。

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh simple_openoven_ppo_psi0

切换其他 Teleop 任务时，先按下载段更新 ``PSI0_RUN_DIR`` 和 ``SIMPLE_RESET_DIR``。
这份配置的 actor、rollout 模型路径和训练 reset 路径会读取这两个变量，无需重复覆盖。
同时在 ``examples/embodiment/config/simple_openoven_ppo_psi0.yaml`` 中修改
``env.train.init_params.task_id``、``env.train.max_episode_steps`` 和
``env.train.max_steps_per_rollout_epoch``，任务 ID 与步数对应关系见前面的测评段。
修改 horizon 后，还需检查 ``actor.global_batch_size`` 与每轮采样 chunk 数的整除关系。
这些任务字段在 YAML 中修改，不追加到上述 bash 命令后。

测评训练后的 checkpoint 时，保持任务、reset 数据、DR level 和 horizon 一致。
下面命令对齐当前 OpenOven 的官方 1000 步 horizon 测评配置。

value head 是 PPO 训练时估计未来回报的 critic，其参数也保存在 ``full_weights.pt``
中。``add_value_head=true`` 是为了创建相同结构并严格加载这些已有参数，
并非重新训练 critic；测评生成动作时不使用它。``PSI0_RUN_DIR`` 仍指向该任务
原始 run 目录，用于构建模型和 processor，再由 ``runner.ckpt_path`` 加载训练后权重。

.. code-block:: bash

   export PSI0_PPO_CHECKPOINT=/absolute/path/to/global_step_20

   bash evaluations/run_eval.sh simple simple_openoven_psi0_eval \
     rollout.model.model_path="${PSI0_RUN_DIR}" \
     rollout.model.add_value_head=true \
     runner.ckpt_path="${PSI0_PPO_CHECKPOINT}/actor/model_state_dict/full_weights.pt" \
     env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}" \
     env.eval.reset_dataset.num_episodes=10 \
     env.eval.rollout_epoch=10
