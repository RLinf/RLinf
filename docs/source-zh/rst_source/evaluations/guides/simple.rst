Psi0 + SIMPLE 评测
====================

通过 RLinf 标准评测入口，在 SIMPLE 的 CloseDoor 或 OpenFaucet Teleop 任务上运行
对应的 task-specific Psi0 checkpoint。
当前接入仅支持评测：Psi0 System-2 和 SIMPLE controller 保持冻结，采样使用官方
deterministic Euler 路径，每个 30 步 RTC plan 实际执行前 24 步。

范围与版本
----------

.. list-table::
   :header-rows: 1
   :widths: 24 46 30

   * - 组件
     - 固定版本
     - 作用
   * - `Psi0 <https://github.com/physical-superintelligence-lab/Psi0/tree/a32e57a3fabb8590c80677f9cd3d1fc3db60eb06>`_
     - ``a32e57a3fabb8590c80677f9cd3d1fc3db60eb06``
     - 冻结 VLM 与 eval-only action expert
   * - `SIMPLE <https://github.com/physical-superintelligence-lab/SIMPLE/tree/5e3d6f84e85343e34e9bca8d157f0d7813231185>`_
     - ``5e3d6f84e85343e34e9bca8d157f0d7813231185``
     - CloseDoor/OpenFaucet task 与 decoupled-WBC runtime
   * - Psi0 checkpoint
     - CloseDoor 或 OpenFaucet 对应 run 的 ``ckpt_40000``
     - task-specific 评测策略
   * - SIMPLE reset states
     - 与所选 task 同名的 LeRobot v2.1 episodes
     - 固定 episode 初始状态

安装 Runtime
------------

使用 Python 3.10 和专用 model/environment installer：

.. code-block:: bash

   bash requirements/install.sh embodied \
     --venv .venv-psi0-simple \
     --model psi0 \
     --env simple
   source .venv-psi0-simple/bin/activate

Psi0/SIMPLE installer 分支会自动选择 Python 3.10，随后检出固定的
Psi0/SIMPLE revisions、初始化 SIMPLE 固定 controller
submodules，并构建 CuRobo extension。它不会安装 SIMPLE 可选的 RLDS/TensorFlow
数据栈，也不会下载 Psi0 checkpoint 或 reset-state dataset。验证 installer 修改时
应使用新的 ``--venv`` 路径；复用已有环境也会复用其中原有的 package。

下载模型与数据
--------------

当前 CloseDoor Eval 只需要一个 task-specific 联合 checkpoint。该 checkpoint 的
``model.safetensors`` 已同时包含 System-2 的 ``vlm_model.*`` 和 System-1 的
``action_header.*``。``argv.txt`` 中列出的两个 foundation checkpoint 只是
fine-tuning 初始化来源，Eval 不会再次加载它们。

`SIMPLE checkpoint 发布树 <https://huggingface.co/USC-PSI-Lab/psi-model/tree/d34a91932d25c45ef211582315b9224c7dc8ace9/psi0/simple-checkpoints>`_
包含一组任务专属 checkpoint，不是一个覆盖全部 SIMPLE 任务的统一 checkpoint。
当前 adapter 支持 CloseDoor 和 OpenFaucet Teleop；两者不能互换 checkpoint。
不要为这两个 Eval 使用 BendPick MP 或其他任务的 checkpoint。不同任务还可能使用不同的
controller/System-0 路径。

先设置专用 artifact 根目录和 Hugging Face cache。后续启动 Eval 时保留同一个
``HF_HOME``：

.. code-block:: bash

   export PSI0_ARTIFACT_ROOT=/absolute/path/to/psi0-simple-artifacts
   export HF_HOME="${PSI0_ARTIFACT_ROOT}/hf-cache"
   mkdir -p "${PSI0_ARTIFACT_ROOT}" "${HF_HOME}"

下载固定 revision 的 CloseDoor run directory。不要只下载
``model.safetensors``；loader 还会读取同目录的 ``argv.txt`` 和
``run_config.json``：

.. code-block:: bash

   export PSI0_MODEL_REV=d34a91932d25c45ef211582315b9224c7dc8ace9
   export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyclosedoorteleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605070100

   hf download USC-PSI-Lab/psi-model \
     "${PSI0_RUN_REL}/argv.txt" \
     "${PSI0_RUN_REL}/envs.txt" \
     "${PSI0_RUN_REL}/run_config.json" \
     "${PSI0_RUN_REL}/checkpoints/ckpt_40000/model.safetensors" \
     --revision "${PSI0_MODEL_REV}" \
     --local-dir "${PSI0_ARTIFACT_ROOT}/psi-model"

   export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"

Ψ₀ 的固定 loader 使用
`Qwen/Qwen3-VL-2B-Instruct <https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/89644892e4d85e24eaac8bacfd4f463576704203>`_
创建模型结构和 processor。
策略权重已经在上面的联合 checkpoint 中，因此这里只预取 config、tokenizer 和
processor 元数据，不下载 Qwen 的 ``model.safetensors``：

.. code-block:: bash

   hf download Qwen/Qwen3-VL-2B-Instruct \
     chat_template.json config.json generation_config.json merges.txt \
     preprocessor_config.json tokenizer.json tokenizer_config.json \
     video_preprocessor_config.json vocab.json \
     --revision 89644892e4d85e24eaac8bacfd4f463576704203

.. note::

   固定上游 loader 仍通过模型 ID 解析 Qwen repository，并没有接收 revision 参数。
   上述命令预取已核对的 metadata snapshot，但首次 load 仍应保留网络访问，以便
   Hugging Face 把该模型 ID 解析到 cache。当前接入不宣称完全离线加载。

下载并解压同任务的固定 reset states：

.. code-block:: bash

   export SIMPLE_DATA_REV=6eeff2d02fdaac5dd4f3e84244fc83d3fad2c203
   export SIMPLE_RESET_REL=simple-eval/G1WholebodyCloseDoorTeleop-v0.zip

   hf download USC-PSI-Lab/psi-data "${SIMPLE_RESET_REL}" \
     --repo-type dataset \
     --revision "${SIMPLE_DATA_REV}" \
     --local-dir "${PSI0_ARTIFACT_ROOT}/psi-data"

   export SIMPLE_RESET_ZIP="${PSI0_ARTIFACT_ROOT}/psi-data/${SIMPLE_RESET_REL}"
   export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/closedoor"
   mkdir -p "${SIMPLE_RESET_DIR}"
   unzip -q "${SIMPLE_RESET_ZIP}" -d "${SIMPLE_RESET_DIR}"

校验两个实际运行 artifact。checkpoint 约为 6.25 GB；reset-state zip 约为
421 KB：

.. code-block:: bash

   printf '%s  %s\n' \
     c632adea90e2d8bfee92559b091e8f8ea465aae0a5ec23a244390bf12c3fce50 \
     "${PSI0_RUN_DIR}/checkpoints/ckpt_40000/model.safetensors" \
     | sha256sum --check
   printf '%s  %s\n' \
     6bff1d5fc5a332968e57eef78b781fe929c70ac833d040cc7e589e80cae2a286 \
     "${SIMPLE_RESET_ZIP}" \
     | sha256sum --check
   test -f "${PSI0_RUN_DIR}/argv.txt"
   test -f "${PSI0_RUN_DIR}/run_config.json"
   find "${SIMPLE_RESET_DIR}" -path '*/level-0/meta/episodes.jsonl' -print -quit

最后一条命令必须只输出一个 LeRobot metadata 文件。Eval adapter 根据
``env.eval.reset_dataset.dr_level`` 选择 ``level-0``，并直接读取每个 episode 的
``environment_config``；reset 阶段不读取 Parquet/video payload。

controller/System-0 权重不需要单独从 Hugging Face 下载。installer 会从固定
SIMPLE checkout 及其 ``decoupled_wbc`` submodule 安装 CloseDoor 所需的
Balance/Walk ONNX 文件。

运行 Smoke Eval
---------------

单机 smoke 直接调用标准评测入口即可。RLinf 会优先连接已有 Ray cluster；没有
cluster 时会自动启动本地 Ray，因此不应在每次 Eval 前重复执行
``ray start --head``。smoke recipe 关闭了可选 metric backend，因此结果直接
输出到进程日志，不会导入 TensorBoard 或 TensorFlow：

.. code-block:: bash

   bash evaluations/run_eval.sh simple simple_closedoor_psi0_eval \
     rollout.model.model_path="${PSI0_RUN_DIR}" \
     env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}"

如果已有 Ray cluster 是由另一个虚拟环境启动的，切换环境前只需停止该旧 cluster
一次。只有明确管理多机 cluster 时才需要预先启动 Ray。

smoke config 使用一个环境和一个 reset episode。adapter 最多执行官方 300 步
stabilization，然后按完整 24 步 chunk 执行，直到 task 成功或到达 1,000 步边界。
配置关闭自动 reset，保证一个 rollout epoch 就是一次官方 trial；episode 结束后
EnvWorker 与 RolloutWorker 同步停止。最后一个 chunk 中未执行的槽位只记录到
``executed_mask``，不会发送给 controller。

运行完整参考 Eval
-----------------

使用 3 个 DR level、每档 10 个固定 episode，复现官方 CloseDoor 的 30 次
评测。``simple_closedoor_psi0_eval`` 已经固定模型推理、动作执行和仿真协议；
命令行只需提供本地路径，并选择要运行的 reset episode。

配置中已经固定以下协议参数，无需在命令行重复设置：

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - 协议部分
     - 固定值
     - 作用
   * - 模型推理
     - bf16、10-step deterministic Euler
     - 与官方 Ψ₀ CloseDoor 推理设置一致
   * - RTC 动作执行
     - 每次规划 30 步，执行 24 步，保留 6 步重叠
     - 与官方 ``--action-exec-horizon=24 --rtc`` 一致
   * - 任务与仿真
     - ``G1WholebodyCloseDoorTeleop-v0``、``decoupled_wbc``、``mujoco_isaac``
     - 使用官方 Teleop controller 和仿真模式
   * - 时间与终止条件
     - 50 Hz control/render、200 Hz physics、最多 300 步 stabilization、1,000 步 horizon
     - 保持官方控制频率和 episode 边界

运行命令中的覆盖参数如下：

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - 参数
     - 本次取值
     - 含义
   * - ``rollout.model.model_path``
     - ``${PSI0_RUN_DIR}``
     - 指向包含 CloseDoor ``ckpt_40000`` 的本地模型目录
   * - ``env.eval.reset_dataset.path``
     - ``${SIMPLE_RESET_DIR}``
     - 指向已解压的官方固定 reset 数据
   * - ``env.eval.reset_dataset.dr_level``
     - ``0``、``1``、``2``
     - 由循环依次选择三个 DR level
   * - ``env.eval.reset_dataset.episode_start``
     - ``0``
     - 从每档的第一个固定 episode 开始
   * - ``env.eval.reset_dataset.num_episodes``
     - ``10``
     - 载入该档全部 10 个固定 episode
   * - ``env.eval.rollout_epoch``
     - ``10``
     - 执行 10 次 rollout，逐一消费上述 10 个 episode
   * - ``env.eval.video_cfg.fps``
     - ``50``
     - 仅设置输出视频播放帧率，不改变仿真速度或成功判定

.. note::

   ``num_episodes=10`` 决定可用的 reset episode 数，``rollout_epoch=10`` 决定
   实际执行次数。此配置关闭 auto-reset，因此两者都设为 10，才能让每个 episode
   恰好运行一次。

依次运行三个 DR level。每次调用都会创建独立的时间戳日志目录：

.. code-block:: bash
  unset HF_HUB_OFFLINE
  unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/mnt/public2/yangtingyuan/RLinf_yty/checkpoint/psi0-
  simple-artifacts/hf-cache

   for SIMPLE_DR_LEVEL in 0 1 2; do
     bash evaluations/run_eval.sh simple simple_closedoor_psi0_eval \
       rollout.model.model_path="${PSI0_RUN_DIR}" \
       env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}" \
       env.eval.reset_dataset.dr_level="${SIMPLE_DR_LEVEL}" \
       env.eval.reset_dataset.episode_start=0 \
       env.eval.reset_dataset.num_episodes=10 \
       env.eval.rollout_epoch=10 \
       env.eval.video_cfg.fps=50
   done

将结果与 SIMPLE 固定 revision 的
`官方结果表 <https://github.com/physical-superintelligence-lab/SIMPLE/blob/5e3d6f84e85343e34e9bca8d157f0d7813231185/README.md#-simulation-benchmarking-results>`_
对比：Ψ₀ 在 ``G1WholebodyCloseDoorTeleop-v0`` 上的 level 0/1/2 均为
10/10，合计 30/30。单次 smoke 的 1/1 只能证明链路打通，不能说明成功率已经
对齐。对齐时应检查三档汇总结果、逐 episode 日志和视频。官方
`SIMPLE Eval 入口 <https://github.com/physical-superintelligence-lab/SIMPLE/blob/5e3d6f84e85343e34e9bca8d157f0d7813231185/src/simple/cli/eval_decoupled_wbc.py>`_
与
`Ψ₀ SIMPLE server 参数 <https://github.com/physical-superintelligence-lab/Psi0/blob/a32e57a3fabb8590c80677f9cd3d1fc3db60eb06/scripts/deploy/serve_psi0_simple.sh>`_
可用于复核协议。

运行 OpenFaucet Eval
--------------------

OpenFaucet 使用独立的 task-specific checkpoint 和 reset states，不能复用
CloseDoor 的 ``PSI0_RUN_DIR`` 或 ``SIMPLE_RESET_DIR``。下载仍由用户执行：

.. code-block:: bash

   export PSI0_MODEL_REV=d34a91932d25c45ef211582315b9224c7dc8ace9
   export PSI0_OPENFAUCET_RUN_REL=psi0/simple-checkpoints/g1wholebodyopenfaucetteleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605081439

   hf download USC-PSI-Lab/psi-model \
     "${PSI0_OPENFAUCET_RUN_REL}/argv.txt" \
     "${PSI0_OPENFAUCET_RUN_REL}/envs.txt" \
     "${PSI0_OPENFAUCET_RUN_REL}/run_config.json" \
     "${PSI0_OPENFAUCET_RUN_REL}/checkpoints/ckpt_40000/model.safetensors" \
     --revision "${PSI0_MODEL_REV}" \
     --local-dir "${PSI0_ARTIFACT_ROOT}/psi-model"

   export PSI0_OPENFAUCET_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_OPENFAUCET_RUN_REL}"
   export SIMPLE_DATA_REV=6eeff2d02fdaac5dd4f3e84244fc83d3fad2c203
   export SIMPLE_OPENFAUCET_RESET_REL=simple-eval/G1WholebodyOpenFaucetTeleop-v0.zip

   hf download USC-PSI-Lab/psi-data "${SIMPLE_OPENFAUCET_RESET_REL}" \
     --repo-type dataset \
     --revision "${SIMPLE_DATA_REV}" \
     --local-dir "${PSI0_ARTIFACT_ROOT}/psi-data"

   export SIMPLE_OPENFAUCET_RESET_ZIP="${PSI0_ARTIFACT_ROOT}/psi-data/${SIMPLE_OPENFAUCET_RESET_REL}"
   export SIMPLE_OPENFAUCET_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/openfaucet"
   mkdir -p "${SIMPLE_OPENFAUCET_RESET_DIR}"
   unzip -q "${SIMPLE_OPENFAUCET_RESET_ZIP}" -d "${SIMPLE_OPENFAUCET_RESET_DIR}"

先运行一个 level-0 episode 验证链路：

.. code-block:: bash

   bash evaluations/run_eval.sh simple simple_openfaucet_psi0_eval \
     rollout.model.model_path="${PSI0_OPENFAUCET_RUN_DIR}" \
     env.eval.reset_dataset.path="${SIMPLE_OPENFAUCET_RESET_DIR}"

完整官方协议仍是三个 DR level、每档 10 个固定 episode：

.. code-block:: bash

   for SIMPLE_DR_LEVEL in 0 1 2; do
     bash evaluations/run_eval.sh simple simple_openfaucet_psi0_eval \
       rollout.model.model_path="${PSI0_OPENFAUCET_RUN_DIR}" \
       env.eval.reset_dataset.path="${SIMPLE_OPENFAUCET_RESET_DIR}" \
       env.eval.reset_dataset.dr_level="${SIMPLE_DR_LEVEL}" \
       env.eval.reset_dataset.episode_start=0 \
       env.eval.reset_dataset.num_episodes=10 \
       env.eval.rollout_epoch=10 \
       env.eval.video_cfg.fps=50
   done

``simple_openfaucet_psi0_eval`` 固定官方 1,000 步 horizon、50 Hz
control/render、200 Hz physics、最多 300 步 stabilization、decoupled-WBC，
以及 30/24/6 RTC。SIMPLE 固定结果表中 Ψ₀ 的 level 0/1/2 结果为
3/10、3/10、4/10；应分别报告三档结果，不能只用 10/30 的总平均掩盖 DR 差异。

验证结果
--------

增加 episode 数之前，先确认：

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 检查项
     - 预期结果
   * - 模型边界
     - System-2 参数保持冻结，policy 只运行 eval mode。
   * - RTC
     - execution acknowledgement 与 decision 严格匹配；上一 plan 的最后 6 步用于条件化下一 plan。
   * - Episode 边界
     - 初始 reset 设置 ``episode_id`` 并清除 pending/committed RTC state；成功后不会隐式开始下一 trial。
   * - Metrics
     - RLinf 报告 SIMPLE task 自身的 return、episode length 与 success。

.. warning::

   P3 不实现 RL sampling、log-probability、actor recompute 或跨 decision credit
   attribution。不要使用该配置进行训练。
