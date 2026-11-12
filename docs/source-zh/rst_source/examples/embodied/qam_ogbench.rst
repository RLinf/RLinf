在 OGBench 上离线训练 QAM
==============================

RLinf 提供原生 PyTorch QAM 实现，用于 OGBench 离线训练。示例任务为 ``antmaze-large-navigate-singletask-task1-v0``。离线更新使用 ``OfflineRunner``，评估复用标准 EnvWorker 和 RolloutWorker。在线微调应由在线 runner 承担，不属于本示例。

环境准备
--------

在 RLinf 仓库根目录执行：

.. code-block:: bash

   bash requirements/install.sh embodied --env ogbench
   source .venv/bin/activate
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl
   export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

训练与评估
----------

``ogbench_qam_offline.yaml`` 默认运行一个 seed、100 万次梯度更新、batch size 256，并每 5 万次更新评估 50 个 episode。通过 ``cluster.component_placement`` 选择空闲 GPU。各 seed 使用独立进程和实验名：

.. code-block:: bash

   for seed in 10001 20002 30003; do
     python examples/embodiment/train_offline_rl.py \
       --config-name ogbench_qam_offline \
       actor.seed="$seed" \
       runner.logger.experiment_name="qam-antmaze-offline-$seed"
   done

以上命令顺序运行三个 seed。并行运行时，需要为三个进程分别指定不同的 ``cluster.component_placement.actor``、``env`` 和 ``rollout`` GPU 编号。Checkpoint 位于 ``logs/<experiment>/checkpoints/global_step_<step>/actor/``；训练指标位于 ``logs/<experiment>/seed_<seed>/train.csv``；评估指标由 RLinf 的 metric logger 保存。

继续训练时指定 checkpoint 目录：

.. code-block:: bash

   python examples/embodiment/train_offline_rl.py \
     --config-name ogbench_qam_offline \
     actor.seed=10001 \
     runner.resume_dir=logs/qam-antmaze-offline-10001/checkpoints/global_step_250000 \
     runner.logger.experiment_name=qam-antmaze-offline-10001-resumed

只评估 checkpoint 时增加 ``runner.only_eval=true``。Checkpoint 保存策略、优化器、数据采样器和随机数生成器状态，可以准确继续离线训练。

验证
----

单元测试覆盖数据采样、QAM 更新和 checkpoint。可选的对照测试使用官方 JAX QAM commit ``2726d767c9a0a7a46d49693f0391f73dc2cf58ac``，比较损失、梯度和一次参数更新。正式训练只调用 RLinf 的 PyTorch 实现。

.. code-block:: bash

   python -m pytest tests/unit_tests/test_qam.py
   QAM_REFERENCE_ROOT=/path/to/pinned/QAM \
     JAX_PLATFORMS=cpu python -m pytest tests/parity_tests/test_qam_reference.py
