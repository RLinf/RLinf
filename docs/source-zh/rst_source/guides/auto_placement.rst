自动 Placement
==============

在启动大规模 RL 训练前，使用自动 placement 工具生成更优的 ``cluster.component_placement`` 方案。工具根据实测的各组件单步耗时，搜索共置、分离与流水线等多种 GPU 划分方式，并输出单步耗时最短的放置配置。

适用范围与前提
--------------

- ``examples/reasoning/run_placement_autotune.sh`` 面向 ``examples/reasoning/config/math/`` 下的 Megatron 数学推理 YAML 配置（默认为 ``qwen2.5-1.5b-grpo-megatron``）。
- 工具会查询当前 Ray 集群的节点与 GPU 数量来校验可用卡号，请在目标集群（或 ``cluster.num_nodes`` 与单机卡数一致的集群）上运行。
- 运行前需先在共置模式下测量 ``actor``、``inference`` 和 ``rollout`` 的单步耗时（见步骤 1）。

工作原理
--------

自动 placement 工具按任务构建组件依赖图（开启 ``algorithm.recompute_logprobs: True`` 时为 ``rollout -> inference -> actor``，关闭时为 ``rollout -> actor``），在满足模型并行度与 batch 整除约束的前提下估算不同 GPU 数量下的组件耗时，并在共置与分离划分间搜索总迭代耗时最小的方案。

步骤 1：收集 Profile 数据
-------------------------

在共置模式（``actor,rollout,reward: all``）下测量各组件的平均单步耗时（单位：秒）：

1. 用共置配置试跑 3–5 个迭代（跳过首步预热）。
2. 从训练日志、TensorBoard 或 :doc:`系统级 Profiling <profile>` 中读取每个迭代的阶段耗时：

   - ``actor_cost``：每个迭代中 actor 训练耗时（秒）。
   - ``inference_cost``：每个迭代中参考模型 / logprob 推理耗时（秒）。若 ``algorithm.recompute_logprobs`` 为 ``False``，配置读取器仍会访问该字段，保留 ``0.0`` 占位即可。
   - ``rollout_cost``：每个迭代中 rollout 生成耗时（秒）。

3. 在配置 YAML 顶层添加 ``profile_data``：

.. code-block:: yaml

   profile_data:
     actor_cost: 95.7      # actor 训练耗时（秒/迭代）
     inference_cost: 30.8  # logprob 推理耗时（秒/迭代）
     rollout_cost: 59.9    # rollout 生成耗时（秒/迭代）

步骤 2：运行自动 Placement
--------------------------

在仓库根目录执行脚本，传入 ``examples/reasoning/config/math/`` 下的配置名（不含 ``.yaml`` 后缀）：

.. code-block:: bash

   bash examples/reasoning/run_placement_autotune.sh qwen2.5-1.5b-grpo-megatron

若配置文件位于其他目录，可直接运行 ``toolkits/auto_placement/auto_placement_worker.py`` 并通过 ``--config-path`` 与 ``--config-name`` 指定路径。

预期输出如下：

.. code-block:: text

   ==================================================
   Best placement for this task is:
   rollout, actor : all

若分离式或混合放置更快，输出会给出各组件对应的 GPU 卡号区间（例如 ``rollout: 0-7``、``actor: 8-15``）。

步骤 3：应用结果
----------------

将输出的放置结果填入训练 YAML 的 ``cluster.component_placement`` 字段，再按常规方式启动训练。放置语法与混合布局说明见 :doc:`Placement 策略 <../concepts/placement>`。

故障排查
--------

- **缺少 ``profile_data`` 字段**：检查 YAML 是否同时包含 ``profile_data.actor_cost``、``profile_data.inference_cost`` 和 ``profile_data.rollout_cost``\ （未启用 logprob 重算时 ``inference_cost`` 填 ``0.0``）。
- **未找到有效放置方案**：确认当前 Ray 集群节点数与 ``cluster.num_nodes`` 一致，且 ``data.rollout_batch_size * algorithm.group_size / algorithm.n_minibatches`` 能被模型并行配置下至少一个数据并行度整除。
