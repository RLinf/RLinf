Env Decoupled Mode
==================

``env_decoupled_mode`` 是 RLinf embodied 任务中用于解耦 Env Worker 与 Rollout Worker
通信的一种模式。它通过配置 ``runner.enable_decoupled_mode: true`` 开启。

开启后，Env Worker 不再与固定的 Rollout Worker rank 一一绑定。Env Worker 会将观测
数据放入共享 Channel，空闲的 Rollout Worker 可以动态获取 batch 进行推理，并在完成后
将结果返回给原始 Env Worker。

该模式适用于 Env Worker 和 Rollout Worker 处理速度不一致的场景，尤其是仿真环境耗时波动较大、
部分 Env Worker 容易阻塞，或希望 Rollout Worker 动态聚合多个 Env 请求进行批量推理时。

如何开启
--------

在 embodied 配置中设置：

.. code-block:: yaml

   runner:
     enable_decoupled_mode: true

   rollout:
     rollout_queue_size: 0

其中：

- ``runner.enable_decoupled_mode: true`` 表示启用 Env Decoupled Mode。
- 不配置 ``runner.enable_decoupled_mode`` 时，使用普通通信模式。
- ``rollout_queue_size`` 控制 Rollout Worker 单次最多聚合多少组 Env 数据。
  设置为 ``0`` 时使用默认策略，此时 Rollout Worker 单次聚合的 Env 数据数量为
  ``ceil(env_world_size // rollout_world_size)``。

示例配置可参考：

.. code-block:: text

   examples/embodiment/config/maniskill_sac_mlp_async_decoupled.yaml

适用条件
--------

当前实现要求 Env Worker 和 Rollout Worker 数量满足一定比例关系，可以是任意比例，
例如 ``env:rollout = 8:3``，但需要保证 Env Worker 数量不小于 Rollout Worker 数量。

当 Env Worker 明显多于 Rollout Worker 时，decoupled 模式可以让 Rollout Worker
持续从共享 Channel 中获取任务，避免绑定到某个固定 Env rank。

这种配置适合 Env Worker 较多、Rollout Worker 相对较少的情况。需要注意的是，
如果 Rollout Worker 成为瓶颈，继续增加 Env Worker 不一定能提升吞吐，反而可能增加
Channel 排队时间。

适合用于：

- 仿真环境数量较多。
- 单个 rollout 推理可以处理较大的 batch。
- 希望用较少 Rollout Worker 服务较多 Env Worker。

可以通过 ``rollout_queue_size`` 控制 Rollout Worker 单次聚合的 Env shard 数量：

.. code-block:: yaml

   runner:
     enable_decoupled_mode: true

   rollout:
     rollout_queue_size: 2

较小的 ``rollout_queue_size`` 通常降低等待时间；较大的值可能提高推理 batch 利用率，
但也可能增加单次聚合等待。

分组路由绑定
------------

默认情况下 decoupled 模式使用单一全局池，任意 Rollout Worker 都可获取任意 Env batch。
设置 ``rollout.enable_group_route_binding: true``（默认 ``false``）会将全局池划分为固定的
分组绑定：Env rank ``e`` 映射到 Rollout rank
``floor(e * rollout_world_size / env_world_size)``。该比例映射支持非整除比例，任意两组的
Env Worker 数量最多相差 1；只要 ``env_world_size >= rollout_world_size``，每个 Rollout
Worker 都至少分到一个 Env Worker。

训练和评估中的 Env→Rollout 请求与 Rollout→Env 返回都使用同一个 ``route_key``。
这个 key 是共享 ChannelWorker 内的逻辑队列隔离，不会额外启动 Channel 进程或创建独立
传输通道；不同 key 之间不会从全局池抢任务。

.. code-block:: yaml

   runner:
     enable_decoupled_mode: true

   rollout:
     enable_group_route_binding: true

由此得到固定的 ``1`` Rollout : ``N`` Env-Worker 重叠，并可作为多个独立分组在机器人集群上复制，
适用于每个 Rollout Worker 驱动一组固定机器人的真机 rollout 场景。只要
``env_world_size >= rollout_world_size``，Env 与 Rollout 数量无需整除。若要求每台机器人
逐条服务、推理 batch size 固定为 1，请设置 ``rollout.rollout_queue_size: 1``，避免聚合请求。

Turtle2 四流 Bringup
~~~~~~~~~~~~~~~~~~~~

``realworld_dummy_turtle2_async_ppo_openpi_pi05_2gpu_4stream`` 是一个短小的单节点系统样例：
actor/train 固定到 GPU 0，rollout 固定到 GPU 1，四个 CPU-only Env rank 共用节点 0。
由于只有一个 Rollout rank，四个 Env rank 都映射到 ``grp0``，并通过
``rollout_queue_size: 1`` 逐条服务。样例复用 ``RLinf-Pi05-LIBERO-SFT`` 及其
normalization stats，仅用于 dummy bringup 与 profile；dummy 环境恒为零的奖励不能证明策略学到了任务。

参考 RTX PRO 5000 主机的双卡 collective 需要 ``NCCL_P2P_DISABLE=1`` 才能可靠运行。
样例已将它配置在 GPU node group 上，并在每个 global step 后使用 bucket 权重同步；迁移到
其他机器时，应先验证 direct P2P，再决定是否删除该环境变量。

在仓库根目录运行五步 smoke；若 checkpoint 没有挂载到 YAML 中的默认路径，同时覆盖 actor 和
rollout 的模型路径：

.. code-block:: bash

   export EMBODIED_PATH="$PWD/examples/embodiment"
   export MODEL_PATH=/absolute/path/to/RLinf-Pi05-LIBERO-SFT
   python examples/embodiment/train_async.py \
     --config-path examples/embodiment/config \
     --config-name realworld_dummy_turtle2_async_ppo_openpi_pi05_2gpu_4stream \
     actor.model.model_path="$MODEL_PATH" \
     rollout.model.model_path="$MODEL_PATH"

启动时会打印 resolved config；TensorBoard 与逐 worker 日志位于
``runner.logger.log_path`` 下。若要在 warm-up 后用 Nsight Systems 采集一个完整 step，开启
profile 配置：

.. code-block:: bash

   python examples/embodiment/train_async.py \
     --config-path examples/embodiment/config \
     --config-name realworld_dummy_turtle2_async_ppo_openpi_pi05_2gpu_4stream \
     actor.model.model_path="$MODEL_PATH" \
     rollout.model.model_path="$MODEL_PATH" \
     cluster.profiling.enabled=true \
     'cluster.profiling.steps=[3]'

RLinf 只为 ``ActorGroup`` 和 ``RolloutGroup`` 注入 ``nsys profile``；四个 Env worker 不被
nsys 包装，其逐 rank chunk 时间来自 RLinf metrics。报告写入
``<log_path>/<experiment_name>/profiling``。可导出两类标准汇总：

.. code-block:: bash

   nsys stats --report nvtx_sum --format csv --timeunit ms worker.nsys-rep
   nsys stats --report cuda_gpu_kern_sum --format csv --timeunit ms worker.nsys-rep

训练流程
--------

开启 decoupled 模式后，训练阶段大致流程如下：

1. Env Worker 执行环境 step，得到 observation。
2. Env Worker 将 observation 发送到 Rollout Channel。
3. 任意 Rollout Worker 从 Channel 中动态获取一个或多个 Env batch。
4. Rollout Worker 执行模型推理，生成 action 或 rollout result，并将结果返回给发送该请求的 Env Worker。
5. Env Worker 根据返回结果继续进行环境交互。

用户通常不需要直接处理路由细节。只要在配置中开启 ``runner.enable_decoupled_mode``，
并使用支持该模式的 Env Worker、Rollout Worker 和 Runner 即可。

评估流程
--------

评估阶段也可以使用 decoupled 模式。此时 Env Worker 会持续发送 eval observation，
Rollout Worker 执行 eval 推理并返回 action。

与训练阶段相比，评估通常不需要收集完整的训练用 rollout 信息，但通信方式相同：
Env Worker 发送请求，Rollout Worker 动态接收并返回结果。

何时使用
--------

建议在以下情况下启用 ``env_decoupled_mode``：

- Env Worker 的 step 时间波动较大。
- 部分环境可能出现长尾延迟或临时阻塞。
- Env Worker 数量大于 Rollout Worker 数量。
- 希望 Rollout Worker 动态聚合多个 Env 请求进行批量推理。
- 异步 embodied 训练中，普通固定 rank 通信容易造成等待。

如果 Env 和 Rollout 的速度稳定、数量相同，并且没有明显阻塞，普通模式通常更简单。

注意事项
--------

- 当前
