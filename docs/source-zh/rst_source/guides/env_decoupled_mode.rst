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
分组绑定：Rollout Worker rank ``k`` 只服务自己的 Env 分组（env rank
``k * ratio .. k * ratio + ratio - 1``，其中 ``ratio = env_world_size //
rollout_world_size``），每组通过独立的 ``route_key`` 隔离，不存在全局池，也没有跨组抢任务。

.. code-block:: yaml

   runner:
     enable_decoupled_mode: true

   rollout:
     enable_group_route_binding: true

由此得到固定的 ``1`` Rollout : ``N`` Env-Worker 重叠，并可作为多个独立分组在机器人集群上复制，
适用于每个 Rollout Worker 驱动一组固定机器人的真机 rollout 场景。它要求 ``env_world_size``
能被 ``rollout_world_size`` 整除（启动时校验）；普通 decoupled 模式则允许任意比例。

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