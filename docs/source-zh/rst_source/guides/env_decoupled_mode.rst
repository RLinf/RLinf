Env Decoupled Mode
==================

``env_decoupled_mode`` 是 RLinf embodied 任务中用于解耦 Env Worker 与 Rollout Worker
通信的一种模式。它通过配置 ``runner.enable_decoupled_mode: true`` 开启。

默认情况下，Env Worker 在此模式下不与固定的 Rollout Worker rank 绑定。Env Worker 将观测数据放入共享 Channel，空闲的 Rollout Worker 动态获取 batch 进行推理，完成后将结果返回给原始 Env Worker。

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
  ``env_world_size // rollout_world_size``。

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

如果每个 Rollout Worker 需要服务一组固定的 Env Worker，例如驱动固定的一组机器人，可在 decoupled 模式下开启 ``rollout.enable_group_route_binding``。该选项默认为 ``false``，使用前文介绍的共享池。

.. code-block:: yaml

   runner:
     enable_decoupled_mode: true

   rollout:
     enable_group_route_binding: true

Env Worker 的分组编号为 ``env_rank * rollout_world_size // env_world_size``，Rollout rank ``k`` 服务编号为 ``k`` 的分组。训练和评估都使用各组独立的 ``route_key``，请求在所属组内处理，结果返回发起请求的 Env Worker。

Worker 数量须满足 ``env_world_size >= rollout_world_size``，但无需整除。例如，4 个 Env Worker 和 3 个 Rollout Worker 会分为包含 Env rank ``[0, 1]``、``[2]``、``[3]`` 的三组。每个 Rollout Worker 都有非空分组，各组大小最多相差 1。

训练流程
--------

开启 decoupled 模式后，训练阶段大致流程如下：

1. Env Worker 执行环境 step，得到 observation。
2. Env Worker 将 observation 发送到 Rollout Channel。
3. Rollout Worker 从共享池获取一个或多个 Env batch；开启分组绑定时，从所属分组获取。
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
