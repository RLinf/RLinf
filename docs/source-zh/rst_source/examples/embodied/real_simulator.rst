真机仿真器
==========

在仿真训练中加入真实的观测延迟和网络条件，无需部署到实际硬件。

RLinf 提供两个独立模块——``delay_sampler`` 和 ``net_emulation``——各自由
YAML 中的对应参数控制。两者**默认关闭**：不配置对应参数时，训练流程不受影响。


观测延迟仿真 (delay_sampler)
----------------------------

在每条观测数据离开 Env Worker 之前插入可配置的延迟等待，模拟每个真机的
传感器延迟。

适用场景：测试可变观测时序对策略训练的影响，或为真实世界部署准备策略时，
模拟不同机器人相机和编码器延迟。

支持的分布类型：

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - 类型
     - 参数
     - 适用场景
   * - ``constant``
     - ``delay``
     - 所有观测使用相同的固定延迟
   * - ``uniform``
     - ``min_delay``, ``max_delay``
     - 延迟在已知范围内均匀变化
   * - ``exponential``
     - ``rate``
     - 延迟遵循类似泊松到达的模式
   * - ``gaussian``
     - ``mean``, ``stddev``
     - 符合真实传感器的抖动，具有典型值和波动范围

所有延迟值单位为**秒**。

.. code-block:: yaml

   env:
     delay_sampler:
       type: uniform
       min_delay: 0.11       # 110 ms
       max_delay: 0.20       # 200 ms

使用要求：

- ``runner.enable_decoupled_mode`` 必须设为 ``true``。
- 延迟仅在训练模式（``mode="train"``）下生效，评估不受影响。
- 需使用异步 Runner（``train_async.py``）。同步 Runner 不支持解耦通信。


网络仿真 (net_emulation)
------------------------

通过全局令牌桶调度器仿真跨 Worker 的网络延迟和带宽限制。一个
``NetEmulationProxy`` Ray actor 拦截所有经过 NCCL 的 Worker 间通信，
强制执行配置的延迟和带宽上限。

适用场景：测试在带宽受限或高延迟链路下策略的表现，例如 Env Worker 和
Rollout Worker 部署在不同集群或云区域时。

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - 键
     - 类型
     - 说明
   * - ``enabled``
     - ``bool``
     - 开启（``true``）或关闭（``false``）。默认：``false``。
   * - ``symmetric``
     - ``bool``
     - 为 ``true`` 时，每对 cross-DC 自动镜像。默认：``true``。
   * - ``proxy.node_rank``
     - ``int``
     - 运行 Proxy actor 的节点 rank。默认：``0``。
   * - ``proxy.num_cpus``
     - ``int``
     - 为 Proxy 分配的 CPU 数。默认：``1``。
   * - ``crossdc_pairs``
     - ``list``
     - 源-目标端点对，每对可配置 ``delay_ms``。
   * - ``bandwidth_groups``
     - ``list``
     - 共享 ``bandwidth_mbps`` 带宽预算的端点组。

.. code-block:: yaml

   net_emulation:
     enabled: true
     symmetric: true
     proxy:
       node_rank: 0
       num_cpus: 1
     crossdc_pairs:
       - src: ["Env:0", "Env:1"]
         dst: ["Rollout:0", "Rollout:1", "Actor:0", "Actor:1"]
         delay_ms: 50
     bandwidth_groups:
       - members: ["Env:0", "Env:1"]
         bandwidth_mbps: 1000
       - members: ["Rollout:0", "Rollout:1", "Actor:0", "Actor:1"]
         bandwidth_mbps: 500

端点名称使用 ``GroupName:Rank`` 格式——``Env:0`` 表示第一个 Env Worker，
``Rollout:1`` 表示第二个 Rollout Worker。``Group`` 后缀会自动去除。

使用要求：

- 同步和异步 Runner 均可使用。
- 与 ``delay_sampler`` 独立。可单独启用、组合启用或全部关闭。


组合使用
--------

同时启用两个模块以模拟真实部署环境：

.. code-block:: yaml

   runner:
     enable_decoupled_mode: true       # delay_sampler 必需

   env:
     delay_sampler:
       type: uniform
       min_delay: 0.11
       max_delay: 0.20

   net_emulation:
     enabled: true
     crossdc_pairs:
       - src: ["Env:0", "Env:1"]
         dst: ["Rollout:0", "Rollout:1"]
         delay_ms: 50


示例
----

完整示例配置：

.. code-block:: text

   examples/embodiment/config/realsimulator_robotwin_adjust_bottle_dagger_openpi.yaml


参考：
  
  - :doc:`RoboTwin <robotwin>` — RoboTwin 环境设置与配置。
  - :doc:`DAgger <dagger>` — 基于专家策略的 DAgger 训练。
  - :doc:`Env 解耦模式 <../../guides/env_decoupled_mode>` — ``delay_sampler`` 的前置条件。
