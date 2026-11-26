真机仿真器
==========

在仿真训练中加入真实的观测延迟和网络条件，无需部署到真机。RLinf 提供两个独立的 YAML 配置模块：``delay_sampler`` 和 ``net_emulation``，两者默认关闭。

概览
----

在仿真训练中注入传感器延迟和跨 worker 网络约束。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀ · 任意具身 policy

   .. grid-item-card:: 算法
      :text-align: center

      DAgger · 同步与异步 RL

   .. grid-item-card:: 任务
      :text-align: center

      RoboTwin · 任意仿真环境

   .. grid-item-card:: 硬件
      :text-align: center

      单节点或多节点 GPU

| **你将完成：** 在 YAML 中开启 ``delay_sampler`` 或 ``net_emulation`` → 运行 ``run_async.sh``\ （或 ``run_embodiment.sh``）→ 验证 policy 鲁棒性。
| **前置条件：** :doc:`RoboTwin <robotwin>`\ （用于内置示例）· :doc:`DAgger <dagger>`。

观测延迟仿真
------------

在每次环境步进和重置返回前注入可配置的等待延迟，模拟每个环境的传感器延迟。适用于测试可变观测时序对训练的影响，或在真机部署前模拟不同机器人的相机和编码器延迟。

支持的分布类型（所有延迟值单位均为 **秒**）：

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - 类型
     - 参数
     - 适用场景
   * - ``constant``
     - ``delay``
     - 所有观测使用相同的固定延迟。
   * - ``uniform``
     - ``min_delay``, ``max_delay``
     - 延迟在已知范围内均匀变化。
   * - ``exponential``
     - ``rate``
     - 延迟遵循类似泊松到达的模式。
   * - ``gaussian``
     - ``mean``, ``stddev``
     - 符合真实传感器的抖动，具有典型值和波动范围。

.. code-block:: yaml

   env:
     delay_sampler:
       type: uniform
       min_delay: 0.11       # 110 ms
       max_delay: 0.20       # 200 ms

- 仅在训练模式下生效，评测不受影响。
- 按环境实例独立生效，同步和异步 runner 均可使用。

网络仿真
--------

在集群范围内仿真跨 worker 的网络延迟和带宽限制。将 ``cluster.net_emulation.enabled`` 设为 ``true`` 后，所配置 worker 组之间的点对点发送和广播（例如 actor 向 rollout worker 同步权重）都会按链路延迟和组带宽预算预约传输时隙并等待对应延迟。广播在发送端计费一次、在每个接收带宽组各计费一次，发送端按最慢的接收端等待。

同设备与同节点的数据交换走共享内存 IPC，不会被仿真。适用于测试带宽受限或高延迟链路下的训练表现，例如 env worker 和 rollout worker 部署在不同集群或云区域时。

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
   * - ``crossdc_pairs``
     - ``list``
     - 源-目标端点对，每对可配置 ``delay_ms``。
   * - ``bandwidth_groups``
     - ``list``
     - 共享 ``bandwidth_mbps`` 带宽预算的端点组。

端点名称使用 ``GroupName:Rank`` 格式，例如 ``Env:0`` 表示第一个 env worker，``Rollout:1`` 表示第二个 rollout worker。支持闭区间 rank 范围，如 ``Env:0-3`` 会展开为 ``Env:0`` 到 ``Env:3``，末尾的 ``Group`` 后缀会自动去除。

- 同步和异步 runner 均可使用。
- 与 ``delay_sampler`` 独立，可单独启用、组合启用或全部关闭。

安装与运行
----------

按 :doc:`RoboTwin <robotwin>` 准备好环境与 checkpoint 后，启动内置示例配置 ``examples/embodiment/config/realsimulator_robotwin_adjust_bottle_dagger_openpi.yaml``：

.. code-block:: bash

   bash examples/embodiment/run_async.sh realsimulator_robotwin_adjust_bottle_dagger_openpi

.. dropdown:: 进阶配置（可选）

   包含 ``crossdc_pairs`` 与 ``bandwidth_groups`` 的完整 ``net_emulation`` 配置示例：

   .. code-block:: yaml

      cluster:
        net_emulation:
          enabled: true
          symmetric: true
          crossdc_pairs:
            - src: ["Env:0-1"]
              dst: ["Rollout:0-1", "Actor:0-1"]
              delay_ms: 50
          bandwidth_groups:
            - members: ["Env:0-1"]
              bandwidth_mbps: 1000
            - members: ["Rollout:0-1", "Actor:0-1"]
              bandwidth_mbps: 500

   同时启用 ``delay_sampler`` 与 ``net_emulation`` 模拟真实部署环境：

   .. code-block:: yaml

      env:
        delay_sampler:
          type: uniform
          min_delay: 0.11
          max_delay: 0.20

      cluster:
        net_emulation:
          enabled: true
          crossdc_pairs:
            - src: ["Env:0", "Env:1"]
              dst: ["Rollout:0", "Rollout:1"]
              delay_ms: 50

相关文档：:doc:`RoboTwin <robotwin>` 与 :doc:`DAgger <dagger>`。
