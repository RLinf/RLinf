Turtle2 DAgger 接管式采集
=========================

这个示例把 Turtle2 作为 master takeover 的 raw 数据采集工具使用。它不是在线
DAgger 训练配置。policy 和 master 都使用同一套 14D absolute pose + gripper
动作语义，机器人最终通过 Turtle2 direct pose backend 执行。

控制链路
--------

运行时链路是：

.. code-block:: text

   OpenPI policy action chunk
      -> DualAbsolutePoseActionWrapper
      -> MasterTakeoverIntervention
      -> Turtle2 direct pose backend
      -> CollectEpisode pickle export

takeover mode 未激活时，wrapper 直接放行 policy action。takeover mode 激活且
master pose 足够新时，wrapper 用 master pose 覆盖 policy action。被 backend
拒绝或判定非法的 takeover pose 只作为 metadata 记录，不会写成 expert label。

Takeover 协议
-------------

env 侧 TCP server 保留 X2Robot takeover 握手语义：

- ``MSG_MODE`` 传递 normal/takeover 模式。
- ``MSG_JOINT`` 发送 slave 当前 joint snapshot，用于 master 对齐。
- ``MSG_POSE`` 携带 master 的 14D absolute pose 指令。

这里的 joint snapshot 只用于同步，不表示 joint command takeover。本示例不包含
joint mirror；如果后续需要 joint takeover，需要单独定义 joint action 的 raw
schema、label 合同和测试。

配置
----

使用：

.. code-block:: bash

   bash examples/embodiment/run_realworld_eval.sh realworld_turtle2_dagger_takeover_collect_openpi

示例配置打开：

- ``env.eval.use_master_takeover: true``
- ``env.eval.action_mode: absolute_pose``
- ``env.eval.override_cfg.pose_control_backend: direct``
- ``env.eval.data_collection.export_format: pickle``
- ``rollout.collect_transitions: false``

pickle episode 记录实际执行动作。direct control 拒绝动作时，raw action 不会被当成
执行动作写入；episode 的 ``actions`` 会记录保持住的 ``executed_action``，被拒绝
的候选动作只保留在 metadata 中。
