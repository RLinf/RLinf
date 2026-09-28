Embodied Data 接口
==================

具身数据接口将运行时通信与 Actor 训练数据分开：

.. code-block:: text

   EnvOutput(obs, EnvTransition)
       -> PolicyInput (+ previous EnvPart)
       -> actions tensor

   PolicyPart + completed EnvPart
       -> TrajectoryCollector
       -> Trajectory / episode shard / pipeline micro-batch

完整生命周期、数据所有权和执行模式语义参见
:doc:`../../concepts/trajectory_collector`。

Shape 记号
----------

类字段注释使用以下符号：

* ``B`` 是 routed environment batch。Decoupled merge 前通常为
  ``env.train.total_num_envs / env_world_size /
  rollout.pipeline_stage_num``。
* ``C`` 是 ``actor.model.num_action_chunks``；only-eval 模式改用 Rollout model
  config。
* ``A`` 是 ``actor.model.action_dim``，``D = C * A``。令
  ``E = env.train.rollout_epoch``，完整 trajectory 通常包含
  ``T = E * env.train.max_steps_per_rollout_epoch / C`` 个 chunk。

环境与策略消息
--------------

``EnvOutput`` 组合 observation 和一个 ``EnvTransition``，不再重复 reward 和
boundary 字段。``PolicyInput`` 携带 observation，并可附带前一个 ``EnvPart``。
Rollout 只将 action tensor 返回 Env；完整 ``PolicyOutput`` 留在 trajectory 路径，
因为它还包含 log-probability、value、version 和 training input。

.. autoclass:: rlinf.data.schema.embodied_types.EnvOutput
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_types.EnvTransition
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_types.PolicyInput
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_types.PolicyOutput
   :members:
   :member-order: bysource

Trajectory Part
---------------

``PolicyPart`` 和 ``EnvPart`` 是 Actor channel 仅有的两种输入，并通过
``TrajectoryKey`` 配对。Channel routing 拆分 batch 时，``TrajectorySource``
保留 source size 和 offset。Env 在 step 后创建未完成的 ``EnvPart``；Rollout 补充
可选 terminal inference 数据后发布同一个类型。

.. autoclass:: rlinf.data.schema.embodied_types.TrajectoryKey
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_types.TrajectorySource
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_types.PolicyPart
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_types.EnvPart
   :members:
   :member-order: bysource

采集
----

``TrajectoryPlan`` 校验输出模式并推导 rollout geometry。所有具身模式都使用同一个
公共 ``TrajectoryCollector``。

.. autoclass:: rlinf.data.schema.embodied_trajectory.TrajectoryMode
   :members:

.. autoclass:: rlinf.data.schema.embodied_trajectory.TrajectoryPlan
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_trajectory.TrajectoryCollector
   :members:
   :member-order: bysource

Actor 输出
----------

``TrajectoryStep`` 解析一对已配对的 policy/environment 数据，包括 reward、
intervention 和 transition 语义。``Trajectory`` 再将 step 堆叠成
``[T, B, ...]`` Actor tensor。Boundary 和 final-value 序列可能包含
``T + E`` 个元素。

.. autoclass:: rlinf.data.schema.embodied_types.TrajectoryStep
   :members:
   :member-order: bysource

.. autoclass:: rlinf.data.schema.embodied_types.Trajectory
   :members:
   :member-order: bysource
