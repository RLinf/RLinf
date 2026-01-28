Embodied Data 接口
====================

本节介绍具身场景下 rollout 与训练过程中使用的核心数据结构：
`EnvOutput`、`ChunkStepResult`、`EmbodiedRolloutResult` 和 `Trajectory`。
它们共同完成从环境输出、chunk step 级结果累积，到轨迹化与批量训练输入的闭环。

整体关系
---------

- `EnvOutput`：环境每个 chunk step 的原始输出（obs、reward、done 等）。
- `ChunkStepResult`：模型推理结果与奖励信号在 chunk step 维度上的封装。
- `EmbodiedRolloutResult`：将多个 chunk step 的结果与 transitions 逐步积累。
- `Trajectory`：将累积结果整理为轨迹张量（形状通常为 `[T, B, ...]`）。

其中 `EmbodiedRolloutResult.to_splited_trajectories()` 可将轨迹按 batch 维度切分，
用于通过 Channel 分发给多个 Actor/Trainer。

EnvOutput
----------

`EnvOutput` 描述环境侧的输出，包含 observation 与 episode 结束信号。
在初始化时，张量会被移动到 CPU 并整理为连续内存。

.. autoclass:: rlinf.data.embodied_io_struct.EnvOutput
   :members:
   :member-order: bysource

ChunkStepResult
----------------

`ChunkStepResult` 描述单步推理的结果与训练所需的附加信息，
包含动作、对数概率、价值估计与额外的 forward inputs。
初始化时会将张量统一移动到 CPU。

.. autoclass:: rlinf.data.embodied_io_struct.ChunkStepResult
   :members:
   :member-order: bysource

EmbodiedRolloutResult
-----------------------

`EmbodiedRolloutResult` 负责在 rollout 期间逐步积累 chunk step 级结果与 transitions，
并提供转换为 `Trajectory` 的方法：

- `append_step_result()`：追加 chunk step 级结果
- `append_transitions()`：追加 curr/next transition 观测
- `to_trajectory()`：拼接为轨迹张量
- `to_splited_trajectories()`：按 batch 维度切分轨迹

.. autoclass:: rlinf.data.embodied_io_struct.EmbodiedRolloutResult
   :members:
   :member-order: bysource

Trajectory
------------

`Trajectory` 是最终进入训练流程的轨迹表示，包含动作、奖励、终止标记、
观测与模型前向输入等字段。其张量维度一般为 `[T, B, ...]`，
其中 **T 表示 chunk step 数**， **B 表示并行环境数** （batch 维度）。

.. autoclass:: rlinf.data.embodied_io_struct.Trajectory
   :members:
   :member-order: bysource
