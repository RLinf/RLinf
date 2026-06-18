配置参考
========

评测配置为 Hydra YAML，默认放在 ``evaluations/<benchmark>/`` 下。以 ``libero_spatial_openpi_pi05_eval.yaml`` 为例，核心结构如下：

.. code-block:: yaml

   defaults:
     - env/libero_spatial@env.eval      # 环境 preset
     - model/pi0_5@rollout.model        # 模型 preset
     - override hydra/job_logging: stdout

   hydra:
     searchpath:
       - file://${oc.env:EMBODIED_PATH}/config/

   runner:
     task_type: embodied_eval   # 必须为 embodied_eval
     only_eval: True            # 仅评测，不训练
     ckpt_path: null            # 可选：加载 .pt 权重
     logger:
       log_path: "../results"

   cluster:
     component_placement:
       env,rollout: all          # env 与 rollout 的 GPU 分配

   env:
     eval:
       total_num_envs: 500       # 并行环境数
       rollout_epoch: 1          # 评测轮次
       max_episode_steps: 240
       auto_reset: True
       is_eval: True
       video_cfg:
         save_video: True

   rollout:
     generation_backend: "huggingface"
     model:
       model_path: "/path/to/model"   # 必填：模型权重路径
       model_type: "openpi"

.. _env-eval-fields:

env.eval 字段说明
-----------------

以下字段位于 ``env.eval`` 下，适用于具身评测的并行规模、轨迹长度与测试集覆盖控制。

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 字段
     - 作用与设置建议
   * - ``total_num_envs``
     - 全局并行环境总数，均匀分布在所有 env worker 上。越大吞吐越高，但显存/内存占用也越高。资源充足时可设为测试集总 init state 数；资源受限时配合 ``auto_reset`` 使用较小值。
   * - ``rollout_epoch``
     - 评测轮次。每轮在相同随机种子下遍历测试集；多轮结果取平均，用于降低方差。完整覆盖测试集通常设为 ``1``；需要更稳定指标时可设为 ``2`` 或更高。
   * - ``max_episode_steps``
     - 单条轨迹的最大交互步数，达到后强制截断（truncation）。应满足 benchmark 要求的最小步数，并与模型训练配置一致。
   * - ``max_steps_per_rollout_epoch``
     - 每个并行环境在一轮 ``rollout_epoch`` 内的总交互步数上限。实际 chunk 步数为 ``max_steps_per_rollout_epoch / rollout.model.num_action_chunks``，**必须能被** ``num_action_chunks`` **整除**。无 ``auto_reset`` 时通常等于 ``max_episode_steps``；有 ``auto_reset`` 时设为 ``max_episode_steps`` 的整数倍以在同一轮内串行多条轨迹。
   * - ``auto_reset``
     - 是否在 episode 结束（成功或截断）后自动 reset 并加载下一个 init state。``True`` 时可在较少并行环境上覆盖完整测试集；``False`` 时每个环境每轮只产生一条轨迹，需在 ``rollout_epoch`` 之间由 ``finish_rollout`` 切换 init state。
   * - ``ignore_terminations``
     - 是否忽略任务成功带来的提前终止。``True`` 时 episode 仅在达到 ``max_episode_steps`` 时结束，成功信号记入 ``success_once`` / ``success_at_end`` 但不提前 reset，保证各轨迹长度一致，便于并行评测。评测示例通常建议 ``True``。
   * - ``use_fixed_reset_state_ids``
     - 是否使用预分配的 reset state ID，而非每轮随机采样。评测时应设为 ``True``，确保每条轨迹对应确定的初始条件。
   * - ``use_ordered_reset_state_ids``
     - 是否按固定顺序遍历 init state。``is_eval=True`` 时部分环境（如 LIBERO）内部已强制有序遍历；显式设为 ``True`` 可在非 eval 场景复用相同顺序。``auto_reset`` 触发 reset 时，会按序取下一个 state ID。
   * - ``is_eval``
     - 评测模式开关，**必须** 为 ``True``。启用后使用有序 init state 列表，并在 ``auto_reset`` 时按序分配下一条轨迹的 reset state。

必须修改的字段
--------------

1. ``rollout.model.model_path``：指向本地模型目录或 HuggingFace 缓存路径。
2. ``env.eval`` 中与资源相关的字段：``total_num_envs``、``max_episode_steps``、``assets_path`` （RoboTwin）等。
3. ``cluster.component_placement``：按可用 GPU 数量调整 ``env`` 与 ``rollout`` 的 placement。
4. **真机评测：** 在 ``cluster.node_groups`` 中配置 Franka IP 与节点拓扑（参考 ``realworld/realworld_eval.yaml``）。

从训练配置派生
--------------

可复制 ``examples/embodiment/config/`` 或 ``tests/e2e_tests/embodied/`` 中对应训练 YAML，删除 ``algorithm``、``actor`` 等训练段，保留 ``env.eval`` 与 ``rollout``，并设置：

- ``runner.task_type: embodied_eval``
- ``runner.only_eval: True``

配置回退
--------

若 ``evaluations/<benchmark>/<config>.yaml`` 不存在，``run_eval.sh`` 会自动回退到 ``examples/embodiment/config/`` 下同名配置，便于复用训练配置做评测。详见 :doc:`cli`。
