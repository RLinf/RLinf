配置参考
========

评测配置为 ``evaluations/<benchmark>/`` 下的 Hydra YAML 文件。每份评测配置的核心字段如下：

.. code-block:: yaml

   defaults:
     - env/libero_spatial@env.eval
     - model/pi0_5@rollout.model
   runner:
     task_type: embodied_eval
     only_eval: True
   cluster:
     component_placement:
       env,rollout: all
   env:
     eval:
       total_num_envs: 500
   rollout:
     model:
       model_path: "./RLinf-Pi05-LIBERO-SFT"

.. dropdown:: 完整示例 YAML（libero_spatial_openpi_pi05_eval.yaml）

   .. code-block:: yaml

      defaults:
        - env/libero_spatial@env.eval
        - model/pi0_5@rollout.model
        - override hydra/job_logging: stdout

      hydra:
        run:
          dir: .
        output_subdir: null
        searchpath:
          - file://${oc.env:EMBODIED_PATH}/config/

      cluster:
        num_nodes: 1
        component_placement:
          env,rollout: all

      runner:
        task_type: embodied_eval
        logger:
          log_path: "../results"
          project_name: rlinf
          experiment_name: "libero_spatial_openpi_pi05_eval"
          logger_backends: ["tensorboard"]
        max_epochs: 1
        max_steps: -1
        only_eval: True
        val_check_interval: -1
        save_interval: -1
        resume_dir: null
        ckpt_path: null

      env:
        group_name: "EnvGroup"
        eval:
          rollout_epoch: 1
          total_num_envs: 500
          auto_reset: True
          ignore_terminations: True
          max_episode_steps: 240
          max_steps_per_rollout_epoch: 240
          group_size: 1
          use_fixed_reset_state_ids: True
          is_eval: True
          video_cfg:
            save_video: True
            video_base_dir: ${runner.logger.log_path}/video/eval

      rollout:
        group_name: "RolloutGroup"
        rollout_backend: "huggingface"
        recompute_logprobs: False
        unnorm_key: libero_10
        enable_offload: False
        pipeline_stage_num: 1
        model:
          model_path: "RLinf/RLinf-Pi05-LIBERO-SFT"
          model_type: "openpi"
          num_action_chunks: 5
          num_steps: 3
          openpi:
            task: eval
            noise_level: 0.3

.. _env-eval-fields:

env.eval 字段说明
-----------------

``env.eval`` 下的字段控制具身评测的并行规模、轨迹长度与测试集覆盖方式：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 字段
     - 作用与设置建议
   * - ``total_num_envs``
     - 全局并行环境总数，均匀分配到各个 env worker。值越大吞吐越高，但显存与内存占用同步增加（``libero_spatial`` 每个环境约占 2 GiB 主机内存）。资源充足时可设为测试集总 init state 数，内存受限时调小并配合 ``auto_reset: True``。
   * - ``rollout_epoch``
     - 评测轮次。每轮在相同随机种子下遍历测试集并对结果取平均；单遍覆盖设 ``1``，需要更低方差时设 ``2`` 或更高。
   * - ``max_episode_steps``
     - 单条轨迹最大交互步数，达到后强制截断。须满足 benchmark 要求的步数上限并与训练配置一致。
   * - ``max_steps_per_rollout_epoch``
     - 每个并行环境在单轮 ``rollout_epoch`` 内的总步数预算，**必须能被** ``rollout.model.num_action_chunks`` **整除**。关闭 ``auto_reset`` 时通常等于 ``max_episode_steps``；开启 ``auto_reset`` 时设为 ``max_episode_steps`` 的整数倍，以在单轮内串行跑完多条轨迹。
   * - ``auto_reset``
     - episode 结束后是否自动重置并加载下一个 init state。``True`` 可用较少并行环境覆盖完整测试集；``False`` 时每个环境每轮只运行一条轨迹。
   * - ``ignore_terminations``
     - 是否忽略任务成功触发的提前终止。设为 ``True`` 时轨迹统一跑到 ``max_episode_steps``，同时记录 ``success_once`` 与 ``success_at_end``，保持并行轨迹长度一致；除非 benchmark 要求提前结束（如 SIMPLE），建议设为 ``True``。
   * - ``use_fixed_reset_state_ids``
     - 使用预分配的 reset state ID 而非随机采样。评测时设为 ``True`` 以保证初始条件确定。
   * - ``use_ordered_reset_state_ids``
     - 按固定顺序遍历 init state。部分环境（如 LIBERO）在 ``is_eval: True`` 时内部已强制有序遍历；显式设为 ``True`` 可确保 ``auto_reset`` 同样按序切换状态。
   * - ``is_eval``
     - 评测模式开关，**必须** 设为 ``True`` 以启用有序 init state 分配。

必须修改的字段
--------------

1. ``rollout.model.model_path``：本地模型目录或 Hugging Face ``owner/repo``\ （首次运行自动下载到 Hugging Face 缓存）。
2. ``env.eval`` 资源配置：按硬件调整 ``total_num_envs``、``max_episode_steps`` 以及 benchmark 资产路径（例如 RoboTwin 的 ``assets_path``）。
3. ``cluster.component_placement``：按可用 GPU 调整 ``env`` 与 ``rollout`` 的 placement。
4. **真机评测：** 在 ``cluster.node_groups`` 中填写机械臂 IP 与节点拓扑（见 :doc:`../guides/realworld`）。

从训练配置派生
--------------

``examples/embodiment/config/`` 中的纯训练配置通常将模型预设绑定到 ``@actor.model``，而 ``rollout.model`` 缺少 ``model_type`` 和 ``num_action_chunks``，直接交给 ``run_eval.sh`` 会触发校验失败。将训练 YAML 复制到 ``evaluations/<benchmark>/``，删除 ``algorithm`` 与 ``actor``，把模型预设从 ``@actor.model`` 重绑定到 ``@rollout.model``，并开启评测模式：

.. code-block:: yaml

   defaults:
     - env/behavior_r1pro@env.eval
     - model/pi0_5@rollout.model        # 从 @actor.model 重绑定
   runner:
     task_type: embodied_eval
     only_eval: True

配置回退
--------

若 ``evaluations/<benchmark>/<config>.yaml`` 不存在，``run_eval.sh`` 仅对 ``examples/embodiment/config/`` 中已定义 ``rollout.model``\ （含 ``model_type`` 与 ``num_action_chunks``）的评测就绪配置生效。详见 :doc:`cli`。
