训练流水线模式
==============

在具身 FSDP 训练中设置 ``runner.use_training_pipeline: True``，即可让环境交互与 actor 训练流水线并行。

开启后，env worker 会在 rollout 过程中将轨迹打包成 micro-batch 并流式发往 actor worker。actor worker 收到后即可开始训练，无需等待整轮 rollout 结束。

rollout 观测包含大尺寸 tensor，或 env worker 与 actor worker 跨节点通过广域网通信时，打包后的较小 payload 能降低传输开销。

配置方法
--------

在具身 FSDP 训练 YAML 中开启：

.. code-block:: yaml

   runner:
     use_training_pipeline: True

   algorithm:
     adv_type: gae
     normalize_advantages: True

限制条件
--------

- ``algorithm.adv_type`` 必须设为 ``gae``。
- 开启 ``algorithm.normalize_advantages`` 后，advantage 会在发往同一 actor rank 的 env worker 间聚合归一化，再流式发送 micro-batch。
- 仅支持 PPO 和 GRPO 类具身 FSDP actor loss，不支持 ``embodied_sac``、``embodied_dagger`` 和 ``embodied_nft``。
