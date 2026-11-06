检查点恢复
=================

意外情况 —— 网络错误、断电、节点被抢占 —— 都可能中断一个长时间运行的分布式任务。  
为了解决这一问题，RLinf 会在每隔 ``runner.save_interval`` 步时保存一个完整的检查点，  
并允许你从最近的快照恢复，最大限度减少工作损失。  

检查点布局
-----------------

假设有如下 YAML 片段：

.. code-block:: yaml

   runner:
     task_type: math
     logger:
       log_path: ${runner.output_dir}/${runner.experiment_name}
       project_name: rlinf
       experiment_name: ${runner.experiment_name}

     save_interval: 50          
     experiment_name: grpo-1.5b
     output_dir: ./logs


如果使用 Megatron 作为训练后端，其检查点会出现在 `output_dir/experiment_name/checkpoints/` 下,
而如果使用 FSDP/FSDP2 作为训练后端，其检查点会出现在 `log_path/experiment_name/checkpoints/` 下。

Megatron 检查点
~~~~~~~~~~~~~~~~

Megatron检查点文件结构如下：

.. code-block:: text

   logs/grpo-1.5b/checkpoints/
   ├── global_step_50/
   │   ├── actor/
   │   │   ├── iter_0000050/
   │   │   │   ├── mp_rank_00/
   │   │   │   │   ├── distrib_optim.pt
   │   │   │   │   └── model_optim_rng.pt
   │   │   │   └── mp_rank_01/                 
   │   │   │       ├── distrib_optim.pt
   │   │   │       └── model_optim_rng.pt
   │   │   └── latest_checkpointed_iteration.txt
   │   └── data/
   │       └── data.pt                         
   └── global_step_100/
       └── …

关键点
^^^^^^^^^^^^^^^

* **分片权重** —— ``mp_rank_*`` 中的文件遵循 Megatron 的张量并行布局；每个 GPU 只会重新加载属于自己的分片。  
* **优化器 / RNG 状态** —— *同时* 保存了 Adam 参数（``distrib_optim.pt``）和随机数生成器，确保恢复后可以比特级复现。  
* **数据采样器** —— ``data.pt`` 存储了 dataloader，保证不会遗漏或重复样本。  

FSDP/FSDP2 检查点
~~~~~~~~~~~~~~~~~~

FSDP/FSDP2 根据 actor worker 的实现，使用 DCP（``torch.distributed.checkpoint``）或按 rank 保存的 ``local_shard`` 检查点。默认 DCP 格式的目录结构如下：

.. code-block:: text

   experiment_name/checkpoints/
   ├── global_step_10/
   │   └── actor/
   │       ├── dcp_checkpoint/
   │       │   ├── __0_0.distcp
   │       │   ├── __1_0.distcp
   │       │   ├── __2_0.distcp
   │       │   └── __3_0.distcp
   │       └── model_state_dict/
   │           └── full_weights.pt
   └── global_step_20/
       └── …


DCP 将训练状态保存到一组分布式检查点文件（``.distcp``）中。可选导出的 ``model_state_dict/full_weights.pt`` 只包含模型权重；恢复训练时应使用完整的检查点目录，同时恢复优化器、scheduler 和 RNG 状态。

部分 worker 为每个 rank 单独保存一个文件。SAC 和 DAgger 在 ``actor.fsdp_config.use_orig_params`` 为 true 时选择 ``local_shard`` 格式，IQL 则使用此格式保存 policy、critic 和 value 模型。例如，两个 rank 的 SAC actor 目录包含：

.. code-block:: text

   global_step_10/actor/
   └── local_shard_checkpoint/
       ├── checkpoint_rank_0.pt
       └── checkpoint_rank_1.pt

每个文件保存对应 rank 的模型分片、优化器状态、scheduler 状态和 RNG 状态。FSDP2 会将文件中的局部张量恢复到当前模型的分布式参数中，也支持各 rank 分片大小不等的参数。恢复时必须保持模型、FSDP 版本、world size、分片配置以及 rank 与分片的对应关系一致；``local_shard`` 不会为不同拓扑重新分配分片。请保留所有 rank 的文件，使用相同的启动脚本和配置，并按下文设置 ``runner.resume_dir``；worker 会自动选择对应的检查点格式。


恢复训练
-----------------

1. **选择最新的检查点**

   如果 ``global_step_10/`` 是编号最高的目录，它就是最新的快照。  

2. **修改 YAML**

   .. code-block:: yaml

      runner:
        resume_dir: ${runner.output_dir}/${runner.experiment_name}/checkpoints/global_step_10

3. **完全按原方式重新启动**

   启动 Ray，然后运行相同的 ``run_main_*.sh`` 启动脚本。  
   RLinf 会自动检测到 ``resume_dir`` 并：  

   * 在每个节点/rank 上恢复模型分片、优化器、RNG 和 dataloader 状态。  
   * 从 ``global_step_10`` 继续计数 —— 下一个保存的检查点将是 ``global_step_20`` （因为 ``save_interval`` 为 10）。  

.. tip::

   想验证恢复是否成功，可以查看日志行。  
   如果下一次训练从 step 30 开始，就说明恢复正常！


选择并恢复最佳 SFT checkpoint
-----------------------------

配置 ``runner.early_stop``，按验证指标选择 ``best_model/``，并在连续多次没有改善时停止训练：

.. code-block:: yaml

   runner:
     early_stop:
       enabled: true
       monitor: value_spearman
       mode: max
       patience: 5
       min_delta: 0.001

loss 使用 ``mode: min``，分数使用 ``mode: max``。指标名必须与 worker 验证方法返回的 key 一致；缺少该指标时，错误信息会列出可用指标。改善幅度必须超过 ``min_delta``；非有限值计入 patience，但不会被选为最佳 checkpoint。

省略 ``mode`` 时，``val_accuracy`` 默认使用 ``max``，其他指标默认使用 ``min``。设置 ``enabled: false`` 后，仍会选择最佳 checkpoint，并训练至配置的步数上限。

SFT checkpoint 在 ``actor/`` 旁保存 ``runner_state.json``，记录验证后的训练步数、最佳指标值和 patience 计数。将 ``runner.resume_dir`` 设为 ``global_step_<N>/`` 或 ``best_model/``，即可随 worker checkpoint 一起恢复这些状态。恢复时保持 ``monitor``、``mode`` 和 ``min_delta`` 一致。已耗尽 patience 的 checkpoint 不再执行训练步骤。启用定期保存后，early-stop 也会在停止的那一步保存 checkpoint。

旧版 ``global_step_<N>/`` checkpoint 仍可加载；由于其中没有 runner 状态，日志会提示最佳指标和 patience 从头开始跟踪，首次有限验证值会替换 ``best_model/``。若保存中断后留有 ``runner_state.incomplete``，恢复时会报错。数据加载器和 RNG 的恢复取决于对应 worker 的 checkpoint 支持。
