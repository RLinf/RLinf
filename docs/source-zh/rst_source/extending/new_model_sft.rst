接入 SFT 模型
=============

将 PyTorch 或 Hugging Face Transformers 模型与数据集接入 RLinf 的 FSDP 监督微调（SFT）流水线。接入完成后，即可使用统一的 SFT runner 运行训练、评估和断点续训。

接入速览
--------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 项目
     - 说明
   * - 新增或修改的文件
     - ``rlinf/workers/sft/<name>_sft_worker.py``\ （或外部模块）以及 ``examples/sft/config/<name>.yaml``
   * - 扩展点
     - ``register_model`` 与 ``FSDPSftWorker``\ （外部包注册见 :ref:`out-of-tree-extensions`\ ）
   * - 必选方法
     - ``build_dataloader``、``get_train_model_output`` 与 ``get_eval_model_output``
   * - 启动命令
     - ``bash examples/sft/run_vla_sft.sh <config_name>``\ （VLM 任务使用 ``bash examples/sft/run_vlm_sft.sh <config_name>``\ ）

SFT 执行流程
------------

RLinf 的 SFT 训练由 ``rlinf/runners/sft_runner.py`` 中的 ``SFTRunner`` 和 ``rlinf/workers/sft/fsdp_sft_worker.py`` 中的抽象基类 ``FSDPSftWorker`` 协同完成：

1. ``SFTRunner.init_workers()`` 初始化 worker（模型、优化器、学习率调度器和数据加载器），并在配置了 ``runner.resume_dir`` 时恢复 checkpoint。
2. 每个训练步调用 ``FSDPSftWorker.run_training()``，依次读取 ``self.gradient_accumulation`` 个 micro-batch、调用 ``get_train_model_output(batch)``、执行梯度缩放反向传播，并更新优化器与学习率调度器。
3. 到达 ``runner.val_check_interval`` 或 ``runner.save_interval`` 后，``SFTRunner.run()`` 调用 ``run_eval()``\ （逐 batch 调用 ``get_eval_model_output(batch)`` 并汇总 ``eval_accuracy``\ ）和 ``save_checkpoint()``。

接入新模型和数据集的核心，是继承 ``FSDPSftWorker`` 并实现三个抽象方法：

.. code-block:: python

   @abstractmethod
   def build_dataloader(self):
       raise NotImplementedError

   @abstractmethod
   def get_train_model_output(
       self, batch: dict[str, Any]
   ) -> tuple[torch.Tensor, dict[str, Any]]:
       raise NotImplementedError

   @abstractmethod
   def get_eval_model_output(self, batch: dict[str, Any]):
       raise NotImplementedError

1. 注册模型类型与构建函数
-------------------------

使用 ``register_model`` 注册模型构建函数，让配置校验（``SupportedModel``\ ）和 ``get_model(cfg)`` 都能识别你的模型。如果代码位于外部仓库并通过 Ray worker 运行，请导出 ``RLINF_EXT_MODULE`` 并在模块中提供 ``register()``\ （见 :ref:`out-of-tree-extensions`\ ）：

.. code-block:: python

   from rlinf.models import register_model


   def build_my_model(cfg, torch_dtype):
       from transformers import AutoModelForCausalLM

       return AutoModelForCausalLM.from_pretrained(
           cfg.model_path,
           torch_dtype=torch_dtype,
           trust_remote_code=cfg.get("trust_remote_code", False),
       )


   def register():
       register_model("my_new_model", build_my_model, category="sft")

在 YAML 中选用注册名称：

.. code-block:: yaml

   actor:
     model:
       model_type: "my_new_model"
       model_path: "/path/to/your/model"

验证 FSDP 模型加载
~~~~~~~~~~~~~~~~~~

``FSDPSftWorker.model_provider_func()`` 按以下顺序加载模型：

.. code-block:: python

   model = get_model(self.cfg.actor.model)
   if model is not None:
       return model
   return super().model_provider_func()

请确保 ``get_model(cfg)`` 能返回对应的 PyTorch 模型实例（或交由 ``FSDPModelManager.model_provider_func()`` 加载），且模型支持可微的训练前向计算。

2. 创建 SFT Worker 子类
-----------------------

新建继承自 ``FSDPSftWorker`` 的 worker 子类（例如 ``rlinf/workers/sft/fsdp_my_model_sft_worker.py``\ ），实现三个方法：

.. code-block:: python

   from typing import Any

   import torch
   from omegaconf import DictConfig

   from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


   class FSDPMyModelSftWorker(FSDPSftWorker):
       def __init__(self, cfg: DictConfig):
           super().__init__(cfg)

       def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
           # 根据 self.cfg.data 构建训练或评估数据集与 dataloader
           # 返回 (data_loader, data_config)
           ...
           return data_loader, {"num_samples": len(dataset)}

       def get_train_model_output(
           self, batch: dict[str, Any]
       ) -> tuple[torch.Tensor, dict[str, Any]]:
           # 执行前向计算并返回 (loss, step_metrics)
           ...
           return loss, {"loss": loss.detach().item()}

       def get_eval_model_output(self, batch: dict[str, Any]) -> int:
           # 执行推理并返回当前 batch 中预测正确的样本数
           ...
           return correct_count

你可以仿照 ``examples/sft/train_vla_sft.py``\ （或 ``examples/sft/train_vlm_sft.py``\ ）在入口脚本中启动自定义 worker；如果模型属于现有的 VLA 或 VLM 体系，也可以直接在 ``FSDPVlaSftWorker``\ （``rlinf/workers/sft/fsdp_vla_sft_worker.py``\ ）或 ``FSDPVlmSftWorker``\ （``rlinf/workers/sft/fsdp_vlm_sft_worker.py``\ ）中增加分支，直接复用现有启动脚本。

3. 构建数据加载器
-----------------

``FSDPSftWorker.__init__`` 在设置了 ``data.train_data_paths`` 时调用 ``self.build_dataloader(self.cfg.data.train_data_paths, eval_dataset=False)``，在设置了 ``data.val_data_paths`` 时调用 ``self.build_dataloader(self.cfg.data.val_data_paths, eval_dataset=True)``。该方法需返回 ``(data_loader, data_config)``，其中 ``data_config`` 为数据元信息字典（例如 ``{"num_samples": len(dataset)}``\ ）。

``run_training()`` 内部读取 ``batch = next(self.data_iter)`` 后直接送入 ``self.get_train_model_output(batch)``，因此训练和评估函数读取的每个字段都必须由 ``collate_fn`` 产出：

- **训练 batch**：包含模型输入（如 ``input_ids``、``attention_mask``\ ）以及 ``labels`` 或构造 ``labels`` 所需的字段。
- **评估 batch**：包含推理输入以及计算准确率所需的参考答案。

编写 dataloader 时注意避开三类常见错误：

1. ``collate_fn`` 返回了 ``list[dict]``，但训练函数按 ``dict`` 索引。
2. 部分样本缺少多模态字段，导致 batch 拼接错位。
3. 评估集样本少于单步全局 batch 大小时仍开启 ``drop_last=True``，导致评估样本被全部丢弃。

4. 计算训练损失
---------------

``FSDPSftWorker.run_training()`` 会解包 ``loss, step_metrics = self.get_train_model_output(batch)``，将 ``step_metrics`` 合并进训练日志，再按 ``self.gradient_accumulation`` 缩放 ``loss`` 并执行 ``self.grad_scaler.scale(loss).backward()``。

返回值应为带梯度的标量 ``loss`` 张量和仅含标量数值的 ``step_metrics`` 字典：

.. code-block:: python

   def get_train_model_output(
       self, batch: dict[str, Any]
   ) -> tuple[torch.Tensor, dict[str, Any]]:
       input_ids = batch["input_ids"].to(self.device)
       attention_mask = batch["attention_mask"].to(self.device)
       labels = batch["labels"].to(self.device)

       with self.amp_context:
           outputs = self.model(
               input_ids=input_ids,
               attention_mask=attention_mask,
               labels=labels,
           )
       loss = outputs.loss
       return loss, {"loss": loss.detach().item()}

5. 计算评估指标
---------------

``FSDPSftWorker.run_eval()`` 会遍历 ``self.eval_data_loader``，把每次 ``get_eval_model_output(batch)`` 的返回值累加到 ``correct``，再除以 ``eval_step * self.eval_batch_size`` 并跨 rank 求平均，得到 ``eval_accuracy``。

.. code-block:: python

   def get_eval_model_output(self, batch: dict[str, Any]) -> int:
       # 1) 在 torch.no_grad() 下生成预测
       # 2) 与参考答案比对
       # 3) 返回当前 batch 中预测正确的样本数
       return correct

如果任务不需要在线评估（与 ``FSDPVlaSftWorker`` 一致），在 ``get_eval_model_output`` 中抛出 ``NotImplementedError`` 并将 ``runner.val_check_interval`` 设为 ``-1`` 即可。

6. YAML 配置与启动
------------------

在 ``examples/sft/config/<config_name>.yaml`` 中创建配置，建议先用较小的 batch size 跑通流程：

.. code-block:: yaml

   runner:
     task_type: sft
     max_epochs: 5
     val_check_interval: -1
     save_interval: -1

   actor:
     training_backend: fsdp
     micro_batch_size: 2
     global_batch_size: 32
     model:
       model_type: my_new_model
       model_path: /path/to/model

   data:
     train_data_paths: /path/to/train_path
     val_data_paths: /path/to/eval_path

使用 VLA 或 VLM 启动脚本运行 SFT（如果模型在外部包中注册，先导出 ``RLINF_EXT_MODULE``\ ）：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh <config_name>

跑通后再按需调大 ``global_batch_size``，并开启 ``val_check_interval`` 和 ``save_interval``。

常见问题排查
------------

- ``get_train_model_output`` 报 ``KeyError``：``collate_fn`` 未输出训练步骤所需的字段。
- ``Expected all tensors to be on the same device``：``batch`` 中仍有张量未调用 ``.to(self.device)``。
- ``global_batch_size is not divisible by micro_batch_size * world_size``：将 ``actor.global_batch_size`` 调整为 ``actor.micro_batch_size * world_size`` 的整数倍。
- ``eval_accuracy`` 异常偏低：检查答案提取逻辑，并确认 ``eval_batch_size`` 与评估 dataloader 的 batch size 一致。
- 断点续训后样本重复或跳过：重写 ``save_checkpoint`` 与 ``load_checkpoint`` 持久化 ``_data_epoch`` 及 dataloader 状态（参考 ``FSDPVlaSftWorker`` 或 ``FSDPVlmSftWorker``\ ）。
