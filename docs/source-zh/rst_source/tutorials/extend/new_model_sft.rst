使用 FSDP+HuggingFace 添加新模型 SFT 训练
=============================================

本文档重点介绍如何使用 HuggingFace Transformers 库与 PyTorch FSDP（Fully Sharded Data Parallel，全分片数据并行）  
来训练和生成模型。它支持 HuggingFace 中实现的任意模型，只要兼容 PyTorch 即可。  
作为示例，本节将提供一个逐步的操作流程，展示如何按照 sft 模式将一个新的 HuggingFace 模型集成到 RLinf 中。  

前置条件
-------------

* 熟悉 **HuggingFace Transformers 库**  
* 理解 **RLinf** 框架架构  
* 掌握 **PyTorch** 与分布式训练知识  

本文目标
--------

你将学会：**把一个“新模型”接入 RLinf 的 SFT 训练流程**，并成功跑通训练 / 评估 / 断点续训。

本文基于 RLinf 当前 SFT 主流程：

- ``rlinf/runners/sft_runner.py``（训练调度器）
- ``rlinf/workers/sft/fsdp_sft_worker.py``（SFT Worker 基类）

当前 RLinf 中 SFT 的主要步骤分为如下几个部分：

1. 启动 Runner
2. Runner 初始化 Worker（加载模型、优化器、数据）
3. 每个 step 调用 Worker 的 ``run_training()``
4. 到达条件后调用 ``run_eval()`` / ``save_checkpoint()``
5. 重复直到训练结束

在代码里对应关系：

- ``SFTRunner.run()``：
  - 调 ``self.actor.run_training()``
  - 根据 ``val_check_interval`` 和 ``save_interval`` 决定 eval/save
- ``FSDPSftWorker.run_training()``：
  - 从 dataloader 拿 batch
  - 调用你实现的 ``get_train_model_output(batch)``
  - backward + optimizer step + lr scheduler step
- ``FSDPSftWorker.run_eval()``：
  - 逐 batch 调用你实现的 ``get_eval_model_output(batch)``
  - 汇总进行 sft 模型效果的评估 ``eval_accuracy``

所以你要适配新模型，核心需要实现在 ``rlinf/workers/sft/fsdp_sft_worker.py`` 中的三个抽象新方法，才能将新数据集以及新模型接入到 RLinf 的 SFT 训练流程中：

.. code:: python

    @abstractmethod
    def build_dataloader(self):
        raise NotImplementedError

    @abstractmethod
    def get_train_model_output(self, batch: dict[str, Any]):
        raise NotImplementedError

    @abstractmethod
    def get_eval_model_output(self, batch: dict[str, Any]):
        raise NotImplementedError


----

训练前置条件
----------------------------

开始适配前请确认：

- 下载需要 sft 训练的新模型权重（HF 路径或本地路径）
- 下载需要 sft 训练的新数据集（文本 / 图文 / 多模态）
- 理解训练数据格式（文本 / 图文 / 多模态）以及如何进行预处理
- 你知道监督目标（如 next-token loss、分类准确率）
- 准备好 eval 数据集进行模型验证

----

让 RLinf 配置文件识别你的模型类型
------------------------------

RLinf 通过 ``SupportedModel`` 识别模型类型。你需要：

1. 在 ``SupportedModel`` 加入你的新类型（sft 类别）
2. 在配置 YAML 里设置 ``actor.model.model_type`` 为这个值

示例：

.. code:: python

   class SupportedModel(Enum):
       ...
       MY_NEW_MODEL_SFT = ("my_new_model", "sft")

示例 YAML：

.. code:: yaml

   actor:
     model:
       model_type: "my_new_model"
       model_path: "/path/to/your/model"

----

确保 FSDP已经支持你的模型， ``get_model(...)`` 能返回你的模型
--------------------------------------------

``FSDPSftWorker.model_provider_func()`` 会调用：

.. code:: python

   model = get_model(self.cfg.actor.model)

必须保证 ``FSDPModelManager.model_provider_func()`` 能返回你的模型：

- ``get_model`` 能识别 ``my_new_model``
- 返回对象支持训练前向（通常是 ``model(..., labels=...)`` 返回 ``loss``）

----

新建一个 Worker 子类，实现 ``build_dataloader``、``get_train_model_output``、``get_eval_model_output``
------------------------------------

建议新建文件，例如：

- ``rlinf/workers/sft/fsdp_my_model_sft_worker.py``

继承 ``FSDPSftWorker``，实现 3 个方法。

.. code:: python

   from typing import Any
   import torch
   from omegaconf import DictConfig
   from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker

   class FSDPMyModelSftWorker(FSDPSftWorker):
       def __init__(self, cfg: DictConfig):
           super().__init__(cfg)

       def build_dataloader(self, data_paths: list[str], eval_dataset: bool = False):
           # 1) 构建 dataset
           # 3) 返回 data_loader 和 data_config(dict)
           ...
           return data_loader, {"num_samples": len(dataset)}

       def get_train_model_output(self, batch: dict[str, Any]):
           # 模型的核心训练过程
           # 返回 loss（Tensor）
           ...
           return loss

       def get_eval_model_output(self, batch: dict[str, Any]):
           # 模型的核心评估过程
           # 返回当前 batch 的正确样本数（整数）
           ...
           return correct_count

----

``build_dataloader`` 方法用于构建数据加载器，你需要确保返回的数据加载器能够正确地处理训练和评估数据。
--------------------

你必须保证 batch 字段和后续训练函数一致。

``run_training()`` 内部是：

- ``batch = next(self.data_iter)``
- ``losses = self.get_train_model_output(batch)``

也就是说你在 ``get_train_model_output`` 里访问的 key，必须由 collate 产出。

建议 checklist：

- 训练时 batch 至少有：
  - ``input_ids``（或你的同义字段）
  - ``attention_mask``（可选，但建议有）
  - ``labels`` 或可构造 labels 的字段

- 评估时 batch 至少有：
  - 推理输入
  - 参考答案（用于算准确率）

常见错误：

1. ``collate_fn`` 输出 ``list[dict]``，但训练代码当成 ``dict`` 用
2. 某些样本缺多模态字段，导致 batch 拼接错位
3. eval 还在 ``drop_last=True``，导致评估样本被丢弃

----

``get_train_model_output`` 方法用于获取模型的训练输出，你需要确保返回的输出能够正确地进行训练。
---------------------------

``FSDPSftWorker`` 会对你返回的 loss 做：

- 支持 list/tuple/tensor 自动归一
- gradient accumulation
- scaler.backward

所以你只要保证最后返回的是 loss（或可堆叠 loss 列表）。

标准 CausalLM 写法（推荐）：

.. code:: python

   def get_train_model_output(self, batch):
       input_ids = batch["input_ids"].to(self.device)
       attention_mask = batch["attention_mask"].to(self.device)
       labels = batch["labels"].to(self.device)

       with self.amp_context:
           outputs = self.model(
               input_ids=input_ids,
               attention_mask=attention_mask,
               labels=labels,
           )
       return outputs.loss

----

``get_eval_model_output`` 方法用于获取模型的评估输出，你需要确保返回的输出能够正确地进行评估。
--------------------------

``run_eval()`` 里逻辑是：

- 累加每个 batch 的返回值到 ``correct``
- 再除以 ``total`` 得 ``eval_accuracy``

所以你的 ``get_eval_model_output`` 应该返回 **当前 batch 的正确样本数**（整数）。

示例：

.. code:: python

   def get_eval_model_output(self, batch):
       # 1) 生成预测
       # 2) 与结果进行正确性比较
       # 3) 返回正确数量
       return correct

----


YAML 配置文件
----------------------------

建议先用保守参数跑通：

.. code:: yaml

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
     eval_data_paths: /path/to/eval_path

跑通后再逐步加大 batch、打开 eval/save。

----


常见问题排查
----------------------------

1. ``KeyError: xxx``  
   - collate 没有产出训练函数需要的字段

2. ``Expected all tensors on same device``  
   - 某些 batch 字段没 ``to(self.device)``

3. ``global_batch_size is not divisible ...``  
   - 调整 ``global_batch_size / micro_batch_size / world_size``

4. ``eval_accuracy 异常偏低``  
   - 检查评估提取答案逻辑
   - 检查 ``drop_last`` 是否导致评估样本丢失

5. ``resume 后数据重复/跳过``  
   - 检查 ``_data_epoch`` / ``_data_iter_offset`` 保存与恢复流程

----