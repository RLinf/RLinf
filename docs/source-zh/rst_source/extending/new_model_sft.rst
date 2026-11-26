具身 SFT 模型（FSDP）
=====================

将 PyTorch 或 Hugging Face Transformers 模型与数据集接入 RLinf 的 FSDP 监督微调（SFT）流水线。接入完成后，``SFTRunner`` 统一负责训练、评估和断点续训。

整个接入分为四个检查点：先注册模型构建函数并确认 ``get_model`` 能正确加载；再继承 ``FSDPSftWorker``，配合具体数据集实现三个抽象方法；随后编写 Hydra 配置并对接启动脚本；最后在本地 CPU 上验证数据批处理、反向传播和评估逻辑，确认无误后再上 GPU 运行。

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

1. 注册模型构建函数
-------------------

``FSDPSftWorker.model_provider_func()`` 会先调用 ``get_model(self.cfg.actor.model)``，仅在返回 ``None`` 时才回退到 ``super().model_provider_func()``。通过 ``register_model`` 注册构建函数，让 ``SupportedModel`` 配置校验和 ``get_model`` 都能解析你的模型。

下面的最小文本分类模型用来替代 Hugging Face 的 ``AutoModelForCausalLM`` 或 VLA 主干网络，不依赖外部权重即可运行：

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.nn.functional as F

   from rlinf.models import register_model


   class TinySFTModel(nn.Module):
       def __init__(self, vocab_size: int, hidden_dim: int, num_classes: int):
           super().__init__()
           self.embedding = nn.Embedding(vocab_size, hidden_dim)
           self.head = nn.Linear(hidden_dim, num_classes)

       def forward(
           self,
           input_ids: torch.Tensor,
           attention_mask: torch.Tensor,
           labels: torch.Tensor | None = None,
       ) -> dict[str, torch.Tensor]:
           mask = attention_mask.unsqueeze(-1).to(dtype=self.embedding.weight.dtype)
           pooled = (self.embedding(input_ids) * mask).sum(dim=1) / mask.sum(
               dim=1
           ).clamp_min(1.0)
           logits = self.head(pooled)
           outputs = {"logits": logits}
           if labels is not None:
               outputs["loss"] = F.cross_entropy(logits, labels)
           return outputs


   def build_my_model(cfg, torch_dtype):
       return TinySFTModel(
           vocab_size=cfg.vocab_size,
           hidden_dim=cfg.hidden_dim,
           num_classes=cfg.num_classes,
       ).to(dtype=torch_dtype)


   def register():
       register_model("my_new_model", build_my_model, category="sft")

接入 Hugging Face 模型时，在 ``build_my_model`` 中调用 ``AutoModelForCausalLM.from_pretrained(cfg.model_path, torch_dtype=torch_dtype, trust_remote_code=cfg.get("trust_remote_code", False))`` 即可。按代码存放位置选择注册方式：

.. tab-set::

   .. tab-item:: 主仓库内注册（In-Tree）
      :sync: ext-in-tree

      在 ``rlinf/models/__init__.py`` 中添加 ``register_model("my_new_model", build_my_model, category="sft")``，所有 worker 导入时自动生效。

   .. tab-item:: 外部仓库注册（Out-of-Tree）
      :sync: ext-out-of-tree

      在自己的包中（例如 ``my_pkg/rlinf_ext.py``\ ）保留上面的 ``register()`` 函数，并在启动前导出 ``RLINF_EXT_MODULE``，每个 Ray worker 和独立 checkpoint 转换工具都会自动加载注册逻辑（详见 :ref:`out-of-tree-extensions`\ ）：

      .. code-block:: bash

         export RLINF_EXT_MODULE=my_pkg.rlinf_ext

2. 实现 SFT Worker 子类
-----------------------

``rlinf/runners/sft_runner.py`` 中的 ``SFTRunner`` 按三个阶段驱动 ``rlinf/workers/sft/fsdp_sft_worker.py`` 中的 ``FSDPSftWorker``：

1. ``FSDPSftWorker.__init__`` 调用 ``build_dataloader`` 构建训练和评估数据加载器，随后 ``init_worker`` 初始化 FSDP 模型、优化器和学习率调度器。
2. 每个训练步调用 ``run_training()``，依次读取 ``self.gradient_accumulation`` 个 micro-batch、调用 ``get_train_model_output(batch)``、按 ``1 / self.gradient_accumulation`` 缩放损失，然后反向传播并更新优化器。
3. 到达 ``runner.val_check_interval`` 后，``run_eval()`` 遍历 ``self.eval_data_loader``，累加 ``get_eval_model_output(batch)`` 返回的正确样本数，再除以 ``eval_step * self.eval_batch_size`` 得到 ``eval_accuracy``。

在 ``rlinf/workers/sft/<name>_sft_worker.py``\ （或外部模块）中实现数据集和 ``FSDPSftWorker`` 子类，补齐三个抽象方法：

.. code-block:: python

   import json
   from typing import Any

   import torch
   from omegaconf import DictConfig
   from torch.utils.data import DataLoader, Dataset, DistributedSampler

   from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


   class TinySFTDataset(Dataset):
       def __init__(self, data_path: str):
           with open(data_path, encoding="utf-8") as f:
               self.samples = [json.loads(line) for line in f if line.strip()]

       def __len__(self) -> int:
           return len(self.samples)

       def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
           sample = self.samples[idx]
           return {
               "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
               "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
               "labels": torch.tensor(sample["label"], dtype=torch.long),
           }


   class TinySFTWorker(FSDPSftWorker):
       def __init__(self, cfg: DictConfig):
           super().__init__(cfg)

       def build_dataloader(self, data_paths: str, eval_dataset: bool = False):
           dataset = TinySFTDataset(data_paths)
           batch_size = self.eval_batch_size if eval_dataset else self.micro_batch_size
           sampler = None
           if torch.distributed.is_available() and torch.distributed.is_initialized():
               sampler = DistributedSampler(
                   dataset,
                   num_replicas=self._world_size,
                   rank=self._rank,
                   shuffle=not eval_dataset,
                   drop_last=not eval_dataset,
               )
           loader = DataLoader(
               dataset,
               batch_size=batch_size,
               sampler=sampler,
               shuffle=(sampler is None and not eval_dataset),
               drop_last=not eval_dataset,
           )
           return loader, {"num_samples": len(dataset)}

       def get_train_model_output(
           self, batch: dict[str, Any]
       ) -> tuple[torch.Tensor, dict[str, Any]]:
           with self.amp_context:
               outputs = self.model(
                   input_ids=batch["input_ids"].to(self.device),
                   attention_mask=batch["attention_mask"].to(self.device),
                   labels=batch["labels"].to(self.device),
               )
           loss = outputs["loss"]
           return loss, {"loss": loss.detach().item()}

       def get_eval_model_output(self, batch: dict[str, Any]) -> int:
           with torch.no_grad():
               outputs = self.model(
                   input_ids=batch["input_ids"].to(self.device),
                   attention_mask=batch["attention_mask"].to(self.device),
               )
           preds = outputs["logits"].argmax(dim=-1)
           return int((preds == batch["labels"].to(self.device)).sum().item())

按调用顺序说明三个抽象方法的职责：

- ``build_dataloader(self, data_paths, eval_dataset=False)``：在 ``FSDPSftWorker.__init__`` 中分别针对 ``data.train_data_paths``\ （``eval_dataset=False``\ ）和 ``data.val_data_paths``\ （``eval_dataset=True``\ ）调用，返回 ``(data_loader, data_config)``，其中 ``data_config`` 是记录样本数等信息的字典。评估阶段保持 ``drop_last=False``，避免验证集样本少于单步 batch 时被全部丢弃；同时确保 ``collate_fn`` 输出包含训练和评估所需全部字段的字典。
- ``get_train_model_output(self, batch)``：将张量移动到 ``self.device``，在 ``self.amp_context`` 下前向计算，返回带梯度的标量 ``loss`` 张量和仅含标量数值的 ``step_metrics`` 字典。
- ``get_eval_model_output(self, batch)``：在 ``torch.no_grad()`` 下推理，返回当前 ``batch`` 中预测正确的样本数（整数）。如果任务不做在线评估（与 ``FSDPVlaSftWorker`` 一致），直接抛出 ``NotImplementedError`` 并将 ``runner.val_check_interval`` 设为 ``-1``。

如果模型属于现有的 VLA 或 VLM 体系，可以直接在 ``FSDPVlaSftWorker``\ （``rlinf/workers/sft/fsdp_vla_sft_worker.py``\ ）或 ``FSDPVlmSftWorker``\ （``rlinf/workers/sft/fsdp_vlm_sft_worker.py``\ ）中增加分支，复用现有启动脚本。需要独立 worker 时，仿照 ``examples/sft/train_vla_sft.py`` 在入口脚本中拉起自定义子类。

3. 配置与启动训练
-----------------

在 ``examples/sft/config/<config_name>.yaml`` 中新建 Hydra 配置，建议先从较小的 batch size 开始：

.. code-block:: yaml

   runner:
     task_type: sft
     max_epochs: 5
     val_check_interval: 1
     save_interval: -1

   actor:
     training_backend: fsdp
     micro_batch_size: 2
     eval_batch_size: 2
     global_batch_size: 8
     model:
       model_type: my_new_model
       vocab_size: 32
       hidden_dim: 16
       num_classes: 2
       precision: fp32
       is_lora: False

   data:
     train_data_paths: /path/to/train.jsonl
     val_data_paths: /path/to/val.jsonl

使用 SFT 启动脚本运行训练（模型和 worker 位于外部包时先导出 ``RLINF_EXT_MODULE``\ ）：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh <config_name>

VLM SFT 配置改用 ``bash examples/sft/run_vlm_sft.sh <config_name>``。短程试跑通过后，再按需调大 ``global_batch_size`` 并开启 ``save_interval``。

4. 在本地 CPU 上验证
--------------------

占用 GPU 或启动 Ray 前，可以先用第 1 步和第 2 步定义的类在本地 CPU 上验证模型注册、dataloader 组批、反向传播和评估统计：

.. code-block:: python

   import json
   import tempfile
   from contextlib import nullcontext
   from pathlib import Path

   import torch
   from omegaconf import OmegaConf

   from rlinf.models import get_model

   register()

   with tempfile.TemporaryDirectory() as tmpdir:
       data_file = Path(tmpdir) / "samples.jsonl"
       records = [
           {"input_ids": [1, 2, 3, 0], "attention_mask": [1, 1, 1, 0], "label": i % 2}
           for i in range(8)
       ]
       data_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

       cfg = OmegaConf.create(
           {
               "actor": {
                   "micro_batch_size": 2,
                   "eval_batch_size": 2,
                   "model": {
                       "model_type": "my_new_model",
                       "vocab_size": 32,
                       "hidden_dim": 16,
                       "num_classes": 2,
                       "precision": "fp32",
                       "is_lora": False,
                       "load_to_device": False,
                   },
               }
           }
       )

       worker = TinySFTWorker.__new__(TinySFTWorker)
       worker.cfg = cfg
       worker.device = torch.device("cpu")
       worker.micro_batch_size = cfg.actor.micro_batch_size
       worker.eval_batch_size = cfg.actor.eval_batch_size
       worker.amp_context = nullcontext()
       worker.model = get_model(cfg.actor.model)

       train_loader, train_meta = worker.build_dataloader(str(data_file), eval_dataset=False)
       eval_loader, eval_meta = worker.build_dataloader(str(data_file), eval_dataset=True)

       train_batch = next(iter(train_loader))
       loss, step_metrics = worker.get_train_model_output(train_batch)
       loss.backward()
       assert worker.model.head.weight.grad is not None

       correct = sum(worker.get_eval_model_output(batch) for batch in eval_loader)
       eval_accuracy = correct / eval_meta["num_samples"]
       print(
           f"SFT check passed: samples={train_meta['num_samples']}, "
           f"loss={step_metrics['loss']:.4f}, eval_accuracy={eval_accuracy:.2f}"
       )

预期输出（具体 loss 取决于参数初始化）：

.. code-block:: text

   SFT check passed: samples=8, loss=0.8288, eval_accuracy=0.50

常见问题排查
------------

- ``get_train_model_output`` 报 ``KeyError``：数据集或 ``collate_fn`` 缺少训练步骤读取的字段。
- ``Expected all tensors to be on the same device``：``batch`` 中仍有张量未调用 ``.to(self.device)``。
- ``global_batch_size is not divisible by micro_batch_size * world_size``：将 ``actor.global_batch_size`` 设为 ``actor.micro_batch_size * world_size`` 的整数倍。
- ``eval_accuracy`` 异常偏低：检查答案提取逻辑，并确认 ``actor.eval_batch_size`` 与评估 dataloader 的 batch size 一致。
- 断点续训后样本重复或跳过：重写 ``save_checkpoint`` 与 ``load_checkpoint`` 持久化 ``_data_epoch`` 及 dataloader 状态（参考 ``FSDPVlaSftWorker`` 或 ``FSDPVlmSftWorker``\ ）。
