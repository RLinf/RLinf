Embodied SFT Model (FSDP)
=========================

Plug a PyTorch or Hugging Face Transformers model and dataset into RLinf's FSDP supervised fine-tuning (SFT) pipeline. Once integrated, ``SFTRunner`` handles training, evaluation, and checkpoint resume.

The integration proceeds through four checkpoints. First, register a model builder and confirm ``get_model`` resolves it. Next, subclass ``FSDPSftWorker`` and implement its three abstract methods with a concrete dataset. Then write the Hydra config and launch command. Finally, verify batching, loss backward, and evaluation locally on CPU before scaling to GPUs.

At a Glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Item
     - Details
   * - Files to create/edit
     - ``rlinf/workers/sft/<name>_sft_worker.py`` (or an external module) and ``examples/sft/config/<name>.yaml``
   * - Extension point
     - ``register_model`` and ``FSDPSftWorker`` (see :ref:`out-of-tree-extensions` for external packages)
   * - Required methods
     - ``build_dataloader``, ``get_train_model_output``, and ``get_eval_model_output``
   * - Launch command
     - ``bash examples/sft/run_vla_sft.sh <config_name>`` (or ``bash examples/sft/run_vlm_sft.sh <config_name>`` for VLM SFT)

1. Register the Model Builder
-----------------------------

``FSDPSftWorker.model_provider_func()`` calls ``get_model(self.cfg.actor.model)`` first and falls back to ``super().model_provider_func()`` only when ``get_model`` returns ``None``. Register a builder with ``register_model`` so ``SupportedModel`` validation and ``get_model`` both resolve your model.

The minimal token classifier below stands in for a Hugging Face ``AutoModelForCausalLM`` or VLA backbone and runs without external weights:

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

For a Hugging Face model, call ``AutoModelForCausalLM.from_pretrained(cfg.model_path, torch_dtype=torch_dtype, trust_remote_code=cfg.get("trust_remote_code", False))`` inside ``build_my_model``. Choose how to register your model builder:

.. tab-set::

   .. tab-item:: In-Tree (Inside RLinf)
      :sync: ext-in-tree

      Add ``register_model("my_new_model", build_my_model, category="sft")`` to ``rlinf/models/__init__.py`` so every worker process imports it automatically.

   .. tab-item:: Out-of-Tree (External Repo)
      :sync: ext-out-of-tree

      Keep ``register()`` in your own package (for example ``my_pkg/rlinf_ext.py``) and export ``RLINF_EXT_MODULE`` before launching so every Ray worker and standalone checkpoint convertor loads the registration automatically (see :ref:`out-of-tree-extensions`):

      .. code-block:: bash

         export RLINF_EXT_MODULE=my_pkg.rlinf_ext

2. Implement the SFT Worker Subclass
------------------------------------

``SFTRunner`` in ``rlinf/runners/sft_runner.py`` drives ``FSDPSftWorker`` in ``rlinf/workers/sft/fsdp_sft_worker.py`` in three stages:

1. ``FSDPSftWorker.__init__`` builds the training and validation dataloaders through ``build_dataloader``, and ``init_worker`` sets up the FSDP model, optimizer, and LR scheduler.
2. Each step calls ``run_training()``, which pulls ``self.gradient_accumulation`` micro-batches, invokes ``get_train_model_output(batch)``, scales the loss by ``1 / self.gradient_accumulation``, and runs backward and optimizer updates.
3. At ``runner.val_check_interval``, ``run_eval()`` iterates over ``self.eval_data_loader``, sums the integer return values of ``get_eval_model_output(batch)``, and divides by ``eval_step * self.eval_batch_size`` to report ``eval_accuracy``.

Implement a dataset and an ``FSDPSftWorker`` subclass under ``rlinf/workers/sft/<name>_sft_worker.py`` (or your external module) with all three abstract methods:

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

Read the three worker methods in call order:

- ``build_dataloader(self, data_paths, eval_dataset=False)`` runs in ``FSDPSftWorker.__init__`` for ``data.train_data_paths`` (``eval_dataset=False``) and ``data.val_data_paths`` (``eval_dataset=True``). Return ``(data_loader, data_config)`` where ``data_config`` is a metadata dict. Keep ``drop_last=False`` during evaluation so small validation sets do not drop all batches, and make sure ``collate_fn`` emits a dict containing every key read by your training and evaluation steps.
- ``get_train_model_output(self, batch)`` moves tensors to ``self.device``, runs the model inside ``self.amp_context``, and returns a scalar ``loss`` tensor with gradients attached plus a ``step_metrics`` dict of detached Python floats or ints.
- ``get_eval_model_output(self, batch)`` runs inference under ``torch.no_grad()`` and returns the integer number of correct predictions in ``batch``. If your task skips inline evaluation (as ``FSDPVlaSftWorker`` does), raise ``NotImplementedError`` here and set ``runner.val_check_interval: -1``.

If your model fits an existing VLA or VLM pipeline, add a branch in ``FSDPVlaSftWorker`` (``rlinf/workers/sft/fsdp_vla_sft_worker.py``) or ``FSDPVlmSftWorker`` (``rlinf/workers/sft/fsdp_vlm_sft_worker.py``) to reuse their launch scripts unchanged. For a standalone worker class, wire it into an entry script modeled on ``examples/sft/train_vla_sft.py``.

3. Configure and Launch Training
--------------------------------

Create a Hydra config under ``examples/sft/config/<config_name>.yaml`` starting with small batch sizes:

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

Launch training with the SFT runner script (export ``RLINF_EXT_MODULE`` first when your model and worker live in an external package):

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh <config_name>

Use ``bash examples/sft/run_vlm_sft.sh <config_name>`` for VLM SFT configs. Once a short run completes cleanly, increase ``global_batch_size`` and set ``save_interval``.

4. Verify Locally on CPU
------------------------

Before allocating GPUs or starting Ray, exercise model registration, dataloader batching, loss backward, and evaluation directly on CPU using the classes from steps 1 and 2:

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

Expected output (exact loss depends on weight initialization):

.. code-block:: text

   SFT check passed: samples=8, loss=0.8288, eval_accuracy=0.50

Troubleshooting
---------------

- ``KeyError`` in ``get_train_model_output``: your dataset or ``collate_fn`` does not emit every key read by the training step.
- ``Expected all tensors to be on the same device``: move every tensor field in ``batch`` with ``.to(self.device)``.
- ``global_batch_size is not divisible by micro_batch_size * world_size``: set ``actor.global_batch_size`` to a multiple of ``actor.micro_batch_size * world_size``.
- Unexpectedly low ``eval_accuracy``: check answer extraction and confirm ``actor.eval_batch_size`` matches the evaluation dataloader batch size.
- Repeated or skipped samples after resume: override ``save_checkpoint`` and ``load_checkpoint`` to persist ``_data_epoch`` and dataloader state (see ``FSDPVlaSftWorker`` or ``FSDPVlmSftWorker``).
