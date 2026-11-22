Add an SFT Model
================

Plug a PyTorch or Hugging Face Transformers model and dataset into RLinf's FSDP supervised fine-tuning (SFT) pipeline. Once integrated, you can train, evaluate, and resume from checkpoints using the shared SFT runner.

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

SFT Pipeline Flow
-----------------

RLinf's SFT loop connects two components: ``SFTRunner`` in ``rlinf/runners/sft_runner.py`` and the abstract worker base class ``FSDPSftWorker`` in ``rlinf/workers/sft/fsdp_sft_worker.py``:

1. ``SFTRunner.init_workers()`` initializes the worker (model, optimizer, LR scheduler, and dataloaders) and optionally loads ``runner.resume_dir``.
2. Each step calls ``FSDPSftWorker.run_training()``, which pulls ``self.gradient_accumulation`` micro-batches, invokes ``get_train_model_output(batch)``, runs scaled backward passes, and steps the optimizer and LR scheduler.
3. At ``runner.val_check_interval`` and ``runner.save_interval``, ``SFTRunner.run()`` calls ``run_eval()`` (which invokes ``get_eval_model_output(batch)`` and aggregates ``eval_accuracy``) and ``save_checkpoint()``.

To connect a custom model and dataset, subclass ``FSDPSftWorker`` and implement its three abstract methods:

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

1. Register the Model Type and Builder
--------------------------------------

Register a builder with ``register_model`` so config validation (``SupportedModel``) and ``get_model(cfg)`` recognize your model. For out-of-tree packages running on Ray workers, expose ``register()`` and export ``RLINF_EXT_MODULE`` (see :ref:`out-of-tree-extensions`):

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

Select the registered name in YAML:

.. code-block:: yaml

   actor:
     model:
       model_type: "my_new_model"
       model_path: "/path/to/your/model"

Verify FSDP Model Loading
~~~~~~~~~~~~~~~~~~~~~~~~~

``FSDPSftWorker.model_provider_func()`` builds the actor model with:

.. code-block:: python

   model = get_model(self.cfg.actor.model)
   if model is not None:
       return model
   return super().model_provider_func()

Make sure ``get_model(cfg)`` returns your PyTorch module (or that ``FSDPModelManager.model_provider_func()`` handles it) and that the returned model supports a differentiable training forward pass.

2. Create an SFT Worker
-----------------------

Create a subclass of ``FSDPSftWorker`` (for example ``rlinf/workers/sft/fsdp_my_model_sft_worker.py``) and implement the three methods:

.. code-block:: python

   from typing import Any

   import torch
   from omegaconf import DictConfig

   from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


   class FSDPMyModelSftWorker(FSDPSftWorker):
       def __init__(self, cfg: DictConfig):
           super().__init__(cfg)

       def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
           # Build train or eval dataset and dataloader from self.cfg.data
           # Return (data_loader, data_config)
           ...
           return data_loader, {"num_samples": len(dataset)}

       def get_train_model_output(
           self, batch: dict[str, Any]
       ) -> tuple[torch.Tensor, dict[str, Any]]:
           # Run forward pass and return (loss, step_metrics)
           ...
           return loss, {"loss": loss.detach().item()}

       def get_eval_model_output(self, batch: dict[str, Any]) -> int:
           # Run inference and return number of correct samples in the batch
           ...
           return correct_count

Launch this worker from an entry script modeled on ``examples/sft/train_vla_sft.py`` (or ``examples/sft/train_vlm_sft.py``). If your model fits directly into an existing worker family, you can instead add a branch to ``FSDPVlaSftWorker`` (``rlinf/workers/sft/fsdp_vla_sft_worker.py``) or ``FSDPVlmSftWorker`` (``rlinf/workers/sft/fsdp_vlm_sft_worker.py``) and reuse the existing launch scripts unchanged.

3. Build the Dataloader
-----------------------

``FSDPSftWorker.__init__`` calls ``self.build_dataloader(self.cfg.data.train_data_paths, eval_dataset=False)`` when ``data.train_data_paths`` is set, and ``self.build_dataloader(self.cfg.data.val_data_paths, eval_dataset=True)`` when ``data.val_data_paths`` is set. Return ``(data_loader, data_config)`` where ``data_config`` is a metadata dict (for example ``{"num_samples": len(dataset)}``).

Inside ``run_training()``, the worker reads ``batch = next(self.data_iter)`` and passes it directly to ``self.get_train_model_output(batch)``. Every key accessed in your training or evaluation method must be produced by your ``collate_fn``:

- **Train batch**: include model inputs (such as ``input_ids`` and ``attention_mask``) plus ``labels`` or the fields needed to build them.
- **Eval batch**: include inference inputs and reference targets for accuracy scoring.

Avoid three common dataloader mistakes:

1. Returning ``list[dict]`` from ``collate_fn`` when ``get_train_model_output`` indexes ``batch`` as a ``dict``.
2. Dropping optional multimodal fields on some samples so tensors misalign across a batch.
3. Setting ``drop_last=True`` on evaluation when the dataset is smaller than one batch across ranks.

4. Compute Training Loss
------------------------

``FSDPSftWorker.run_training()`` unpacks ``loss, step_metrics = self.get_train_model_output(batch)``, appends ``step_metrics`` to logged training metrics, divides ``loss`` by ``self.gradient_accumulation``, and calls ``self.grad_scaler.scale(loss).backward()``.

Return a scalar ``loss`` tensor with gradient attached and a ``step_metrics`` dict of detached floats or scalars:

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

5. Compute Evaluation Metrics
-----------------------------

``FSDPSftWorker.run_eval()`` iterates over ``self.eval_data_loader``, adds each return value of ``get_eval_model_output(batch)`` to ``correct``, and divides by ``eval_step * self.eval_batch_size`` to report ``eval_accuracy`` (averaged across ranks).

.. code-block:: python

   def get_eval_model_output(self, batch: dict[str, Any]) -> int:
       # 1) Generate predictions under torch.no_grad()
       # 2) Compare against ground-truth targets
       # 3) Return the number of correct samples in this batch
       return correct

If your task does not run inline evaluation (as in ``FSDPVlaSftWorker``), raise ``NotImplementedError`` in ``get_eval_model_output`` and set ``runner.val_check_interval: -1``.

6. YAML Configuration and Launch
--------------------------------

Create a config under ``examples/sft/config/<config_name>.yaml`` starting from conservative batch sizes:

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

Launch SFT training with the VLA or VLM entry script (export ``RLINF_EXT_MODULE`` first if your model lives outside RLinf):

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh <config_name>

Once training runs cleanly, scale up ``global_batch_size`` and enable ``val_check_interval`` and ``save_interval``.

Troubleshooting
---------------

- ``KeyError`` in ``get_train_model_output``: your ``collate_fn`` does not emit every key read by the training step.
- ``Expected all tensors to be on the same device``: move all tensor fields in ``batch`` with ``.to(self.device)``.
- ``global_batch_size is not divisible by micro_batch_size * world_size``: set ``actor.global_batch_size`` to a multiple of ``actor.micro_batch_size * world_size``.
- Unexpectedly low ``eval_accuracy``: verify answer extraction and confirm ``eval_batch_size`` matches the dataloader batch size.
- Repeated or skipped samples after resume: override ``save_checkpoint`` and ``load_checkpoint`` to persist ``_data_epoch`` and dataloader state (see ``FSDPVlaSftWorker`` or ``FSDPVlmSftWorker`` for examples).
