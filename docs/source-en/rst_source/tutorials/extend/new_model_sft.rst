Add a New Model for SFT Training with FSDP + HuggingFace
=========================================================

This document explains how to use HuggingFace Transformers together with PyTorch FSDP
(Fully Sharded Data Parallel) to train and generate with models.  
It supports any model implemented in HuggingFace as long as it is compatible with PyTorch.  
As an example, this guide provides a step-by-step workflow showing how to integrate a new
HuggingFace model into RLinf under SFT mode.

Prerequisites
-------------

- Familiarity with **HuggingFace Transformers**
- Understanding of the **RLinf** framework architecture
- Basic knowledge of **PyTorch** and distributed training

Goal
----

You will learn how to integrate a new model into RLinf's SFT training pipeline, and run:

- training
- evaluation
- resume from checkpoint

This tutorial is based on RLinf's current SFT pipeline:

- ``rlinf/runners/sft_runner.py`` (training scheduler)
- ``rlinf/workers/sft/fsdp_sft_worker.py`` (SFT worker base class)

The main SFT process in RLinf is:

1. Start the Runner
2. Runner initializes Worker (model, optimizer, data)
3. Each step calls Worker ``run_training()``
4. At configured intervals, call ``run_eval()`` / ``save_checkpoint()``
5. Repeat until training ends

Code mapping:

- ``SFTRunner.run()``
  - calls ``self.actor.run_training()``
  - decides eval/save based on ``val_check_interval`` and ``save_interval``
- ``FSDPSftWorker.run_training()``
  - reads batch from dataloader
  - calls your ``get_train_model_output(batch)``
  - backward + optimizer step + lr scheduler step
- ``FSDPSftWorker.run_eval()``
  - calls your ``get_eval_model_output(batch)`` for each batch
  - aggregates SFT evaluation metric ``eval_accuracy``

To adapt a new model, the key is to implement the three abstract methods in
``rlinf/workers/sft/fsdp_sft_worker.py`` so the new dataset and model can enter the SFT pipeline:

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

Pre-training Checklist
----------------------

Before adaptation, make sure you have:

- Downloaded model weights for SFT (HF path or local path)
- Downloaded the target SFT dataset (text / vision-language / multimodal)
- Understood dataset format and preprocessing logic
- Defined the supervision target (e.g., next-token loss, classification accuracy)
- Prepared an evaluation dataset for validation

----

Make RLinf Config Recognize Your Model Type
-------------------------------------------

RLinf uses ``SupportedModel`` to identify model types. You need to:

1. Add your new model type to ``SupportedModel`` (SFT category)
2. Set ``actor.model.model_type`` to that value in YAML

Example:

.. code:: python

   class SupportedModel(Enum):
       ...
       MY_NEW_MODEL_SFT = ("my_new_model", "sft")

YAML example:

.. code:: yaml

   actor:
     model:
       model_type: "my_new_model"
       model_path: "/path/to/your/model"

----

Ensure FSDP Supports Your Model 
---------------------------------

``FSDPSftWorker.model_provider_func()`` calls:

.. code:: python

   model = get_model(self.cfg.actor.model)

You must ensure ``FSDPModelManager.model_provider_func()`` can return your model:

- ``get_model`` can recognize ``my_new_model``
- returned model supports training forward (typically ``model(..., labels=...)`` returns ``loss``)

----

Create a Worker
------------------

Recommended new file:

- ``rlinf/workers/sft/fsdp_my_model_sft_worker.py``

Inherit from ``FSDPSftWorker`` and implement the 3 methods.

.. code:: python

   from typing import Any
   import torch
   from omegaconf import DictConfig
   from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker

   class FSDPMyModelSftWorker(FSDPSftWorker):
       def __init__(self, cfg: DictConfig):
           super().__init__(cfg)

       def build_dataloader(self, data_paths: list[str], eval_dataset: bool = False):
           # 1) Build dataset
           # 3) Return data_loader and data_config(dict)
           ...
           return data_loader, {"num_samples": len(dataset)}

       def get_train_model_output(self, batch: dict[str, Any]):
           # Core training logic
           # Return loss (Tensor)
           ...
           return loss

       def get_eval_model_output(self, batch: dict[str, Any]):
           # Core evaluation logic
           # Return number of correct samples in current batch (int)
           ...
           return correct_count

----

Implement build_dataloader
------------------------------

``build_dataloader`` constructs your dataloader. You must ensure it can correctly serve both train and eval.

You must keep batch fields consistent with later training logic.

Inside ``run_training()``:

- ``batch = next(self.data_iter)``
- ``losses = self.get_train_model_output(batch)``

So every key used in ``get_train_model_output`` must be produced by your collate function.

Suggested checklist:

- Train batch should include at least:
  - ``input_ids`` (or equivalent field names)
  - ``attention_mask`` (optional but recommended)
  - ``labels`` or enough fields to construct labels
- Eval batch should include at least:
  - model inference input
  - reference answer (for accuracy)

Common mistakes:

1. ``collate_fn`` outputs ``list[dict]``, but training code treats it as ``dict``
2. Some samples miss multimodal fields, causing batch misalignment
3. Eval still uses ``drop_last=True``, dropping evaluation samples

----

Implement get_train_model_output
------------------------------------

``get_train_model_output`` returns training output. Ensure the return can be used for training.

``FSDPSftWorker`` handles your returned loss by:

- auto-normalizing list/tuple/tensor
- gradient accumulation
- scaler backward

So you only need to guarantee the final return is a loss tensor (or stackable loss list).

Recommended CausalLM style:

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

Implement get_eval_model_output
-----------------------------------

``get_eval_model_output`` returns evaluation output. Ensure it can be used for metric aggregation.

``run_eval()`` logic:

- accumulates batch return values into ``correct``
- divides by ``total`` to get ``eval_accuracy``

So your ``get_eval_model_output`` should return the number of correct samples in the current batch.

Example:

.. code:: python

   def get_eval_model_output(self, batch):
       # 1) Generate prediction
       # 2) Compare with ground truth
       # 3) Return number of correct samples
       return correct

----

YAML Configuration
------------------

Start with conservative settings:

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

After this runs successfully, gradually increase batch size and enable eval/save intervals.

----

Troubleshooting
---------------

1. ``KeyError: xxx``  
   - Your collate function did not output the fields required by training code

2. ``Expected all tensors on same device``  
   - Some batch fields were not moved via ``to(self.device)``

3. ``global_batch_size is not divisible ...``  
   - Adjust ``global_batch_size / micro_batch_size / world_size``

4. ``eval_accuracy is unexpectedly low``  
   - Check answer extraction/parsing in eval
   - Check whether ``drop_last`` is dropping eval samples

5. ``data repeats/skips after resume``  
   - Check save/load flow for ``_data_epoch`` and ``_data_iter_offset``
