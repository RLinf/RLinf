Supervised Fine-Tuning
=======================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This page explains how to run **full-parameter supervised fine-tuning (SFT)** and **LoRA fine-tuning** with the RLinf framework. SFT is typically the first stage before reinforcement learning: the model imitates high-quality examples so RL can continue optimization with a strong prior.

Contents
----------

- How to configure full-parameter SFT and LoRA SFT in RLinf
- How to launch training on a single machine or multi-node cluster
- How to monitor and evaluate results


Supported datasets
--------------------

RLinf currently supports datasets in the LeRobot format, selected via **config_type**.

Supported formats include:

- pi0_maniskill
- pi0_libero
- pi05_libero
- pi05_maniskill
- pi05_metaworld
- pi05_calvin

You can also train with a custom dataset format. Refer to the files below:

1. In ``examples/sft/config/custom_sft_openpi.yaml``, set the data format.

.. code:: yaml

  model:
    openpi:
      config_name: "pi0_custom"

2. In ``rlinf/models/embodiment/openpi/__init__.py``, set the data format to ``pi0_custom``.

.. code:: python

    TrainConfig(
        name="pi0_custom",
        model=pi0_config.Pi0Config(),
        data=CustomDataConfig(
            repo_id="physical-intelligence/custom_dataset",
            base_config=DataConfig(
                prompt_from_task=True
            ),  # we need language instruction
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_base/assets"),
            extra_delta_transform=True,  # True for delta action, False for abs_action
            action_train_with_rotation_6d=False,  # User can add extra config in custom dataset
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
    ),

3. In ``rlinf/models/embodiment/openpi/dataconfig/custom_dataconfig.py``, define the custom dataset config.

.. code:: python

    class CustomDataConfig(DataConfig):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.repo_id = "physical-intelligence/custom_dataset"
            self.base_config = DataConfig(
                prompt_from_task=True
            )
            self.assets = AssetsConfig(assets_dir="checkpoints/torch/pi0_base/assets")
            self.extra_delta_transform = True
            self.action_train_with_rotation_6d = False


Training configuration
----------------------

A full example lives in ``examples/sft/config/libero_sft_openpi.yaml``. Key fields:

.. code:: yaml

    cluster:
        num_nodes: 1                 # number of nodes
        component_placement:         # component → GPU mapping
            actor: 0-3

To enable LoRA fine-tuning, set ``actor.model.is_lora`` to True and configure ``actor.model.lora_rank``.

.. code:: yaml

    actor:
        model:
            is_lora: True
            lora_rank: 32

Dependency Installation
-----------------------

This section describes the dependency for the SFT of OpenPI model. 
For other models, please refer to the ``Dependency Installation`` section of the corresponding examples.

**Option 1: Docker Image**

Use the Docker image ``rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0`` for the experiment.

Please switch to the corresponding virtual environment via the built-in `switch_env` utility in the image:

.. code:: bash

   source switch_env openpi

**Option 2: Custom Environment**

Install dependencies directly in your environment by running the following command:

.. code:: bash

   pip install uv
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

Launch scripts
----------------

First start the Ray cluster, then run the helper script:

.. code:: bash

   export RANK=0  # set the rank of the current node
   cd /path_to_RLinf/ray_utils
   bash start_ray.sh

   # return to repo root
   bash examples/sft/train_embodied_sft.sh --config libero_sft_openpi

The same script works for generic text SFT; just swap the config file.