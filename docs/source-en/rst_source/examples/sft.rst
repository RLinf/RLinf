
Supervised Fine-Tuning (SFT)
============================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document explains how to run **full-parameter supervised fine-tuning (SFT)** in the RLinf framework. SFT is typically the first stage before RLHF / RLAIF: the model first imitates high-quality demonstrations, then reinforcement learning continues optimizing from that prior.

This guide covers
--------

- How to configure RLinf for general SFT
- How to start training on a single machine or multi-node cluster
- How to monitor and evaluate results


Supported datasets
------------------

RLinf supports LeRobot-format datasets. Use **config_type** to specify the dataset type.

Currently supported dataset formats:

- pi0_maniskill
- pi0_libero
- pi05_libero
- pi05_maniskill
- pi05_metaworld
- pi05_calvin
- pi05_custom

You can also customize the dataset format to train on a specific dataset. Refer to:

#. In ``examples/sft/config/custom_sft_openpi.yaml``, specify the dataset format.

   .. code:: yaml

      model:
        openpi:
          config_name: "pi0_custom"

#. In ``rlinf/models/embodiment/openpi/__init__.py``, set the dataset format to ``pi0_custom``.

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
              raw_action_is_delta=True,  # True for delta action, False for abs_action
              action_train_with_rotation_6d=False,
          ),
          pytorch_weight_path="checkpoints/torch/pi0_base",
      ),

#. In ``rlinf/models/embodiment/openpi/dataconfig/custom_dataconfig.py``, define the custom dataset config.

   .. code:: python

      class CustomDataConfig(DataConfig):
          def __init__(self, *args, **kwargs):
              super().__init__(*args, **kwargs)
              self.repo_id = "physical-intelligence/custom_dataset"
              self.base_config = DataConfig(
                  prompt_from_task=True
              )
              self.assets = AssetsConfig(assets_dir="checkpoints/torch/pi0_base/assets")
              self.raw_action_is_delta = True
              self.action_train_with_rotation_6d = False


Training config
----------------------

A full example config is in ``examples/sft/config/libero_sft_openpi.yaml``. Key fields:

.. code:: yaml

   cluster:
     num_nodes: 1                 # number of physical machines
     component_placement:         # component → GPU mapping
       actor: 0-3


Launch scripts
-------------

Start the Ray cluster first, then run the helper script:

.. code:: bash

   cd /path_to_RLinf/ray_utils
   bash start_ray.sh                 # start head + workers

   # return to repo root
   bash examples/sft/train_embodied_sft.py --config libero_sft_openpi.yaml

The same script works for general text SFT; just swap the config file.


