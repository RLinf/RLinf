Checkpoint Convertor
=====================

RLinf provides Megatron-LM and FSDP checkpoint convertor scripts, supporting converting from ``.distcp`` or ``.pt`` format checkpoint files to ``.safetensors`` format.

FSDP Convertor Script
----------------------

FSDP Checkpoint File Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The FSDP checkpoint file structure is as follows:

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
   │           └── full_weigths.pt
   └── global_step_20/
       └── …

RLinf saves a complete copy of model weights when saving checkpoints, such as ``global_step_10/actor/model_state_dict/full_weigths.pt``.
This pt file can be used to evaluate checkpoints (set ``runner.ckpt_path`` to this file path), or continue converting to safetensors format.


Converting distcp Format to pt Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are using the latest code to save checkpoints, the ``model_state_dict/full_weigths.pt`` file has already been saved, and you can skip this step.
If you still want to convert the saved ``.distcp`` files to ``.pt`` format, or if you are using older code that only saved ``.distcp`` format files, you can use RLinf/toolkits/ckpt_convertor/fsdp_convertor/convert_dcp_to_pt.pt
to first convert ``.distcp`` to ``.pt`` format, then proceed with subsequent operations.

.. code-block:: bash

   python convert_dcp_to_pt.py [-h] --dcp_path DCP_PATH --output_path OUTPUT_PATH


Where ``DCP_PATH`` is the directory containing DCP files, and ``OUTPUT_PATH`` is the path to save the converted model State Dict file.



Converting pt Format to safetensors Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Modify the config file**

Before running the convertor script, please modify the ``RLinf/toolkits/ckpt_convertor/fsdp_convertor/config/fsdp_model_convertor.yaml`` file.
Please check that the following 7 parameters are correct:

    ``defaults``, ``convertor.save_path``, ``convertor.merge_lora_weighs``, ``convertor.ckpt_path``, 
    ``model.model_type``, ``model.model_path``, ``model.is_lora``

.. code-block:: yaml

    defaults:
        - model/openvla_oft@model                                    # Default model parameters, specify the corresponding model according to the model type to be converted
        - override hydra/job_logging: stdout

    convertor:
        save_path: /path/to/save                                     # Path to save converted files
        merge_lora_weighs: True                                      # Whether to save merged LoRA weights, if set to False, only LoRA weights will be saved
        ckpt_path: /path/to/model_state_dict/full_weights.pt         # Complete model weights file

    # Override the default values in model/openvla_oft
    model:
        model_type: "openvla_oft"                                    # Model type, specify according to the model type to be converted
        model_path: "/path/to/Openvla-oft-SFT-libero-goal-traj1/"    # Initial model weights
        is_lora: True                                                # Whether LoRA is enabled


2. **（Optional）Add model_save_helper**

If your model has special saving logic, please add the corresponding save function in the ``get_model_save_helper`` in the ``RLinf/toolkits/ckpt_convertor/fsdp_convertor/utils.py`` file.

3. **Run the script**


.. code-block:: bash

   bash convert_pt_to_hf.sh

4. **View saved safetensors files**

After the script runs successfully, you can view the HuggingFace format files saved under ``convertor.save_path``.


Megatron Convertor Script
--------------------------

The Megatron checkpoint file structure is as follows:

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


**Method 1: Edit the script file**

Manually open ``mg2hf_7b.sh`` or ``mg2hf_1.5b.sh``, and set the following variables to your desired paths.

1. ``CKPT_PATH_MG`` (Megatron checkpoint path, e.g., ``results/run_name/checkpoints/global_step_xx/actor/``),
2. ``CKPT_PATH_HF`` (HuggingFace target path, any path), and
3. ``CKPT_PATH_ORIGINAL_HF`` (Base model checkpoint for initial training, e.g., ``/path/to/DeepSeek-R1-Distill-Qwen-1.5B``)

**Method 2: Command-line arguments**

A more flexible way is to pass paths directly through command-line arguments.

.. code-block:: bash

    bash mg2hf_1.5b.sh /path/to/megatron_checkpoint /target/path/to/huggingface_checkpoint /path/to/base_model_checkpoint

