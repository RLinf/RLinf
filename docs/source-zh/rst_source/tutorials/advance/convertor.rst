Checkpoint 转换
=================

RLinf提供Megatron-LM和FSDP checkpoint转换脚本，支持从 ``.distcp`` 或 ``.pt`` 格式的checkpoint文件转换到 ``.safetensors`` 格式。

FSDP 转换脚本
----------------------

FSDP Checkpoint 文件结构
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FSDP checkpoint文件结构如下： 

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

RLinf在保存checkpoint时会同时保存一份完整的模型权重，比如 ``global_step_10/actor/model_state_dict/full_weigths.pt``。
该pt文件可用于评估checkpoint（设置 ``runner.ckpt_path`` 为该文件路径即可），或继续转换为safetensors格式。


distcp 格式转换到 pt 格式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如果您使用最新的代码保存的checkpoint，已经保存了 ``model_state_dict/full_weigths.pt`` 文件，可以跳过此步骤。
如果您仍想把保存的 ``.distcp`` 文件转换成 ``.pt`` 格式，或者使用老版本代码，仅保存了 ``.distcp`` 格式的文件，可使用 RLinf/toolkits/ckpt_convertor/fsdp_convertor/convert_dcp_to_pt.pt，
先将 ``.distcp`` 转换成 ``.pt`` 格式，再执行后续操作。

.. code-block:: bash

   python convert_dcp_to_pt.py [-h] --dcp_path DCP_PATH --output_path OUTPUT_PATH


其中 ``DCP_PATH`` 是包含 DCP 文件的目录，``OUTPUT_PATH`` 是保存转换后模型 State Dict 文件的路径。



pt 格式转换到 safetensors 格式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **修改config文件**

运行转换脚本前，请您修改 ``RLinf/toolkits/ckpt_convertor/fsdp_convertor/config/fsdp_model_convertor.yaml`` 文件。
请检查以下7个参数是否正确： 

    ``defaults``， ``convertor.save_path``， ``convertor.merge_lora_weighs``， ``convertor.ckpt_path``， 
    ``model.model_type``， ``model.model_path``， ``model.is_lora``

.. code-block:: yaml

    defaults:
        - model/openvla_oft@model                                    # 默认的模型参数，根据需要转换的模型类型指定对应的模型
        - override hydra/job_logging: stdout

    convertor:
        save_path: /path/to/save                                     # 转换后的文件保存路径
        merge_lora_weighs: True                                      # 是否保存merge lora后的权重，如果设置为False则仅保存Lora权重
        ckpt_path: /path/to/model_state_dict/full_weights.pt         # 完整的模型权重文件

    # Override the default values in model/openvla_oft
    model:
        model_type: "openvla_oft"                                    # 模型类型，根据需要转换的模型类型指定
        model_path: "/path/to/Openvla-oft-SFT-libero-goal-traj1/"    # 初始模型权重
        is_lora: True                                                # 是否开启Lora


2. **（Optional）新增model_save_helper**

如果您的模型有特殊的保存逻辑，请在 ``RLinf/toolkits/ckpt_convertor/fsdp_convertor/utils.py`` 文件中的 ``get_model_save_helper`` 添加对应的保存函数。

3. **运行脚本**


.. code-block:: bash

   bash convert_pt_to_hf.sh

4. **查看保存的safetensors文件**

脚本正常运行结束后，可以查看 ``convertor.save_path`` 下保存的huggingface格式文件。


Megatron 转换脚本
---------------------

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


**方式一：编辑脚本文件**

手动打开 ``mg2hf_7b.sh`` 或 ``mg2hf_1.5b.sh``，将以下变量设置为你想要的路径。

1. ``CKPT_PATH_MG`` （Megatron checkpoint路径，例如 ``results/run_name/checkpoints/global_step_xx/actor/``）， 
2. ``CKPT_PATH_HF`` （Huggingface目标路径，任意路径），以及
3. ``CKPT_PATH_ORIGINAL_HF`` （初始化训练的基模checkpoint，例如 ``/path/to/DeepSeek-R1-Distill-Qwen-1.5B``） 

**方式二：命令行参数**

更灵活的方式是直接通过命令行参数传入路径。

.. code-block:: bash

    bash mg2hf_1.5b.sh /path/to/megatron_checkpoint /target/path/to/huggingface_checkpoint /path/to/base_model_checkpoint
