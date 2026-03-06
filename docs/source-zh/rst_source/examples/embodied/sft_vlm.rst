VLM模型监督微调训练
==========================

本文档介绍如何在 RLinf 框架中对VLM模型进行 **全量监督微调（Full-parameter SFT）**。

本教程重点需要关注两个文件：

- 启动脚本：``examples/sft/run_vlm_sft.sh``
- 训练配置：``examples/sft/config/custom_sft_vlm.yaml``

----------------------

启动脚本：``examples/sft/run_vlm_sft.sh``

- 当前脚本默认使用配置yaml文件 ``examples/sft/config/custom_sft_vlm.yaml``
- 重定向文件的输出在：``<repo>/logs/<timestamp>/``
- 实际执行命令：

.. code:: bash

   python examples/sft/train_vlm_sft.py \
     --config-path examples/sft/config/ \
     --config-name <你的配置名> \
     runner.logger.log_path=<自动生成的日志目录>

配置模板：``examples/sft/config/custom_sft_vlm.yaml``

 VLM 配置与 RLinf 中的其他 RL 训练文件结构基本一样，其中 ``data`` 和 ``actor.model`` 的具体值改为 VLM 场景。

具体的运行流程开始前准备
------------------------

1. 准备好环境，下载 RLinf 官方镜像 ``rlinf/rlinf:math-rlinf0.1-torch2.6.0-sglang0.4.6.post5-vllm0.8.5-megatron0.13.0-te2.1``
2. 准备好模型权重目录，下载网址 ``https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct``
3. 准备好 Robo2VLM 数据集目录 ``https://huggingface.co/datasets/keplerccc/Robo2VLM-1``
4. 修改 ``examples/sft/config/custom_sft_vlm.yaml`` 文件，运行脚本 ``examples/sft/run_vlm_sft.sh``

下面是实例 yaml 文件

请注意，Robo2VLM数据集下载后由于它将 train 数据和 evaluate 数据放在一起，命名方式为 ``train-00000-of-00262.parquet`` 和 ``test-0000X-of-00003.parquet``，所以需要将它们分开，并分别放在不同的文件夹下，否则 RLinf 会直接读取整个数据集。

测试中所有需要修改的值已经在下面样例中注释，其他在实验过程中使用的参数和如下 yaml 保持一致。

.. code:: yaml

   defaults:
     - override hydra/job_logging: stdout

   hydra:
     run:
       dir: .
     output_subdir: null

   cluster:
     num_nodes: 1
     component_placement:
       actor: all

   runner:
     task_type: sft
     logger:
       log_path: "../results"
       project_name: rlinf
       experiment_name: "qwen2_5_vl_sft_demo"
       logger_backends: ["tensorboard"]

     max_epochs: 6000
     max_steps: -1
     val_check_interval: 1000
     save_interval: 1000

   data:
     type: vlm
     dataset_name: "robo2vlmsft"

     # 数据路径，需要将 train 数据和 evaluate 数据分开，并分别放在不同的文件夹下
     train_data_paths: "/path/to/Robo2VLM-1/train_data"
     # 如果不需要进行训练，只需要进行评估，请对将 train_data_paths 设置为 null
     eval_data_paths: "/path/to/Robo2VLM-1/test_data"

     # 数据字段名（要和你的数据列一致）
     prompt_key: "question"
     choice_key: "choices"
     answer_key: "correct_answer"
     image_keys: ["image"]

     apply_chat_template: True
     use_chat_template: True
     max_prompt_length: 1024
     lazy_loading: false
     num_workers: 4

   algorithm:
     adv_type: gae

   actor:
     group_name: "ActorGroup"
     training_backend: "fsdp"
     micro_batch_size: 4
     eval_batch_size: 4
     global_batch_size: 256
     seed: 42

     model:
       model_type: "qwen2.5_vl"
       precision: fp32
       # 模型路径，需要将模型权重下载后放在本地，并设置为模型路径
       model_path: "/path/to/Qwen2.5-VL-3B-Instruct"
       is_lora: False

     optim:
       lr: 1e-5
       adam_beta1: 0.9
       adam_beta2: 0.999
       adam_eps: 1.0e-08
       weight_decay: 0.01
       clip_grad: 1.0
       lr_scheduler: "cosine"
       total_training_steps: ${runner.max_epochs}
       lr_warmup_steps: 200

     fsdp_config:
       strategy: "fsdp"
       sharding_strategy: "no_shard"
       use_orig_params: False
       gradient_checkpointing: False
       mixed_precision:
         param_dtype: bf16
         reduce_dtype: fp32
         buffer_dtype: bf16

   reward:
     use_reward_model: False

   critic:
     use_critic_model: False

----

启动训练
----------------------------

在仓库根目录执行：

.. code:: bash

   bash examples/sft/run_vlm_sft.sh

说明：

- 不传参数时，脚本默认 ``custom_sft_vlm``
- 如果你文件名不同，比如 ``my_vlm_config.yaml``，就传参数：

.. code:: bash

   bash examples/sft/run_vlm_sft.sh my_vlm_config

----

怎么看训练是否正常
--------------------------

1. 终端日志里看 loss 是否下降
2. 查看脚本输出的日志目录（脚本会自动创建 ``logs/时间戳``）
3. 用 TensorBoard 可视化：

.. code:: bash

   tensorboard --logdir /path/to/RLinf/logs --port 6006

浏览器打开：``http://localhost:6006``

----

评估模型
----------

如果你只想跑 evaluate，把配置改成：

- ``data.train_data_paths: null``
- ``data.eval_data_paths: "/eval_data_path"``

其余启动命令不变，仍用：

.. code:: bash

   bash examples/sft/run_vlm_sft.sh <配置名>

----

实验结果：
我们提供当前已经实验好的结果，当前实验在单台 8 x H100 Nvidia GPU 的机器上测试 6000 次迭代，实验结果如下：

6000 次迭代， 每 1000 iter 对 test_data 的评估结果：

.. image:: https://github.com/RLinf/misc/raw/main/pic/sft_vlm_eval_accuracy.png
   :alt: VLM SFT eval accuracy
   :width: 85%
   :align: center

grad_norm 曲线：

.. image:: https://github.com/RLinf/misc/raw/main/pic/sft_vlm_eval_grad_norm.png
   :alt: VLM SFT grad norm
   :width: 85%
   :align: center

loss 曲线：

.. image:: https://github.com/RLinf/misc/raw/main/pic/sft_vlm_eval_loss.png
   :alt: VLM SFT loss
   :width: 85%
   :align: center

最后一次 eval 的 accuracy 为 ``0.8995802998542786`` (约 ``89.96%``)。

模型 checkpoint 说明
----------------------------

当前 SFT 使用 FSDP 训练后保存的是 FSDP 权重（例如 ``full_weights.pt``）。
如果需要转成 HuggingFace 权重，建议使用仓库内置脚本：

- 脚本：``toolkits/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.sh``
- 配置：``toolkits/ckpt_convertor/fsdp_convertor/config/fsdp_model_convertor.yaml``

先修改配置中的以下字段：

- ``convertor.ckpt_path``：指向 ``full_weights.pt``
- ``convertor.save_path``：输出 HF 权重目录
- ``model.model_path``：原始基座模型路径
- ``model.model_type``：对应模型类型（如 qwen2.5_vl）

运行命令：

.. code:: bash

   bash toolkits/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.sh

字段解释
-------------------

- ``micro_batch_size``：单卡一次前向/反向的样本数
- ``global_batch_size``：全局 batch（需满足可整除关系）
- ``max_epochs``：按数据集完整遍历的轮数
- ``save_interval``：每多少 step 存一次 checkpoint
- ``model_path``：本地模型目录（必须存在）
- ``train_data_paths/eval_data_paths``：数据目录或文件路径

----

最常见报错与排查
----------------

1. **找不到模型路径**
   - 检查 ``actor.model.model_path`` 是否正确、是否有读取权限

2. **数据字段不匹配**
   - 检查 ``prompt_key/choice_key/answer_key/image_keys`` 是否和数据实际列名一致

3. **显存不足（OOM）**
   - 先把 ``micro_batch_size`` 降低
   - 再减少 ``num_workers``
   - 必要时缩小模型或降低输入长度

4. **只想先跑通流程**
   - 用很小的数据子集
   - ``max_epochs`` 设为 1
   - ``save_interval`` 设小一点方便观察
