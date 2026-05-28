训练可视化
======================

RLinf 支持实时实验追踪。  
你可以将损失曲线、准确率、GPU 利用率以及任意自定义指标，  
流式传输到以下一个或多个后端：

- `TensorBoard <https://www.tensorflow.org/tensorboard>`_:  
  一个广泛使用的开源可视化工具  
  （来自 TensorFlow，同时兼容 PyTorch、Hugging Face 等），  
  可以追踪损失和准确率等指标，  
  并可视化模型计算图、embedding、图像等。  

- `Weights & Biases (W&B) <https://wandb.ai/site/>`_:  
  一个基于 SaaS 的平台，提供实验追踪、超参数搜索、  
  artifacts（用于模型与数据的版本管理）、  
  报告与团队协作功能。  

- `SwanLab <https://pypi.org/project/swanlab/>`_:  
  一个开源、轻量级的实验日志与可视化工具，  
  适用于本地或自建环境。  
  它提供直观的 Python API，记录指标、超参数、硬件与代码信息，  
  并通过简洁的界面支持实验对比 —— 非常适合注重隐私的工作流。  

启用后端
-------------------

在 YAML 中将所需的 logger 添加到 ``runner.logger.logger_backends`` 中：

.. code-block:: yaml

   runner:
     task_type: math
     logger:
       log_path: ${runner.output_dir}/${runner.experiment_name}
       project_name: rlinf
       experiment_name: ${runner.experiment_name}
       logger_backends: ["tensorboard", "wandb", "swanlab"]   # <─ 选择任意子集
     experiment_name: grpo-1.5b
     output_dir: ./logs

RLinf 会为每个启用的后端创建一个子目录：

.. code-block:: text

   logs/grpo-1.5b/
   ├── checkpoints/
   ├── converted_ckpts/
   ├── log/                
   ├── swanlab/            # SwanLab 事件文件
   ├── tensorboard/        # TensorBoard 事件文件
   └── wandb/              # WandB 运行目录


TensorBoard
-----------

.. code-block:: bash

   tensorboard --logdir ./logs/grpo-1.5b/tensorboard --port 6006

在浏览器中打开 `http://localhost:6006`  
即可查看标量曲线、直方图和计算图。  


Weights & Biases (WandB)
------------------------

#. 在 `wandb.ai <https://wandb.ai>`__ 创建一个免费账户并复制你的 *API key*。  
#. 在每台机器上认证一次：  

.. code-block:: bash

    wandb login          # 按提示粘贴 API key

之后 RLinf 会自动启动一个新的 *run* 并流式传输所有指标。  
你可以通过 dashboard 查看这些指标。  


SwanLab
-------

#. 在 `swanlab.ai <https://swanlab.ai>`__ 注册并获取 *access token*。  
#. 认证：  

.. code-block:: bash

    swanlab login        # 按提示粘贴 access token

之后 RLinf 会自动启动一个新的 *run* 并流式传输所有指标。  
你可以通过 dashboard 查看这些指标。  


.. tip::

   三个 logger 可以 **并行运行**；你可以自由组合使用。
