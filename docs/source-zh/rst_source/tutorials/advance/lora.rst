LoRA 集成
===================

LoRA（Low-Rank Adaptation，低秩适配）是一种参数高效的微调方法，  
它通过在现有层中注入小的低秩矩阵，实现对大模型的高效适配，  
同时减少内存和训练成本，而不改变原始权重。  
本指南将介绍如何在 RLinf 中使用 LoRA。

配置
-------------

可以在 YAML 配置文件的 actor 模型部分启用 LoRA：

.. code:: yaml

  actor:
    model:
      is_lora: True
      lora_rank: 32
      lora_path: null  # 或已有 LoRA 权重的路径

**参数说明：**

- ``is_lora``：是否启用 LoRA 微调（True/False）  
- ``lora_rank``：LoRA 矩阵的秩（通常为 8–64），LoRA 会为每一层训练两个矩阵 A 和 B，  
  其形状分别为 [input-dim, lora-rank] 和 [lora-rank, output-dim]  
- ``lora_path``：预训练 LoRA 权重的路径（为 null 表示新训练）  

目标模块
---------------

RLinf 会自动将 LoRA 应用于以下模块：

.. code:: python

  target_modules = [
      "proj",      # 通用投影层
      "qkv",       # Query-Key-Value 投影
      "fc1",       # 前馈层
      "fc2",       # 视觉相关层
      "q",         # Query 投影
      "kv",        # Key-Value 投影
      "fc3",       # 额外投影层
      "q_proj",    # Query 投影
      "k_proj",    # Key 投影
      "v_proj",    # Value 投影
      "o_proj",    # 输出投影
      "gate_proj", # 门控投影（用于 SwiGLU）
      "up_proj",   # 上投影
      "down_proj", # 下投影
      "lm_head",   # 语言模型输出头
  ]

新建 LoRA 训练
~~~~~~~~~~~~~~~~~

若要从头开始使用 LoRA 训练：

.. code:: yaml

  actor:
    model:
      is_lora: True
      lora_rank: 32
      lora_path: null

**流程：**

1. 加载基础模型  
2. 应用指定秩的 LoRA 配置  
3. 使用高斯分布初始化 LoRA 权重  
4. 仅训练 LoRA 参数  

加载预训练的 LoRA
~~~~~~~~~~~~~~~~~~~~~~~~~

若要在已有 LoRA 权重基础上继续训练：

.. code:: yaml

  actor:
    model:
      is_lora: True
      lora_rank: 32
      lora_path: "/path/to/pretrained/lora/weights"

**流程：**

1. 加载基础模型  
2. 从指定路径加载预训练 LoRA 权重  
3. 将 LoRA 参数设为可训练  
4. 继续训练  

全模型微调
~~~~~~~~~~~~~~~~~~~~~~

若要关闭 LoRA，使用全模型微调：

.. code:: yaml

  actor:
    model:
      is_lora: False
      lora_rank: 32  # 当 is_lora=False 时会被忽略

**流程：**

1. 加载基础模型  
2. 将所有参数设为可训练  
3. 训练整个模型  
