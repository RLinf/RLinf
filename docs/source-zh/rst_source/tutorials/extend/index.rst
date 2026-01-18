扩展框架
========================

对于希望进行更深层次定制的高级用户，本章演示如何通过集成自定义环境和新的模型架构来扩展 RLinf。  

你将学习如何：

- 将一个 :doc:`新环境 <new_env>` 集成到 RLinf 的任务系统中  
- 添加一个使用 FSDP + HuggingFace 后端的 :doc:`新模型 <new_model_fsdp>`  
- 添加一个使用 Megatron + SGLang 后端的 :doc:`新模型 <new_model_megatron>`  
- 配置 :doc:`数据采集 <data_collection>` 用于离线学习和奖励模型训练  
- 集成 :doc:`奖励模型 <reward_model>` 用于基于图像的奖励学习  

RLinf 支持多种模型训练后端，每种后端都有自己的初始化逻辑和执行流程。  
本指南提供了逐步说明，帮助你完成以下任务：

- 在 RLinf 中注册并加载自定义模型  
- 配置 YAML 文件以引用你的新模型或环境  
- 如果你的模型类型尚未被支持，扩展特定后端的代码  
- 调整环境封装器和接口以集成新的模拟器或 API  
- 采集并存储交互数据用于奖励建模等下游任务  

无论你是要训练一种新的模型架构，还是要在自定义 RL 环境中进行实验，  
本节都将提供工具，帮助你直接接入 RLinf 的模块化设计。  

.. toctree::
   :hidden:
   :maxdepth: 2

   new_env
   new_model_fsdp
   new_model_megatron
   data_collection
   reward_model
