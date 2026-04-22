扩展框架
========================

对于希望进行更深层次定制的高级用户，本章演示如何通过集成自定义环境和新的模型架构来扩展 RLinf。  

你将学习如何：

- 将一个 :doc:`新环境 <new_env>` 集成到 RLinf 的任务系统中  
- 添加一个使用 FSDP + HuggingFace 后端的 :doc:`新模型 <new_model_fsdp>`  
- 添加一个使用 Megatron + SGLang 后端的 :doc:`新模型 <new_model_megatron>`  
- 参考一条完整的 :doc:`Reward Model 工作流 <reward_model>`
- 参考 :doc:`真机 Reward Model 数据集采集 <reward_model_realworld>`

RLinf 支持多种模型训练后端，每种后端都有自己的初始化逻辑和执行流程。  
本指南提供了逐步说明，帮助你完成以下任务：

- 在 RLinf 中注册并加载自定义模型  
- 配置 YAML 文件以引用你的新模型或环境  
- 如果你的模型类型尚未被支持，扩展特定后端的代码  
- 调整环境封装器和接口以集成新的模拟器或 API  

从其他仓库添加自定义模型
---------------------------------------

如果你的项目把 RLinf 当作依赖库使用，现在可以直接注册自定义模型，
而不需要修改 RLinf 源码。

推荐做法如下：

- 在你自己的仓库中实现模型构建函数
- 在调用 ``build_config(...)`` 或启动训练之前执行 ``register_model(...)``
- 在 YAML 配置中直接使用注册后的 ``model_type``

.. code-block:: python

  from rlinf.models import register_model

  def build_my_model(cfg, torch_dtype):
      from my_repo.models.custom_policy import CustomPolicy

      return CustomPolicy(cfg, torch_dtype)

  register_model("my_custom_model", build_my_model, category="embodied")

注册后，RLinf 会自动：

- 在配置校验阶段接受 ``model.model_type: my_custom_model``
- 在 ``get_model(cfg)`` 时路由到你注册的 builder

这也是当前推荐的扩展方式，适合在 RLinf 主仓库之外维护自定义具身模型。

无论你是要训练一种新的模型架构，还是要在自定义 RL 环境中进行实验，  
本节都将提供工具，帮助你直接接入 RLinf 的模块化设计。  

.. toctree::
   :hidden:
   :maxdepth: 2

   new_env
   new_model_fsdp
   new_model_megatron
   new_model_sft
   reward_model
   reward_model_realworld
