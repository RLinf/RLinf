扩展
====

请选择与改动范围相匹配的扩展点。如果 RLinf 已支持目标真机，只需修改奖励、复位流程或成功条件，请新增真机任务；如果需要接入传感器、执行器或整台机器人，请扩展 robotics。接入新硬件时，应先在本机验证单个零部件，再配置远程部署。

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: 扩展概览
      :link: overview
      :link-type: doc

      了解各类扩展涉及的模块及其组合关系。

   .. grid-item-card:: 新环境
      :link: new_env
      :link-type: doc

      实现一个 RL 环境，并加入环境注册表。

   .. grid-item-card:: 新增真机任务
      :link: new_task
      :link-type: doc

      在 RLinf 已支持的真机上新增一个任务。

   .. grid-item-card:: 新机器人
      :link: new_robot
      :link-type: doc

      从本地零部件开始，完成机器人组合和远程部署。

   .. grid-item-card:: FSDP 新模型
      :link: new_model_fsdp
      :link-type: doc

      在 FSDP 后端上添加 HuggingFace 模型。

   .. grid-item-card:: Megatron 新模型
      :link: new_model_megatron
      :link-type: doc

      在 Megatron+SGLang 后端上添加 HuggingFace 模型。

   .. grid-item-card:: 新 SFT 模型
      :link: new_model_sft
      :link-type: doc

      将新模型接入 SFT 训练流程。

   .. grid-item-card:: 高级集成
      :link: advanced-integrations/index
      :link-type: doc

      添加 Megatron-Bridge 与权重同步工作流。

   .. grid-item-card:: SGLang 具身模型
      :link: sglang_embodied_model
      :link-type: doc

      使用 SGLang 后端接入具身模型，并在模拟器中评测。

.. toctree::
   :hidden:

   扩展概览 <overview>
   新环境 <new_env>
   新增真机任务 <new_task>
   新机器人 <new_robot>
   FSDP 新模型 <new_model_fsdp>
   Megatron 新模型 <new_model_megatron>
   新 SFT 模型 <new_model_sft>
   高级集成 <advanced-integrations/index>
   SGLang 具身模型 <sglang_embodied_model>
