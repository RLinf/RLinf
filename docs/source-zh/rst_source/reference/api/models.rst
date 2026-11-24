Model 接口
==========

查阅 RLinf 的模型注册函数与具身 policy 共用的 ``BasePolicy`` 接口。接入新模型的完整步骤见 :doc:`../../extending/new_model_fsdp`。

模型注册表
----------

.. autofunction:: rlinf.models.register_model

.. autofunction:: rlinf.models.get_model

.. autofunction:: rlinf.models.resolve_model_path

.. autofunction:: rlinf.models.is_hf_repo_id

.. autofunction:: rlinf.models.merge_lora

具身 Policy 接口
----------------

.. autoclass:: rlinf.models.embodiment.base_policy.ForwardType
   :members:
   :undoc-members:
   :member-order: bysource

.. autoclass:: rlinf.models.embodiment.base_policy.BasePolicy
   :members: forward, default_forward, predict_action_batch, predict_value_batch, supports_value_only_bootstrap, refl_forward, enable_torch_compile, capture_cuda_graph, release_cuda_graph, is_cuda_graph_enabled
   :member-order: bysource
