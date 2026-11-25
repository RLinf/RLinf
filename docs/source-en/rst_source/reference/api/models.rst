Model Interface
===============

Look up RLinf's model registry functions and the ``BasePolicy`` contract shared by embodied policies. To register a new model, see :doc:`../../extending/new_model_fsdp`.

Model Registry
--------------

.. autofunction:: rlinf.models.register_model

.. autofunction:: rlinf.models.get_model

.. autofunction:: rlinf.models.resolve_model_path

.. autofunction:: rlinf.models.is_hf_repo_id

.. autofunction:: rlinf.models.merge_lora

Embodied Policy Contract
------------------------

.. autoclass:: rlinf.models.embodiment.base_policy.ForwardType
   :members:
   :undoc-members:
   :member-order: bysource

.. autoclass:: rlinf.models.embodiment.base_policy.BasePolicy
   :members: forward, default_forward, predict_action_batch, predict_value_batch, supports_value_only_bootstrap, refl_forward, enable_torch_compile, capture_cuda_graph, release_cuda_graph, is_cuda_graph_enabled
   :member-order: bysource
