Extending the Framework
========================

For advanced users seeking deeper customization, this chapter demonstrates how to extend RLinf  
by integrating custom environments and new model architectures.

You will learn how to:

- Integrate a :doc:`new environment <new_env>` into RLinf’s task system  
- Add a :doc:`new model <new_model_fsdp>` using the FSDP + HuggingFace backend  
- Add a :doc:`new model <new_model_megatron>` using the Megatron + SGLang backend  
- Follow a complete :doc:`reward model workflow <reward_model>`
- Collect a :doc:`real-world reward model dataset <reward_model_realworld>`

RLinf supports multiple backends for model training, each with its own initialization logic and execution flow.  
This guide provides step-by-step instructions on how to:

- Register and load custom models in RLinf  
- Configure YAML files to reference your new model or environment  
- Extend backend-specific code if your model type is not yet supported  
- Adapt environment wrappers and interfaces to integrate new simulators or APIs

Adding a Custom Model from Another Repo
---------------------------------------

If your project depends on RLinf as a library, you can now register a custom model
without modifying RLinf source code directly.

The recommended pattern is:

- Implement your model builder in your own repo
- Call ``register_model(...)`` before ``build_config(...)`` or training starts
- Reference the registered ``model_type`` in your YAML config

.. code-block:: python

  from rlinf.models import register_model

  def build_my_model(cfg, torch_dtype):
      from my_repo.models.custom_policy import CustomPolicy

      return CustomPolicy(cfg, torch_dtype)

  register_model("my_custom_model", build_my_model, category="embodied")

After registration, RLinf will:

- accept ``model.model_type: my_custom_model`` during config validation
- route ``get_model(cfg)`` to your registered builder

This is the preferred extension path for custom embodied models maintained outside
the main RLinf repository.

Whether you're training a novel model architecture or experimenting with a custom RL environment,  
this section gives you the tools to plug directly into RLinf’s modular design.

.. toctree::
   :hidden:
   :maxdepth: 2

   new_env
   new_model_fsdp
   new_model_megatron
   new_model_sft
   reward_model
   reward_model_realworld
