Extending the Framework
========================

For advanced users seeking deeper customization, this chapter demonstrates how to extend RLinf  
by integrating custom environments and new model architectures.

You will learn how to:

- Integrate a :doc:`new environment <new_env>` into RLinf's task system  
- Add a :doc:`new model <new_model_fsdp>` using the FSDP + HuggingFace backend  
- Add a :doc:`new model <new_model_megatron>` using the Megatron + SGLang backend  
- Set up :doc:`data collection <data_collection>` for offline learning and reward model training  
- Integrate a :doc:`reward model <reward_model>` for image-based reward learning  

RLinf supports multiple backends for model training, each with its own initialization logic and execution flow.  
This guide provides step-by-step instructions on how to:

- Register and load custom models in RLinf  
- Configure YAML files to reference your new model or environment  
- Extend backend-specific code if your model type is not yet supported  
- Adapt environment wrappers and interfaces to integrate new simulators or APIs  
- Collect and store interaction data for downstream tasks like reward modeling

Whether you're training a novel model architecture or experimenting with a custom RL environment,  
this section gives you the tools to plug directly into RLinf’s modular design.

.. toctree::
   :hidden:
   :maxdepth: 2

   new_env
   new_model_fsdp
   new_model_megatron
   data_collection
   reward_model
