Extending
=========

Choose the smallest extension point that matches your change. If RLinf already
connects the physical hardware, add a real-world task without touching robotics.
If you need a new sensor, actuator, or robot, start with one local part and add
placement later. For model work, continue from FSDP to Megatron, SGLang, and
SFT, then use Advanced Integrations for backend integration and weight transport.

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Extending Overview
      :link: overview
      :link-type: doc

      Where each extension point lives and how the pieces fit together.

   .. grid-item-card:: New Environment
      :link: new_env
      :link-type: doc

      Add a new RL environment and wire it into the env registry.

   .. grid-item-card:: New Real-World Tasks
      :link: new_task
      :link-type: doc

      Add a task on physical hardware RLinf already supports.

   .. grid-item-card:: New Robot
      :link: new_robot
      :link-type: doc

      Add one local part, compose the robot, then place it remotely.

   .. grid-item-card:: New Model with FSDP
      :link: new_model_fsdp
      :link-type: doc

      Add a HuggingFace model on the FSDP backend.

   .. grid-item-card:: New Model with Megatron
      :link: new_model_megatron
      :link-type: doc

      Add a HuggingFace model on the Megatron+SGLang backend.

   .. grid-item-card:: SGLang Embodied Model
      :link: sglang_embodied_model
      :link-type: doc

      Adapt the embodied model to the RLinf rollout worker using the sglang backend, and use various simulators to evaluate the model.

   .. grid-item-card:: New SFT Model
      :link: new_model_sft
      :link-type: doc

      Plug a new model into the SFT training flow.

   .. grid-item-card:: Advanced Integrations
      :link: advanced-integrations/index
      :link-type: doc

      Add Megatron-Bridge and weight synchronization workflows.

.. toctree::
   :hidden:

   Extending Overview <overview>
   New Environment <new_env>
   New Real-World Tasks <new_task>
   New Robot <new_robot>
   New Model with FSDP <new_model_fsdp>
   New Model with Megatron <new_model_megatron>
   SGLang Embodied Model <sglang_embodied_model>
   New SFT Model <new_model_sft>
   Advanced Integrations <advanced-integrations/index>
