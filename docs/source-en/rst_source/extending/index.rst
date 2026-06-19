Extending
=========

Use Extending when you add a model, environment, or advanced integration path
to RLinf.

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

   .. grid-item-card:: New Model with FSDP
      :link: new_model_fsdp
      :link-type: doc

      Add a HuggingFace model on the FSDP backend.

   .. grid-item-card:: New Model with Megatron
      :link: new_model_megatron
      :link-type: doc

      Add a HuggingFace model on the Megatron+SGLang backend.

   .. grid-item-card:: New SFT Model
      :link: new_model_sft
      :link-type: doc

      Plug a new model into the SFT training flow.

   .. grid-item-card:: New Real-World Robot
      :link: new_realworld_robot
      :link-type: doc

      Add support for a new physical robot, walked through end-to-end on
      the LeRobot SO101 arm.

   .. grid-item-card:: Advanced Integrations
      :link: advanced-integrations/index
      :link-type: doc

      Add Megatron-Bridge, weight synchronization, or reward model workflows.

.. toctree::
   :hidden:

   Extending Overview <overview>
   New Environment <new_env>
   New Model with FSDP <new_model_fsdp>
   New Model with Megatron <new_model_megatron>
   New SFT Model <new_model_sft>
   New Real-World Robot <new_realworld_robot>
   Advanced Integrations <advanced-integrations/index>
