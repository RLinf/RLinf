Configure & Run
===============

Start here once your first run works: monitor training curves, resume interrupted jobs, export checkpoints, reduce GPU memory with LoRA, or train on multiple environments together.

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Logging & Visualization
      :link: ../logger
      :link-type: doc

      Stream metrics and videos to TensorBoard, Weights & Biases, or SwanLab.

   .. grid-item-card:: Resume Training
      :link: ../resume
      :link-type: doc

      Save periodic checkpoints and resume interrupted training cleanly.

   .. grid-item-card:: Checkpoint Conversion
      :link: ../convertor
      :link-type: doc

      Convert FSDP or Megatron training checkpoints into Hugging Face or evaluation formats.

   .. grid-item-card:: LoRA Fine-Tuning
      :link: ../lora
      :link-type: doc

      Train large policies with lower GPU memory using LoRA adapters.

   .. grid-item-card:: Multi-Env Training
      :link: ../hetero_envs
      :link-type: doc

      Train or evaluate one policy across multiple environment configurations in a single run.

   .. grid-item-card:: Multi-Turn Visual Agent Training (Qwen3.5)
      :link: ../qwen35_multi_turn_images
      :link-type: doc

      Enable multi-turn image alignment for Qwen3.5 visual Agent RL across SGLang rollout and Megatron-Bridge training.

.. toctree::
   :hidden:

   Logging & Visualization <../logger>
   Resume Training <../resume>
   Checkpoint Conversion <../convertor>
   LoRA Fine-Tuning <../lora>
   Multi-Env Training <../hetero_envs>
   Multi-Turn Visual Agent Training (Qwen3.5) <../qwen35_multi_turn_images>
