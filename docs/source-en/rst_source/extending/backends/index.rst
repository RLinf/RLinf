Add Hardware & System Backends
==============================

Pick these guides when porting RLinf to a new GPU / NPU accelerator or customizing actor-to-rollout weight synchronization.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Accelerator Backend
      :link: ../new_accelerator
      :link-type: doc

      Add a new GPU or NPU accelerator backend to RLinf's hardware scheduler and installer.

   .. grid-item-card:: Weight Synchronization
      :link: ../weight_syncer
      :link-type: doc

      Customize and optimize actor-to-rollout weight transport for embodied training.

.. toctree::
   :hidden:

   Accelerator Backend <../new_accelerator>
   Weight Synchronization <../weight_syncer>
