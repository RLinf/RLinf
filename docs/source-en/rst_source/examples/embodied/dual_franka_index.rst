Dual-Arm Franka
===============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dual-franka.jpg
   :align: center
   :width: 80%
   :alt: Dual-Arm Franka

   Dual-Franka robot platform.

This section collects RLinf workflows for dual-Franka data collection,
supervised fine-tuning, deployment, and DAgger training.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Collection / SFT / Deployment
      :link: dual_franka
      :link-type: doc

      Collect GELLO demonstrations, convert data, fine-tune a policy, and deploy it.

   .. grid-item-card:: VR HG-DAgger
      :link: dual_franka_pico_dagger
      :link-type: doc

      Collect dual-arm PICO data and run online human-gated DAgger.

   .. grid-item-card:: Collection / SFT / Deployment (OpenPI PyTorch)
      :link: dual_franka_openpi_pytorch
      :link-type: doc

      Fine-tune and deploy a dual-Franka policy with OpenPI PyTorch.

.. toctree::
   :hidden:
   :maxdepth: 1

   Collection / SFT / Deployment <dual_franka>
   VR HG-DAgger <dual_franka_pico_dagger>
   Collection / SFT / Deployment (OpenPI PyTorch) <dual_franka_openpi_pytorch>
