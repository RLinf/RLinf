Franka (Dual-Arm)
=================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dual-franka.jpg
   :align: center
   :width: 80%
   :alt: Dual-Arm Franka

   Dual-arm Franka platform for bimanual data collection, SFT, deployment, and online HG-DAgger.

Choose a dual-arm Franka guide for GELLO or PICO data collection, OpenPI π₀.₅ fine-tuning and deployment, or online HG-DAgger.

.. note::

   **Workflow:** 1. Collect demonstrations (GELLO or PICO) → 2. SFT and deploy OpenPI π₀.₅ → 3. Improve online with PICO HG-DAgger.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: OpenPI PyTorch
      :link: dual_franka_openpi_pytorch
      :link-type: doc

      Fine-tune and deploy a dual-Franka policy with OpenPI PyTorch.

   .. grid-item-card:: OpenPI JAX
      :link: dual_franka
      :link-type: doc

      Collect GELLO demonstrations, convert data, fine-tune a policy, and deploy it.

   .. grid-item-card:: PICO VR HG-DAgger
      :link: dual_franka_pico_dagger
      :link-type: doc

      Collect dual-arm PICO data and run online human-gated DAgger.

.. toctree::
   :hidden:
   :maxdepth: 1

   OpenPI PyTorch <dual_franka_openpi_pytorch>
   OpenPI JAX <dual_franka>
   PICO VR HG-DAgger <dual_franka_pico_dagger>
