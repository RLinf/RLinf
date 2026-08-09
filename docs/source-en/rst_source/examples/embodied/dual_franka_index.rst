Dual-Arm Franka
===============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dual-franka.jpg
   :align: center
   :width: 80%
   :alt: Dual-Arm Franka

   Dual-Franka robot platform.

This section collects RLinf workflows for dual-Franka data collection,
supervised fine-tuning, deployment, and DAgger training.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Dual-Arm Collect-SFT-Deploy (OpenPI_RLinf)
      :link: dual_franka_openpi_rlinf
      :link-type: doc

      Train and deploy Dual-Franka Pi0 and Pi0.5 policies with OpenPI_RLinf.

   .. grid-item-card:: HG-DAgger via VR
      :link: dual_franka_pico_dagger
      :link-type: doc

      Collect dual-arm PICO data and run online human-gated DAgger.

.. toctree::
   :hidden:
   :maxdepth: 1

   Dual-Arm Collect-SFT-Deploy (OpenPI_RLinf) <dual_franka_openpi_rlinf>
   HG-DAgger via VR <dual_franka_pico_dagger>
