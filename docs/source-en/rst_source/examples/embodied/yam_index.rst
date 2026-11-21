YAM
===

Choose a guide for dual-arm YAM data collection, OpenPI π₀.₅ fine-tuning, and real-world evaluation.

.. note::

   **Workflow:** 1. Collect demonstrations with PICO → 2. Train and evaluate OpenPI π₀.₅ on both arms.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: 1. PICO Data Collection
      :link: yam
      :link-type: doc

      Collect 14-D absolute-position demonstrations with PICO controllers and three RGB cameras.

   .. grid-item-card:: 2. OpenPI π₀.₅ Training and Evaluation
      :link: yam_openpi
      :link-type: doc

      Fine-tune OpenPI π₀.₅ on YAM demonstrations and evaluate the checkpoint on both arms.

.. toctree::
   :hidden:
   :maxdepth: 1

   PICO Data Collection <yam>
   Pi0.5 Training and Evaluation <yam_openpi>
