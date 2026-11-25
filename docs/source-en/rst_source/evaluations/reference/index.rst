Eval Reference
==============

Pick this section to look up evaluation YAML fields, CLI usage, supported models, and output paths.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: CLI
      :link: cli
      :link-type: doc

      ``evaluations/run_eval.sh`` syntax, benchmark prefix inference, config fallback, and Hydra overrides.

   .. grid-item-card:: Configuration
      :link: configuration
      :link-type: doc

      Hydra YAML structure, ``env.eval`` fields, and how to derive eval configs from training YAMLs.

   .. grid-item-card:: Supported Models
      :link: models
      :link-type: doc

      Models and inference backends with ready-to-run configs under ``evaluations/``.

   .. grid-item-card:: Logs & Results
      :link: results
      :link-type: doc

      Log directories, key terminal metrics, rollout video paths, and TensorBoard output.

.. toctree::
   :hidden:
   :maxdepth: 1

   CLI <cli>
   Configuration <configuration>
   Supported Models <models>
   Logs & Results <results>
