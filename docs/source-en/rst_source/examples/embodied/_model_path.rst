.. This file is a reusable include, not a standalone page.
   Include right after the recipe-specific download command with
   ``.. include:: _model_path.rst``.

Set ``actor.model.model_path`` in the recipe YAML (rollout inherits the same path automatically), or pass ``actor.model.model_path=/path/to/downloaded-checkpoint`` on the command line:

.. code-block:: yaml

   actor:
     model:
       model_path: /path/to/downloaded-checkpoint
