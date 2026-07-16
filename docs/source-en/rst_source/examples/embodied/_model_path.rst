.. This file is a reusable include, not a standalone page.
   It holds the identical "point your config at the downloaded checkpoint" tail
   shared by the embodied recipes. Include it right after the recipe-specific
   download commands with ``.. include:: _model_path.rst``.

After downloading, point your config YAML at the checkpoint — set the **same** path
for both the rollout and the actor model:

.. code:: yaml

   rollout:
      model:
         model_path: /path/to/downloaded-checkpoint
   actor:
      model:
         model_path: /path/to/downloaded-checkpoint
