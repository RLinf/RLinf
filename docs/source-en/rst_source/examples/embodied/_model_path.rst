.. This file is a reusable include, not a standalone page.
   Include right after the recipe-specific download command with
   ``.. include:: _model_path.rst``.

Set ``actor.model.model_path`` in the recipe YAML, or pass ``actor.model.model_path=/path/to/checkpoint`` on the launch command. Rollout uses the same path unless you set ``rollout.model.model_path`` separately.
