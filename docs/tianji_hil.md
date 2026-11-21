# Tianji HIL integration

This branch carries the Tianji HIL control path used by the Stage2 30 Hz joint deployment.

The main controller is rlinf/envs/realworld/tianji/tianji_hil.py. It owns the
loopback HIL panel and the episode control state machine:

- start and saved-pose reset;
- policy pause/resume;
- H/P human takeover and handback;
- subtask selection and prompt revision;
- success/failure/end-of-episode handling;
- revision checks that invalidate stale policy chunks.

The controller does not drive the robot SDK directly. Tianji environment code
uses it as the control-plane gate and sends executable joint chunks to a
separate LoopInfra collector through the policy schedule API. The LoopInfra
checkout and its runtime configuration therefore remain separate from this
repository.

The implementation also includes the Tianji real-world environment adapter,
the optional tianji_shadow monitor transport, the joint VLA/RLT adapters, and
the deployment configuration needed by the Stage2 integration. Model weights,
machine-specific site.env values, reset-pose identifiers, logs, and recorded
episodes are intentionally excluded.

The Stage2 launcher should pin this branch (or a commit from it) together with
the matching LoopInfra tianji_hil branch. A normal deployment still needs the
external model assets and a configured LoopInfra collector.
