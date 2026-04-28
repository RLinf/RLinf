Turtle2 DAgger Takeover Collection
===================================

This example runs Turtle2 as a raw data-collection tool with master takeover.
It is not an online DAgger training recipe. The policy and the master both use
the same 14D absolute pose + gripper action contract, and the robot executes
through the Turtle2 direct pose backend.

Control Path
------------

The runtime path is:

.. code-block:: text

   OpenPI policy action chunk
      -> DualAbsolutePoseActionWrapper
      -> MasterTakeoverIntervention
      -> Turtle2 direct pose backend
      -> CollectEpisode pickle export

When takeover mode is inactive, the wrapper passes the policy action through.
When takeover mode is active and a fresh master pose is available, the wrapper
uses the master pose as the executed action. Rejected or invalid takeover poses
are recorded as metadata and are not marked as expert labels.

Takeover Protocol
-----------------

The env-side TCP server keeps the X2Robot takeover handshake:

- ``MSG_MODE`` communicates normal/takeover mode.
- ``MSG_JOINT`` sends the current slave joint snapshot for master alignment.
- ``MSG_POSE`` carries the master 14D absolute pose command.

Joint snapshots are kept for synchronization only. Joint command takeover is
outside this example and should use a separate data contract if added later.

Configuration
-------------

Use:

.. code-block:: bash

   bash examples/embodiment/run_realworld_eval.sh realworld_turtle2_dagger_takeover_collect_openpi

The example config enables:

- ``env.eval.use_master_takeover: true``
- ``env.eval.action_mode: absolute_pose``
- ``env.eval.override_cfg.pose_control_backend: direct``
- ``env.eval.data_collection.export_format: pickle``
- ``rollout.collect_transitions: false``

The pickle episodes record the actual executed action. If direct control rejects
an action, the previous held action is recorded as ``executed_action`` while the
rejected candidate stays in metadata.
