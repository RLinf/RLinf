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

Prerequisites:

- Run inside the Turtle2/X2Robot ROS environment with ``rospy`` available.
- The Turtle2 SDK and follower controllers must be connected to the real robot.
- ``/follow_pos_cmd_1`` and ``/follow_pos_cmd_2`` must be owned by RLinf only;
  do not start the legacy slave executors at the same time.
- The ROS param ``/running_mode`` must exist and be set to integer mode values
  such as ``1`` for policy and ``2`` for takeover.
- The master program connects to the RLinf takeover TCP server and provides
  fresh ``MSG_POSE`` frames during takeover.

Use:

.. code-block:: bash

   bash examples/embodiment/run_realworld_eval.sh realworld_turtle2_dagger_takeover_collect_openpi

The example config enables:

- ``env.eval.use_master_takeover: true``
- ``env.eval.action_mode: absolute_pose``
- ``env.eval.override_cfg.pose_control_backend: direct``
- ``env.eval.data_collection.export_format: pickle``
- ``env.eval.data_collection.record_executed_action: true``
- ``rollout.collect_transitions: false``

The pickle episodes record the actual executed action. If direct control rejects
an action, the previous held action is recorded as ``executed_action`` while the
rejected candidate stays in metadata.

This config uses registered placeholder algorithm fields only for config
compatibility because ``runner.only_eval`` is true. Existing button-pressing SAC
examples use ``embodied_sac`` because their training scripts dispatch to the SAC
worker; this takeover example does not start SAC or online DAgger training.

Data Format
-----------

RLinf's ``CollectEpisode`` supports both ``pickle`` and LeRobot export. LeRobot
is appropriate for clean demonstration datasets that are ready for standard
training pipelines. This example intentionally uses ``pickle`` because raw
takeover collection must preserve policy segments, master takeover segments,
hold/reject/recovery metadata, and the actual ``executed_action``. If a clean
LeRobot dataset is needed later, convert the saved pickle episodes into cleaned
expert segments as a separate preprocessing step.
