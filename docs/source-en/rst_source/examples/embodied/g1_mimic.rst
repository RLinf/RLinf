G1 Motion Tracking
==================

This page describes the Unitree G1 motion-tracking environment in MuJoCo
Warp: how a task runs, what the policy reads and commands, what goes into
the reward for one step, and when an episode ends. Train this environment
with SuperMimic on the :doc:`SuperMimic page <supermimic>`.

Overview
--------

The G1 follows a reference motion clip in simulation.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      Unitree G1

   .. grid-item-card:: Algorithms
      :text-align: center

      SuperMimic

   .. grid-item-card:: Tasks
      :text-align: center

      LAFAN1 tracking; standing push recovery

   .. grid-item-card:: Hardware
      :text-align: center

      NVIDIA GPU

| **You'll learn:** the task → observations and actions → the reward →
  when an episode ends.
| **Prerequisites:** installation, the clips, and the training commands are
  on the :doc:`SuperMimic page <supermimic>`.

The Task
~~~~~~~~

The environment steps at 50 Hz. Each control step runs four physics steps of
5 ms. The policy commands a target angle for each of the 29 joints, and a
PD controller tracks those targets through the four physics steps.

Motion tracking starts at one frame of the reference clip. The environment
places the robot in that pose and has it follow the motion from there. An
episode lasts at most 10 seconds, and it also ends when the clip ends. The
episode ends as a fall when the pelvis height error exceeds 0.25 m or the
tilt error exceeds 0.8. The tilt error is the difference between the z
components of gravity in the robot's pelvis frame and in the reference
pelvis frame. Reaching the end of the clip, or the 10-second limit, ends the
episode and is not a fall.

Standing push recovery is an evaluation task in this environment. The robot
holds mjlab's knees-bent standing pose, takes one horizontal push, and tries
to settle. The push speeds, the number of trials, and the settle test are
on the :doc:`evaluation page <supermimic_eval>`.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Description
   * - Observation
     - The robot's current state, plus ten reference frames starting at the
       current frame and taken every five control steps. Past the end of the
       clip, the window stays on the last frame. Training can read this state
       as generalized coordinates or as the poses and velocities of the 14
       tracked bodies. The two readings are on the
       :doc:`algorithm page <../../reference/algorithms/supermimic>`.
   * - Action
     - Target angles for the 29 joints at 50 Hz, tracked by a PD controller.
       Training adds noise with standard deviation 0.1. Evaluation applies
       the policy output as commanded.
   * - Reward
     - How closely the robot matches the reference clip on this control step.
       A perfect match with no penalty scores 0.1. The next section gives
       the composition.
   * - Prompt
     - None. This is low-dimensional whole-body control.

Reward Design
-------------

The reward on a control step compares the robot with the reference clip. It
has six tracking terms, plus penalties for a changing action and for joints
past their soft limits.

The tracking terms score the pelvis position and orientation, and the
position, orientation, linear velocity, and angular velocity of 14 tracked
bodies on the pelvis, the legs, the torso, and the arms. For the body
positions and orientations, the reference pose is moved to the robot
pelvis's current horizontal position and yaw before the comparison, so those
terms score the shape of the motion. The pelvis position and orientation are
compared with the reference pelvis directly.

The six weights are 0.5, 0.5, 1, 1, 1, and 1. A smaller error brings a term
closer to its weight. A perfect match sums to 5, and multiplying by the
0.02 s control step gives 0.1. A large change from the previous action, or a
joint past its soft limit, brings the reward below that. The formula and
the way the world model uses this reward are on the
:doc:`algorithm page <../../reference/algorithms/supermimic>`.

Installation
------------

Use the virtual environment from the :doc:`SuperMimic page <supermimic>`.
The install command and the LAFAN1 preparation are on that page.

Run It
------

The training, resume, and evaluation commands are on the
:doc:`SuperMimic page <supermimic>`. The evaluation metrics are on the
:doc:`evaluation page <supermimic_eval>`.
