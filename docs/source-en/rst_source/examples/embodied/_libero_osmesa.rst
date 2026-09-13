Use CPU rendering for LIBERO on AMD, Huawei Ascend, and Moore Threads. Set both
variables in the active model environment before launching a run:

.. code-block:: bash

   export MUJOCO_GL=osmesa
   export PYOPENGL_PLATFORM=osmesa
   export ROBOT_PLATFORM=LIBERO

``run_embodiment.sh`` preserves these values. The installer includes the
``libosmesa6`` system library through ``requirements/sys_deps.sh``.

.. warning::

   Set both rendering variables when launching Python directly as well.
   Software rendering uses CPU resources; adjust environment counts to the
   host's capacity before scaling up rollouts.
