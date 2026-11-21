Use CPU rendering for LIBERO on AMD, Huawei Ascend, and Moore Threads. Set both
variables in the active model environment before launching a run:

.. code-block:: bash

   export MUJOCO_GL=osmesa
   export PYOPENGL_PLATFORM=osmesa

``run_embodiment.sh`` preserves these values. The installer includes the
``libosmesa6`` system library through ``requirements/sys_deps.sh``.

On AMD Radeon GPUs, LIBERO can render on the GPU instead: set ``MUJOCO_GL`` and
``PYOPENGL_PLATFORM`` to ``egl``, and each env worker renders on the GPU
assigned to it. Mesa on the host, or in the container, must be recent enough to
support the GPU; an older Mesa renders on the CPU and reports no error. AMD
Instinct accelerators have no rasterization hardware, so keep ``osmesa`` on
them. GPU rendering was verified on two Radeon AI PRO R9700 GPUs with Mesa 26.

.. warning::

   Set both rendering variables when launching Python directly as well.
   Software rendering uses CPU resources; adjust environment counts to the
   host's capacity before scaling up rollouts.
