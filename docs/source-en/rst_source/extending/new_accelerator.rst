Add a Hardware Backend
======================

Add a GPU or NPU backend to RLinf's hardware scheduler and installer without changing training or rollout workers. By the end of this guide, RLinf enumerates your devices on every node, isolates visible devices per worker, exposes the vendor PyTorch module through ``self.torch_platform``, configures collective communication, and installs dependencies through ``requirements/install.sh --platform``.

The port follows four checkpoints. First, implement an ``AcceleratorManager`` subclass encapsulating vendor discovery, visibility, PyTorch platform, and collective hooks. Next, register the ``AcceleratorType`` enum, profiling config, and module export so ``Accelerator`` discovers the device automatically. Then wire ``--platform`` in ``requirements/install.sh``. Finally, verify the contract locally on CPU before testing on hardware. For cluster architecture background, see :doc:`Cluster & Hardware <../concepts/cluster>` and :doc:`Installation <../start/installation>`.

At a Glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Aspect
     - Details
   * - Files to create or edit
     - ``rlinf/scheduler/hardware/accelerators/<name>.py``, ``rlinf/scheduler/hardware/accelerators/accelerator.py``, ``rlinf/scheduler/hardware/accelerators/__init__.py``, and ``requirements/install.sh``
   * - Extension points
     - ``AcceleratorType``, ``@AcceleratorManager.register_manager``, ``@AcceleratorManager.register_profiling_config``, ``AcceleratorUtil.CCL_SUPPORT_LIST``, and ``SUPPORTED_PLATFORMS``
   * - Core contract
     - Static methods on ``AcceleratorManager`` for device count, model name, visibility env vars, CCL backend, socket interface env var, PyTorch platform, device type, and process-group options
   * - Local CPU check
     - Direct contract check via ``AcceleratorUtil``

1. Implement the Accelerator Manager
------------------------------------

RLinf keeps all vendor-specific accelerator logic inside ``rlinf/scheduler/hardware/accelerators/``. Worker code calls ``self.torch_platform`` inside a ``Worker`` (and ``Worker.torch_platform`` elsewhere) instead of ``torch.cuda``. During worker startup, ``Worker._env_setup_before_init()`` resolves ``AcceleratorUtil.get_torch_platform()`` and ``AcceleratorUtil.get_device_type()`` from the registered manager and runs ``setup_worker_torch()``, keeping FSDP, Megatron, and rollout workers portable across NVIDIA, AMD, Ascend, MUSA, Kunlun, Biren, MetaX, and Intel backends.

Create a backend module (for example ``rlinf/scheduler/hardware/accelerators/example_accel.py``) with a ``ProfileConfig`` subclass and an ``AcceleratorManager`` subclass. Always lazy-import vendor packages inside methods so CPU hosts and other platforms import the module without errors:

.. code-block:: python

   import os
   from dataclasses import dataclass
   from typing import TYPE_CHECKING, Optional

   from .accelerator import AcceleratorManager, AcceleratorType, ProfileConfig

   if TYPE_CHECKING:
       from ...collective import CollectiveGroupOptions


   def _ensure_vendor_torch() -> bool:
       try:
           import torch

           if not hasattr(torch, "example"):
               import torch_example  # noqa: F401

           return hasattr(torch, "example") and torch.example.is_available()
       except Exception:
           return False


   @AcceleratorManager.register_profiling_config(AcceleratorType.EXAMPLE_ACCEL)
   @dataclass
   class ExampleProfileConfig(ProfileConfig):
       """Profiling configuration placeholder when no profiler is wired yet."""


   @AcceleratorManager.register_manager(AcceleratorType.EXAMPLE_ACCEL)
   class ExampleAcceleratorManager(AcceleratorManager):
       """Accelerator manager for Example hardware."""

       @staticmethod
       def get_num_devices() -> int:
           if not _ensure_vendor_torch():
               return 0
           import torch

           return torch.example.device_count()

       @staticmethod
       def get_accelerator_type() -> AcceleratorType:
           return AcceleratorType.EXAMPLE_ACCEL

       @staticmethod
       def get_accelerator_model() -> str:
           if not _ensure_vendor_torch():
               return "UNKNOWN"
           try:
               import torch

               return torch.example.get_device_name(0)
           except Exception:
               return "UNKNOWN"

       @staticmethod
       def get_accelerator_env_var(visible_accelerators: list[str]) -> dict[str, str]:
           return {
               "EXAMPLE_VISIBLE_DEVICES": ",".join(visible_accelerators),
               "RAY_EXPERIMENTAL_NOSET_EXAMPLE_VISIBLE_DEVICES": "1",
           }

       @staticmethod
       def get_visible_devices() -> list[int]:
           raw = os.environ.get("EXAMPLE_VISIBLE_DEVICES", "")
           if not raw:
               return []
           try:
               return [int(idx.strip()) for idx in raw.split(",")]
           except ValueError as exc:
               raise ValueError(
                   f"Invalid visible device IDs: {raw}. "
                   "Expected comma-separated integers."
               ) from exc

       @staticmethod
       def get_ccl_backend() -> str:
           return "eccl"

       @staticmethod
       def get_ccl_socket_ifname_env_var() -> str:
           return "ECCL_SOCKET_IFNAME"

       @staticmethod
       def get_torch_platform():
           _ensure_vendor_torch()
           import torch

           return torch.example

       @staticmethod
       def get_device_type() -> str:
           return "example"

       @staticmethod
       def get_accel_pg_options(options: Optional["CollectiveGroupOptions"]):
           return None

Each static method answers one scheduler or runtime query:

- ``get_num_devices()`` and ``get_accelerator_model()`` probe the local node during hardware enumeration. Return ``0`` and ``"UNKNOWN"`` when the vendor runtime is unavailable so discovery continues to other registered backends.
- ``get_accelerator_type()`` returns the enum member registered for this backend.
- ``get_accelerator_env_var(visible_accelerators)`` runs in the driver when launching workers. Include the ``RAY_EXPERIMENTAL_NOSET_*`` variable so Ray does not override RLinf's device assignment.
- ``get_visible_devices()`` reads that variable back inside the worker process as a list of integer device indices.
- ``get_ccl_backend()``, ``get_ccl_socket_ifname_env_var()``, and ``get_accel_pg_options(options)`` configure ``torch.distributed`` collective groups and optional vendor process-group options.
- ``get_torch_platform()`` returns the PyTorch backend module bound to ``self.torch_platform``.
- ``get_device_type()`` returns the exact ``tensor.device.type`` string reported by PyTorch on your hardware (RLinf compares ``tensor.device.type`` against this value during weight sync and utilities).

Override optional hooks on ``AcceleratorManager`` only when your platform needs them: ``get_worker_local_env_var()`` for node-local values resolved inside the worker (such as EGL or oneCCL topology in ``intel_gpu.py``), ``setup_worker_torch()`` for per-worker PyTorch patches, and the profiling methods (``modify_profiling_context()``, ``start_profiling()``, ``stop_profiling()``, ``profiling_range()``, ``get_profiling_env_vars()``) when adding profiler support. For shipped compact implementations, compare ``rlinf/scheduler/hardware/accelerators/biren_supa.py`` and ``rlinf/scheduler/hardware/accelerators/musa_gpu.py``.

2. Register the Hardware Type and Discovery
-------------------------------------------

``Accelerator`` (the default ``Hardware`` policy in ``rlinf/scheduler/hardware/accelerators/accelerator.py``) discovers devices automatically by iterating over ``AcceleratorManager.manager_register``. Wire your backend in two files so registration runs on package import:

1. In ``rlinf/scheduler/hardware/accelerators/accelerator.py``, add your enum member to ``AcceleratorType``. If ``MultiChannelProcessGroup`` supports your CCL backend, add the same member to ``AcceleratorUtil.CCL_SUPPORT_LIST``; otherwise multi-channel collectives fall back to Gloo.

.. code-block:: python

   class AcceleratorType(str, Enum):
       ...
       EXAMPLE_ACCEL = "EXAMPLE_ACCEL"

   class AcceleratorUtil:
       CCL_SUPPORT_LIST = [
           ...,
           AcceleratorType.EXAMPLE_ACCEL,
       ]

2. In ``rlinf/scheduler/hardware/accelerators/__init__.py``, import and export your manager so importing ``rlinf.scheduler.hardware`` executes both ``@AcceleratorManager.register_manager`` and ``@AcceleratorManager.register_profiling_config``:

.. code-block:: python

   from .example_accel import ExampleAcceleratorManager

   __all__ = [
       ...,
       "ExampleAcceleratorManager",
   ]

At cluster startup, ``Accelerator.enumerate()`` calls ``get_num_devices()`` on each registered manager. For the first backend with a non-zero count, it builds a ``HardwareResource`` whose ``HardwareInfo`` entries carry ``type="Accelerator"`` and ``model=f"{accel_type.value}:{model}"``. All downstream scheduler, worker, and collective lookups then dispatch through ``AcceleratorUtil``.

.. warning::

   Register an empty ``ProfileConfig`` subclass with ``@AcceleratorManager.register_profiling_config(...)`` even if your backend has no profiler yet. Node probes inspect ``AcceleratorManager.profiling_config_register`` and expect every ``AcceleratorType`` to be present.

3. Wire the Installer Platform Flag
-----------------------------------

To let users build environments with ``bash requirements/install.sh <target> --platform <name>``, add three hooks to ``requirements/install.sh``:

1. Append your platform name to ``SUPPORTED_PLATFORMS``.
2. Define ``configure_<platform>()`` and add a matching ``case`` branch inside ``configure_platform()``. Platforms that run inside a vendor base container reuse the image's PyTorch stack through ``configure_vendor_torch_platform`` and ``PLATFORM_VENDOR_DISTS``; wheel-index platforms set ``PLATFORM_TORCH_INDEX`` and ``PLATFORM_TORCH_PACKAGES`` instead.
3. Define ``install_<platform>_extras()`` and add a matching ``case`` branch inside ``install_platform_extras()`` to validate or install vendor runtime packages after dependency sync.

The pattern below follows ``configure_biren`` and ``install_biren_extras`` in ``requirements/install.sh``:

.. code-block:: bash

   SUPPORTED_PLATFORMS=("nvidia" "amd" "ascend" "musa" "kunlun" "biren" "metax" "example")

   configure_example() {
       configure_vendor_torch_platform example "is this an Example runtime container?"
       PLATFORM_VENDOR_DISTS=(torch torch_example torchvision torchaudio)
   }

   install_example_extras() {
       install_vendor_torch_extras torch_example \
           "Run this inside an Example runtime image or activate its vendor environment." \
           "At runtime that means no Example device is visible to the container."
   }

Then route both functions in ``configure_platform()`` and ``install_platform_extras()`` under ``example)``. When adding files under ``rlinf/scheduler/hardware/accelerators/``, verify they match the path filters in ``.github/workflows/ci-tests.yml``. For Docker image and CI contribution conventions, see ``CONTRIBUTING.md`` and :doc:`Installation <../start/installation>`.

4. Verify Locally on CPU
------------------------

Before moving to target hardware, verify registration, visibility round-trips, and ``AcceleratorUtil`` dispatch on a CPU host. The snippet below checks a registered backend (``AcceleratorType.BIREN_GPU``) without requiring any GPU:

.. code-block:: bash

   python -c '
   import os
   import rlinf.scheduler.hardware.accelerators  # triggers manager registrations
   from rlinf.scheduler.hardware.accelerators import (
       AcceleratorManager,
       AcceleratorType,
       AcceleratorUtil,
   )

   accel = AcceleratorType.BIREN_GPU
   assert accel in AcceleratorManager.manager_register
   assert accel in AcceleratorManager.profiling_config_register

   env = AcceleratorUtil.get_accelerator_env_var(accel, ["0", "2"])
   os.environ.update(env)
   assert AcceleratorUtil.get_visible_devices(accel) == [0, 2]
   print(
       "backend:", AcceleratorUtil.get_ccl_backend(accel),
       "device_type:", AcceleratorUtil.get_device_type(accel),
       "env:", env,
   )
   '

Expected output:

.. code-block:: text

   backend: bccl device_type: supa env: {'SUPA_VISIBLE_DEVICES': '0,2', 'RAY_EXPERIMENTAL_NOSET_SUPA_VISIBLE_DEVICES': '1'}
