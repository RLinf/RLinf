新增硬件后端
============

无需修改训练或推理 worker，即可把新的 GPU 或 NPU 后端接入 RLinf 的硬件调度层与安装脚本。完成本指南后，RLinf 会自动在每个节点发现该加速卡，按 worker 隔离可见设备，通过 ``self.torch_platform`` 暴露对应的 PyTorch 后端模块，配置集合通信，并支持使用 ``requirements/install.sh --platform`` 安装依赖。

适配工作按四个检查点推进：先实现 ``AcceleratorManager`` 子类封装设备发现、可见性、PyTorch 平台与集合通信接口，再注册 ``AcceleratorType`` 枚举、profiling 配置和模块导出以接入自动发现，随后把 ``--platform`` 接入 ``requirements/install.sh``，最后在本地 CPU 上完成契约验证再上真卡测试。集群调度与安装约定请参阅 :doc:`集群与硬件 <../concepts/cluster>` 和 :doc:`安装指南 <../start/installation>`。

接入速览
--------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 项目
     - 说明
   * - 创建或修改的文件
     - ``rlinf/scheduler/hardware/accelerators/<name>.py``、``rlinf/scheduler/hardware/accelerators/accelerator.py``、``rlinf/scheduler/hardware/accelerators/__init__.py`` 与 ``requirements/install.sh``
   * - 扩展点
     - ``AcceleratorType``、``@AcceleratorManager.register_manager``、``@AcceleratorManager.register_profiling_config``、``AcceleratorUtil.CCL_SUPPORT_LIST`` 与 ``SUPPORTED_PLATFORMS``
   * - 核心契约
     - ``AcceleratorManager`` 上的一组静态方法，负责设备数量、型号名称、可见性环境变量、集合通信后端、网卡环境变量、PyTorch 平台模块、设备类型与进程组选项
   * - 本地 CPU 验证
     - 通过 ``AcceleratorUtil`` 执行契约检查脚本

1. 实现加速卡管理器
-------------------

RLinf 将所有厂商相关的加速卡逻辑收敛在 ``rlinf/scheduler/hardware/accelerators/`` 下。worker 代码统一调用 ``self.torch_platform``\ （在非 worker 代码中使用 ``Worker.torch_platform``），替代直接调用 ``torch.cuda``。worker 启动时，``Worker._env_setup_before_init()`` 会通过 ``AcceleratorUtil.get_torch_platform()`` 和 ``AcceleratorUtil.get_device_type()`` 解析已注册后端，并调用 ``setup_worker_torch()``，从而让 FSDP、Megatron 与 rollout worker 在 NVIDIA、AMD、Ascend、MUSA、Kunlun、Biren、MetaX 与 Intel 后端上保持一致。

新建后端模块（例如 ``rlinf/scheduler/hardware/accelerators/example_accel.py``），实现对应的 ``ProfileConfig`` 子类与 ``AcceleratorManager`` 子类。厂商运行库必须在方法内部延迟导入，确保 CPU 机器及其他平台导入模块时不报错：

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

按调用顺序说明每个静态方法的职责：

- ``get_num_devices()`` 与 ``get_accelerator_model()`` 在节点硬件枚举时探测本机设备。厂商运行库缺失时返回 ``0`` 和 ``"UNKNOWN"``，避免阻断其他后端探测。
- ``get_accelerator_type()`` 返回该后端对应的 ``AcceleratorType`` 枚举值。
- ``get_accelerator_env_var(visible_accelerators)`` 在驱动进程拉起 worker 时生成设备隔离环境变量；一并返回 ``RAY_EXPERIMENTAL_NOSET_*``，防止 Ray 覆盖 RLinf 分配的设备列表。
- ``get_visible_devices()`` 在 worker 进程内解析上述环境变量，返回整数设备编号列表。
- ``get_ccl_backend()``、``get_ccl_socket_ifname_env_var()`` 与 ``get_accel_pg_options(options)`` 分别提供 ``torch.distributed`` 的集合通信后端名、通信网卡环境变量和进程组选项。
- ``get_torch_platform()`` 返回绑定到 ``self.torch_platform`` 的 PyTorch 平台模块。
- ``get_device_type()`` 返回 PyTorch 在该硬件上实际报告的 ``tensor.device.type`` 字符串（权重同步与工具函数会据此比对设备类型）。

如果平台还有额外需求，再按需覆写 ``AcceleratorManager`` 的可选钩子：在 worker 本地解析节点相关环境变量（如 ``intel_gpu.py`` 中的 EGL 或 oneCCL 拓扑）可覆写 ``get_worker_local_env_var()``；需要安装进程级 PyTorch 补丁可覆写 ``setup_worker_torch()``；接入性能分析工具则覆写 ``modify_profiling_context()``、``start_profiling()``、``stop_profiling()``、``profiling_range()`` 与 ``get_profiling_env_vars()``。简洁的内置实现可参考 ``rlinf/scheduler/hardware/accelerators/biren_supa.py`` 和 ``rlinf/scheduler/hardware/accelerators/musa_gpu.py``。

2. 注册硬件类型与设备发现
-------------------------

``Accelerator``\ （``rlinf/scheduler/hardware/accelerators/accelerator.py`` 中的默认 ``Hardware`` 策略）会遍历 ``AcceleratorManager.manager_register`` 自动完成设备发现。只需修改以下两处，就能在导入硬件包时自动完成注册：

1. 在 ``rlinf/scheduler/hardware/accelerators/accelerator.py`` 中向 ``AcceleratorType`` 添加枚举成员。若 ``MultiChannelProcessGroup`` 已支持该集合通信后端，再把同一枚举加入 ``AcceleratorUtil.CCL_SUPPORT_LIST``；否则多通道集合通信会回退到 Gloo。

.. code-block:: python

   class AcceleratorType(str, Enum):
       ...
       EXAMPLE_ACCEL = "EXAMPLE_ACCEL"

   class AcceleratorUtil:
       CCL_SUPPORT_LIST = [
           ...,
           AcceleratorType.EXAMPLE_ACCEL,
       ]

2. 在 ``rlinf/scheduler/hardware/accelerators/__init__.py`` 中导入并导出管理器，确保导入 ``rlinf.scheduler.hardware`` 时触发 ``@AcceleratorManager.register_manager`` 与 ``@AcceleratorManager.register_profiling_config``：

.. code-block:: python

   from .example_accel import ExampleAcceleratorManager

   __all__ = [
       ...,
       "ExampleAcceleratorManager",
   ]

集群启动时，``Accelerator.enumerate()`` 会依次调用已注册管理器的 ``get_num_devices()``，选中首个设备数大于 ``0`` 的后端，并生成包含 ``HardwareInfo(type="Accelerator", model=f"{accel_type.value}:{model}")`` 的 ``HardwareResource``。后续调度器、worker 与集合通信调用都会经由 ``AcceleratorUtil`` 统一分发。

.. warning::

   即便暂不接入性能分析工具，也应通过 ``@AcceleratorManager.register_profiling_config(...)`` 注册一个空的 ``ProfileConfig`` 子类。节点探测会检查 ``AcceleratorManager.profiling_config_register``，要求每个 ``AcceleratorType`` 均有对应条目。

3. 接入安装脚本的平台参数
-------------------------

如需支持通过 ``bash requirements/install.sh <target> --platform <name>`` 创建虚拟环境，在 ``requirements/install.sh`` 中补充三处钩子：

1. 将平台名加入 ``SUPPORTED_PLATFORMS``。
2. 定义 ``configure_<platform>()`` 并在 ``configure_platform()`` 中添加 ``case`` 分支。依赖厂商基础镜像的平台可直接复用 ``configure_vendor_torch_platform`` 与 ``PLATFORM_VENDOR_DISTS``；从独立 wheel 源安装则配置 ``PLATFORM_TORCH_INDEX`` 与 ``PLATFORM_TORCH_PACKAGES``。
3. 定义 ``install_<platform>_extras()`` 并在 ``install_platform_extras()`` 中添加 ``case`` 分支，在依赖同步完成后校验或安装厂商运行时包。

下面参考 ``requirements/install.sh`` 中的 ``configure_biren`` 与 ``install_biren_extras`` 给出标准写法：

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

随后在 ``configure_platform()`` 与 ``install_platform_extras()`` 中分别增加 ``example)`` 分支。新增 ``rlinf/scheduler/hardware/accelerators/`` 文件后，请确认路径已命中 ``.github/workflows/ci-tests.yml`` 的过滤规则。Docker 镜像与 CI 维护流程请参阅 ``CONTRIBUTING.md`` 与 :doc:`安装指南 <../start/installation>`。

4. 在本地 CPU 上验证
--------------------

上真实加速卡之前，先在 CPU 主机上检查注册表、环境变量往返解析与 ``AcceleratorUtil`` 分发。下面以已注册的 ``AcceleratorType.BIREN_GPU`` 为例演示无卡验证：

.. code-block:: bash

   python -c '
   import os
   import rlinf.scheduler.hardware.accelerators  # 触发管理器注册
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

预期输出：

.. code-block:: text

   backend: bccl device_type: supa env: {'SUPA_VISIBLE_DEVICES': '0,2', 'RAY_EXPERIMENTAL_NOSET_SUPA_VISIBLE_DEVICES': '1'}
