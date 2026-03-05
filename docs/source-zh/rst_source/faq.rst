常见问题
========

下面整理了 RLinf 的常见问题。该部分会持续更新，欢迎大家不断提问，帮助我们改进！

------------------------------------

RuntimeError: The MUJOCO_EGL_DEVICE_ID environment variable must be an integer between 0 and 0 (inclusive), got 1.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：** 运行设置了 MUJOCO_GL 环境变量为 "egl" 的模拟器时出现上述错误信息。

**原因：** 该错误是因为您的 GPU 环境未正确设置图形渲染，尤其是在 NVIDIA GPU 上。

**修复：** 检查您是否有此文件 `/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0`。

1. 如果您有此文件，请检查您是否还拥有 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`。如果没有，请创建此文件并添加以下内容：

   .. code-block:: json

      {
         "file_format_version" : "1.0.0",
         "ICD" : {
            "library_path" : "libEGL_nvidia.so.0"
         }
      }

   然后在您的运行脚本中添加以下环境变量：

   .. code-block:: shell

      export NVIDIA_DRIVER_CAPABILITIES="all"

2. 如果您没有此文件，则表示您的 NVIDIA 驱动程序未正确安装图形功能。您可以尝试以下解决方案：

   * 重新安装 NVIDIA 驱动程序，并使用正确的选项启用图形功能。安装 NVIDIA 驱动程序时，有几个选项会禁用图形驱动程序。因此，您需要尝试安装NVIDIA的图形驱动。在Ubuntu上可以通过命令 ``apt install libnvidia-gl-<driver-version>`` 完成，具体参见NVIDIA的文档 https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html#compute-only-headless-and-desktop-only-no-compute-installation 。

   * 使用 **osmesa** 进行渲染，将运行脚本中的 `MUJOCO_GL` 和 `PYOPENGL_PLATFORM` 环境变量更改为 "osmesa"。但是，这可能会导致滚动过程比 EGL 慢 10 倍，因为它使用 CPU 进行渲染。



------------------------------------

任务迁移时出现 NCCL “cuda invalid argument”
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：** P2P 任务传输失败，报错 ``NCCL cuda invalid argument``。

**修复：** 若此机器上之前运行过任务，请先停止 Ray 并重新启动。

.. code-block:: bash

   ray stop

------------------------------------

SGLang 加载参数时出现 NCCL “cuda invalid argument”
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：** SGLang 在加载权重时报 ``NCCL cuda invalid argument``。

**原因：** Placement 不匹配。例如配置使用 *共享式（collocated）*，但训练（trainer）与生成（generation）实际跑在不同 GPU 上。

**修复：** 检查 Placement 策略。确保训练组与生成组按照 ``cluster.component_placement`` 指定的 GPU 放置。

------------------------------------

torch_memory_saver.cpp 中 CUDA CUresult Error（result=2）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：**
``CUresult error result=2 file=csrc/torch_memory_saver.cpp func=cu_mem_create line=103``

**原因：** SGLang 恢复缓存缓冲区时可用显存不足；常见于在更新前没有卸载推理权重的情况。

**修复：**

- 降低 SGLang 的静态显存占用（例如调低 ``static_mem_fraction``）。
- 确保在重新加载前，已正确释放推理权重。

------------------------------------

Gloo 超时 / “Global rank x is not part of group”
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：**

- ``RuntimeError: [../third_party/gloo/.../unbound_buffer.cc:81] Timed out waiting ... for recv``
- ``ValueError: Global rank xxx is not part of group``

**可能原因：** 之前的 SGLang 故障（见上面的 CUresult 错误）导致生成阶段未完成，Megatron 随后一直等待，直到 Gloo 超时。

**修复：**

1. 在日志中定位上一阶段的 SGLang 错误。  
2. 先解决 SGLang 的恢复/显存问题。  
3. 重新启动作业（必要时也重启 Ray）。

------------------------------------

数值精度 / 推理后端
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**提示：** SGLang 默认使用 **flashinfer** 作为注意力实现。若需更高稳定性或兼容性，可尝试 **triton**：

.. code-block:: yaml

   rollout:
     attention_backend: triton

------------------------------------

无法连接 GCS（ip:port）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：** Worker 节点无法连接到给定地址上的 Ray head（GCS）。

**原因：** 在 0 号节点上通过以下命令获取 head 节点 IP：

.. code-block:: bash

   hostname -I | awk '{print $1}'

若该命令选择了其他节点不可达的网卡（如网卡顺序不一致；可达的是 ``eth0``，却选中了别的接口），Worker 将连接失败。

**修复：**

- 确认所选 IP 能被其他节点访问（例如使用 ping 测试）。  
- 如有需要，请显式选择正确网卡对应的 IP 作为 Ray head，并将该 IP 告知各 Worker。
