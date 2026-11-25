在 AMD、华为昇腾和摩尔线程上运行 LIBERO 时，使用 CPU 渲染。在已激活的模型环境中设置以下变量，再启动训练：

.. code-block:: bash

   export MUJOCO_GL=osmesa
   export PYOPENGL_PLATFORM=osmesa

``run_embodiment.sh`` 会保留这些变量。安装脚本通过 ``requirements/sys_deps.sh`` 安装所需的 ``libosmesa6`` 系统库。

在 AMD Radeon GPU 上，LIBERO 可以改用 GPU 渲染：把 ``MUJOCO_GL`` 和 ``PYOPENGL_PLATFORM`` 都设为 ``egl``，各 env worker 就在各自分配到的 GPU 上渲染。宿主机或容器里的 Mesa 版本必须支持所用的 GPU；版本过旧时，渲染会退回 CPU，并且不报错。AMD Instinct 加速卡没有光栅化硬件，仍使用 ``osmesa``。GPU 渲染已在两张 Radeon AI PRO R9700 上验证，Mesa 版本为 26。

.. warning::

   直接启动 Python 时也需要设置这两个渲染变量。软件渲染会占用 CPU 资源，增加 rollout 并行度前，应按主机能力调整环境数量。
