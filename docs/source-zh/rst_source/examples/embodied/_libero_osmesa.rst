在 AMD、华为昇腾和摩尔线程上运行 LIBERO 时，使用 CPU 渲染。在已激活的模型环境中设置以下变量，再启动训练：

.. code-block:: bash

   export MUJOCO_GL=osmesa
   export PYOPENGL_PLATFORM=osmesa
   export ROBOT_PLATFORM=LIBERO

``run_embodiment.sh`` 会保留这些变量。安装脚本通过 ``requirements/sys_deps.sh`` 安装所需的 ``libosmesa6`` 系统库。

.. warning::

   直接启动 Python 时也需要设置这两个渲染变量。软件渲染会占用 CPU 资源，增加 rollout 并行度前，应按主机能力调整环境数量。
