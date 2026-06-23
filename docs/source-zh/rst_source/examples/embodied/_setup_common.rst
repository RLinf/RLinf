.. 本文件是可复用的 include 片段，并非独立页面。
   各具身示例方案共享此片段，以避免重复"克隆仓库 + 选择安装方式"这段完全相同的样板内容。
   使用 ``.. include:: _setup_common.rst`` 引入，并在其后补充本方案特定的
   Docker 镜像标签 / ``--env`` 取值。

首先，克隆 RLinf 仓库：

.. code:: bash

   # 为提高国内下载速度，可以使用镜像：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

然后，使用下列两种方式之一准备依赖：预构建的 Docker 镜像（推荐）或自定义环境。
通用的安装流程（前置依赖、GPU 驱动、镜像内置的 ``switch_env`` 工具、镜像加速、常见问题排查）
在 :doc:`/rst_source/start/installation` 中统一说明；本方案中的命令仅在 Docker
镜像标签和 ``--env`` 取值上有所不同。
