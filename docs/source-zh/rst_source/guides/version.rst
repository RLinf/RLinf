SGLang 与 vLLM 版本选择
=======================

RLinf 支持将不同的生成后端接入强化学习流水线，同时支持 **SGLang 与 vLLM**。

.. note::

   RLinf 兼容 **SGLang 0.4.4 → 0.5.13**\ （具身 ``--model sglang`` 使用 **SGLang 0.5.19**）与 **vLLM 0.8.5 → 0.23.x**，无需手动打补丁，框架会自动检测已安装的版本并加载匹配的 shim。

支持的引擎版本
-------------------------

RLinf 能安装的每个引擎版本都对应 ``requirements/agentic/`` 下的一个文件，命名为 ``<engine>_<version>_<cu12|cu13>.txt``。目录中的文件集合就是受支持的版本集合，列出该目录即可知道可以安装哪些版本。

============  ==============  =========  =======
引擎          版本            CUDA 分支  torch
============  ==============  =========  =======
SGLang        0.5.12.post1    cu12,cu13  2.11.0
SGLang        0.5.4           cu12       2.8.0
SGLang        0.5.2           cu12       2.8.0
SGLang        0.4.6.post5     cu12       2.6.0
vLLM          0.23.0          cu12,cu13  2.11.0
vLLM          0.8.5           cu12       2.6.0
============  ==============  =========  =======

默认版本是上表中最高的版本（目前为 torch 2.11 上的 SGLang 0.5.12.post1 与 vLLM 0.23.0），该默认值直接由这些文件推导得出，因此新增更高版本文件即可更新默认值。CUDA 分支跟随 torch wheel 而非驱动，没有 cu13 构建的 torch 版本即使在 CUDA 13 主机上也会留在 cu12 分支。

每个 venv 只装一个引擎
-------------------------

一个 venv 只包含一个引擎。SGLang 与 vLLM 会把同一批 kernel 库（``nvidia-cutlass-dsl``、``flashinfer-python``、``tilelang``、``tokenspeed-mla``）固定到不同版本，有时 torch 版本也不同；若共用一个 venv，后安装的引擎会静默降级先装的 kernel，直到真正调用 kernel 时才暴露。用 ``--engine`` 选择引擎；需要两者时用不同的 ``--venv`` 各装一次：

.. code-block:: bash

   bash requirements/install.sh agentic --engine sglang
   bash requirements/install.sh agentic --venv .venv-vllm --engine vllm

reason Docker 镜像同时提供两者：``reason``\ （SGLang，默认激活）与 ``reason-vllm``。

切换版本
-------------------------

把版本传给安装脚本即可，torch 会随之确定，无需另外指定：

.. code-block:: bash

   # 默认：torch 2.11 上的 SGLang 0.5.12.post1
   bash requirements/install.sh agentic

   # SGLang 0.4.x 分支，torch 2.6
   bash requirements/install.sh agentic --sglang 0.4.6.post5

   # vLLM 0.8.5，torch 2.6
   bash requirements/install.sh agentic --engine vllm --vllm 0.8.5

传入不受支持的版本会立即报错并列出实际存在的版本，而不会解析出一个不可用的环境。

.. note::

   不要用 ``pip install sglang`` / ``pip install vllm`` 装进已有环境。两者都会固定整个 torch 系列，而它们的 CUDA 13 版本会固定 CUDA 13 运行时 wheel 并就地覆盖对应的 CUDA 12 版本；这些 requirements 文件正是为了避免这种情况。

Rollout 引擎配置
----------------

有关 ``rollout.rollout_backend``\ （``sglang`` 或 ``vllm``）及后端专用配置项，见 :doc:`agentic_config`。

