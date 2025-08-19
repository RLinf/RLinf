Switch SGLang Versions
======================

RLinf can plug different *generation backends* into its
reinforcement-learning pipeline. For the current release **only
SGLang** is supported; vLLM integration is under development.

.. note::

   RLinf is compatible with **SGLang 0.4.4 → 0.4.9**.  
   No manual patching is required – the framework detects the installed
   version and loads the matching shim automatically.

Installation Requirements
-------------------------

* **CUDA** ≥ 11.8 (or 12.x matching your PyTorch build)  
* **Python** ≥ 3.8  
* Sufficient **GPU memory** for the chosen model  
* Compatible versions of **PyTorch** and *transformers*

.. note::

   Mismatched CUDA / PyTorch wheels are the most common installation
   issue.  Verify both before installing SGLang.

Install via *pip*
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Reference version
   pip install sglang==0.4.4

   # Recommended for production
   pip install sglang==0.4.8

   # Latest supported
   pip install sglang==0.4.9

Install from Source
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/sgl-project/sglang.git
   cd sglang
   git checkout v0.4.8          # pick the tag you need
   pip install -e .

.. note::

   Building from source can be time-consuming and heavy on disk space;
   prefer the pre-built wheels unless you need bleeding-edge fixes.


Selecting SGLang in the YAML
----------------------------

.. code-block:: yaml

    ....
    generation:
        # Model configuration
        model_dir: "/path/to/your/model"
        tensor_parallel_size: 4
        
        # SGLang-specific settings
        backend: "sglang"     
    ...


Internal Version Routing
------------------------

Directory layout::

   infini_rl/generation/sglang/
   ├── __init__.py               # Version detection and routing
   ├── sglang_worker.py          # Main worker implementation
   ├── sglang_0_4_4/             # SGLang 0.4.4 specific implementation
   │   ├── __init__.py
   │   ├── io_struct.py          # I/O structures for 0.4.4
   │   ├── sgl_engine.py         # Engine implementation for 0.4.4
   │   ├── sgl_scheduler.py      # Scheduler for 0.4.4
   │   └── tokenizer_manager.py  # Tokenizer management for 0.4.4
   └── sglang_0_4_x/             # Future version implementations
       └── ...

The loader in ``__init__.py`` resolves the installed package:

.. code-block:: python

   from importlib.metadata import PackageNotFoundError, version

   def get_version(pkg):
       try:
           return version(pkg)
       except PackageNotFoundError:
           return None

   package_name = "sglang"
   package_version = get_version(package_name)
   
   if package_version == "0.4.4":
       sglang_version = "0.4.4"
       from .sglang_0_4_4 import io_struct
       from .sglang_0_4_4.sgl_engine import Engine
   else:
       raise ValueError(f"sglang version {package_version} not supported")
