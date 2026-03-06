πRL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models
========================================================================

**Paper:** `arXiv:2510.25889 <https://arxiv.org/abs/2510.25889>`__ 

Overview
--------

.. image:: https://github.com/RLinf/misc/raw/main/pic/pi_rl_teaser.png
   :alt: πRL teaser
   :align: center

πRL provides online reinforcement learning fine-tuning for flow-based vision-language-action (VLA) models π₀ and π₀.₅ within the **RLinf** framework. By combining PPO/GRPO with flow matching policies, the method enables few-shot SFT models to achieve strong manipulation performance through environment feedback. It supports LIBERO, ManiSkill3, MetaWorld, and CALVIN benchmarks, with visual understanding, language comprehension, and continuous action generation jointly optimized via RL.

Results
-------

π₀ Model
~~~~~~~~

.. list-table:: Evaluation results of π₀ model
   :header-rows: 1
   :widths: 15 25 15 20 20
   :align: left

   * - Environment
     - Task
     - SFT
     - Flow-SDE
     - Flow-Noise
   * - LIBERO
     - Spatial, Object, Goal
     - `SFT <https://huggingface.co/RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT>`__
     - —
     - —
   * - LIBERO
     - Long
     - `SFT <https://huggingface.co/RLinf/RLinf-Pi0-LIBERO-Long-SFT>`__
     - —
     - —
   * - ManiSkill3
     - Multi-task
     - 38.4%
     - `78.8% <https://huggingface.co/RLinf/RLinf-Pi0-ManiSkill-25Main-RL-FlowSDE>`__
     - `77.8% <https://huggingface.co/RLinf/RLinf-Pi0-ManiSkill-25Main-RL-FlowNoise>`__
   * - MetaWorld
     - MT50
     - 50.8%
     - `78.1% <https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-RL-FlowSDE>`__
     - `85.8% <https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-RL-FlowNoise>`__
   * - CALVIN
     - ABC-D
     - 57.5%
     - `61.7% <https://huggingface.co/RLinf/RLinf-Pi0-CALVIN-ABC-D-RL-FlowSDE>`__
     - `59.9% <https://huggingface.co/RLinf/RLinf-Pi0-CALVIN-ABC-D-RL-FlowNoise>`__

π₀.₅ Model
~~~~~~~~~~

.. list-table:: Evaluation results of π₀.₅ model
   :header-rows: 1
   :widths: 15 30 15 20 20
   :align: left

   * - Environment
     - Task
     - SFT
     - Flow-SDE
     - Flow-Noise
   * - LIBERO
     - Spatial, Object, Goal, Long
     - `SFT <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT>`__
     - —
     - —
   * - ManiSkill3
     - Multi-task
     - 40.1%
     - `90.9% <https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowSDE>`__
     - `89.7% <https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise>`__
   * - MetaWorld
     - MT50
     - 43.8%
     - `70.7% <https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-RL-FlowSDE>`__
     - `66.1% <https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-RL-FlowNoise>`__
   * - CALVIN
     - ABC-D
     - 61.3%
     - `87.0% <https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-RL-FlowSDE>`__
     - `84.5% <https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-RL-FlowNoise>`__

Quickstart
----------

**Full guide:** :doc:`../examples/embodied/pi0`

**Run:** ``bash examples/embodiment/run_embodiment.sh <CONFIG_NAME>`` (configs in ``examples/embodiment/config/``)

**Model Selection:**

- **π₀:** Configs **without** ``_pi05`` in the name
- **π₀.₅:** Configs **with** ``_pi05`` in the name (e.g. ``*_openpi_pi05.yaml``)

**Benchmarks:**

- **LIBERO:** :doc:`../examples/embodied/libero`
- **ManiSkill3:** :doc:`../examples/embodied/maniskill`
- **MetaWorld:** :doc:`../examples/embodied/metaworld`
- **CALVIN:** :doc:`../examples/embodied/calvin`
- **Real2Sim2Real (GSEnv):** :doc:`../examples/embodied/gsenv`

Citation
--------

.. code-block:: bibtex

   @article{chen2025pi_rl,
     title={$$\backslash$pi\_$\backslash$texttt $\{$RL$\}$ $: Online RL Fine-tuning for Flow-based Vision-Language-Action Models},
     author={Chen, Kang and Liu, Zhihao and Zhang, Tonghe and Guo, Zhen and Xu, Si and Lin, Hao and Zang, Hongzhi and Li, Xiang and Zhang, Quanlu and Yu, Zhaofei and others},
     journal={arXiv preprint arXiv:2510.25889},
     year={2025}
   }
