Why RLinf
=========

RLinf is built to post-train foundation models with reinforcement learning at
scale. This page collects the design ideas, performance numbers, and SOTA
recipes behind the framework — read it when you want the *why* rather than the
*how*.

What Makes RLinf Unique
-----------------------

- **Macro-to-Micro Flow (M2Flow):** a new paradigm that executes macro-level
  logical flows through micro-level execution flows, decoupling logical workflow
  construction (programmable) from physical communication and scheduling
  (efficient).

- **Flexible execution modes:**

  - *Collocated* — share all GPUs across all workers.
  - *Disaggregated* — enable fine-grained pipelining.
  - *Hybrid* — a customizable combination of collocated and disaggregated modes.

- **Auto scheduling:**

  - *Dynamic* — schedule resource allocation on the fly to maximize utilization.
  - *Static* — automatically pick the best execution mode for the workload, with
    no manual resource allocation.

- **Embodied agent support:**

  - Fast adaptation for mainstream VLA models: `OpenVLA`_, `OpenVLA-OFT`_, `π₀`_, `GR00T-N1.5`_.
  - Mainstream CPU & GPU simulators via standardized RL interfaces: `ManiSkill3`_, `LIBERO`_, `IsaacLab`_.
  - The first RL fine-tuning of the π₀ model family with a flow-matching action expert.

Performance
-----------

- **Hybrid mode with fine-grained pipelining** achieves a **120%+** throughput
  improvement over comparable frameworks.
- **Automatic online scaling** scales training resources dynamically, completing
  GPU switching within seconds and improving efficiency by a further **20–40%**
  while preserving the on-policy nature of RL algorithms.

Flexible and Easy to Use
------------------------

- **Multiple backends** behind one interface — switch without code changes:

  - *FSDP + Hugging Face* — rapid adaptation to new models and algorithms; ideal
    for beginners and fast prototyping.
  - *Megatron + SGLang* — optimized for large-scale training and maximum
    efficiency for demanding workloads.

- **Adaptive communication** via the asynchronous communication channel.
- **Built-in RL methods**, including `PPO`_, `GRPO`_, `DAPO`_, `Reinforce++`_, and more.

SOTA RL Training Reproduction
-----------------------------

RLinf provides end-to-end recipes that reproduce or match **state-of-the-art
(SOTA) RL results** out of the box — run our configs and scripts directly to
obtain published numbers without custom engineering.

- **Embodied tasks:** RLinf reaches or matches SOTA success rates on benchmarks
  such as **LIBERO**, **ManiSkill**, and **RoboTwin** with OpenVLA, OpenVLA-OFT,
  π₀/π₀.₅, GR00T, and other VLAs. See the :doc:`../examples/index` gallery and the
  :doc:`../reference/index` algorithm specs.
- **Agentic tasks (including math reasoning):** RLinf achieves SOTA performance on
  **AIME24 / AIME25 / GPQA-diamond** with DeepSeek-R1-Distill-Qwen models, and
  supports single- and multi-agent tasks such as Search-R1 and Coding-Online-RL.
  See :doc:`../examples/agentic/index`.

.. _PPO: https://arxiv.org/abs/1707.06347
.. _GRPO: https://arxiv.org/abs/2402.03300
.. _DAPO: https://arxiv.org/abs/2503.14476
.. _Reinforce++: https://arxiv.org/abs/2501.03262
.. _OpenVLA: https://github.com/openvla/openvla
.. _OpenVLA-OFT: https://github.com/moojink/openvla-oft
.. _ManiSkill3: https://github.com/haosulab/ManiSkill
.. _LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO
.. _IsaacLab: https://github.com/isaac-sim/IsaacLab
.. _π₀: https://github.com/Physical-Intelligence/openpi
.. _GR00T-N1.5: https://github.com/NVIDIA/Isaac-GR00T.git
