Why RLinf
=========

RLinf is an open-source distributed reinforcement learning framework for post-training embodied, agentic, and generative foundation models. Read this page for the architecture, benchmarked throughput gains, and workload coverage behind the framework.

What Makes RLinf Unique
-----------------------

- **Macro-to-Micro Flow (M2Flow):** decouples logical RL workflow programming from physical communication and hardware scheduling (:doc:`publications/rlinf_system`).
- **Flexible execution modes:** switch between *Collocated*, *Disaggregated*, and *Hybrid* placement through YAML without rewriting training loops (:doc:`../concepts/execution_modes`).
- **Auto and dynamic scheduling:** static mode selection and online dynamic resource scaling adapt GPU allocation during training while preserving on-policy semantics (:doc:`publications/dynarl`, :doc:`../guides/dynamic_scheduling`).
- **Broad workload coverage (v0.4+):**

  - *Embodied Simulation:* standardized RL recipes across LIBERO, ManiSkill3, RoboTwin, IsaacLab, BEHAVIOR, CALVIN, MetaWorld, and Genesis with OpenVLA, OpenVLA-OFT, π₀ / π₀.₅, GR00T, LingBot-VLA, and MLP/CNN policies (:doc:`../examples/simulators_index`, :doc:`../examples/vla_wam_index`).
  - *Real-World Robots:* unified ``rlinf.robotics`` hardware abstraction and online RL / DAgger / teleoperation across Franka, Dual Franka, SO-101, Piper, Turtle2, YAM, and more (:doc:`../examples/real_world_index`).
  - *World Models & Generative RL:* world-model RL (WoVR / Wan, OpenSora, DreamZero, Cosmos3) and diffusion/video post-training (:doc:`../examples/world_model_index`).
  - *Reasoning & Agentic AI:* math reasoning, VQA, multi-turn search and tool-use agents (Search-R1, WideSeek-R1, rStar2, Coding-Online-RL, AgentLightning) (:doc:`../examples/agentic/index`).
  - *SFT & Offline RL:* supervised fine-tuning, IQL, RECAP, STEAM, and sim-real co-training (:doc:`../examples/sft_index`, :doc:`../examples/methods_index`).

- **Multi-accelerator portability:** runs across NVIDIA GPUs, AMD ROCm, Huawei Ascend, Moore Threads MUSA, Kunlun, and Biren accelerators behind one ``Worker`` abstraction (:doc:`../start/installation`).

Measured Performance
--------------------

- **End-to-end training speedup:** M2Flow with hybrid placement and fine-grained pipelining delivers **1.07×–2.43×** end-to-end throughput over comparable RL frameworks—up to **1.7×** on math reasoning against veRL and Slime and up to **2.43×** on embodied VLA training (see :doc:`publications/rlinf_system` and :doc:`blog/compare_with_verl`).
- **Dynamic online scaling:** DynaRL rebalances worker allocations within seconds to improve utilization by **20–40%** on dynamic rollout workloads while preserving on-policy behavior (see :doc:`publications/dynarl`).

Unified Backends and Algorithms
-------------------------------

- **Pluggable training and rollout engines:** pair **FSDP + Hugging Face** for rapid VLA and research iteration, or **Megatron + SGLang / vLLM** (with :doc:`../extending/mbridge`) for large-scale LLM and VLM training.
- **Built-in algorithms:** PPO, Async PPO, GRPO, DAPO, REINFORCE++, SAC, CrossQ, RLPD, and IQL behind unified advantage and loss registries (:doc:`../reference/algorithms/index`).
- **Configuration and metrics:** inspect Hydra YAML structure and metric namespaces in :doc:`../reference/training/index`.

Reproducible Recipes
--------------------

RLinf ships end-to-end configs and scripts that reproduce published numbers across embodied and reasoning benchmarks:

- **Embodied benchmarks:** verified training and evaluation recipes on **LIBERO**, **ManiSkill**, **RoboTwin**, and real-world Franka / Turtle2 setups (:doc:`../examples/index`, :doc:`../evaluations/index`).
- **Reasoning and agentic benchmarks:** reproducible **AIME24 / AIME25 / GPQA-diamond** math training with DeepSeek-R1-Distill-Qwen models plus single- and multi-agent search and coding pipelines (:doc:`../examples/agentic/index`).
