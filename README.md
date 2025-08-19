<div align="center">
  <img src="docs/source/_static/svg/logo_white.svg" alt="RLinf-logo" width="600"/>
</div>


<div align="center">


<!-- <a href="TODO"><img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv"></a> -->
<a href="TODO"><img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
<a href="TODO"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&logo=readthedocs"></a>
<a href="TODO"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="Ask DeepWiki.com" style="height:20px;"></a>
<a href="TODO"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>

</div>

<h1 align="center">RLinf: Reinforcement Learning Infrastructure for Agentic AI</h1>

RLinf is a flexible and scalable open-source infrastructure designed for post-training foundation models (LLMs, VLMs, VLAs) via reinforcement learning. The 'inf' in RLinf stands for Infrastructure, highlighting its role as a robust backbone for next-generation training. It also stands for Infinite, symbolizing the system’s support for open-ended learning, continuous generalization, and limitless possibilities in intelligence development.


<div align="center">
  <img src="docs/source/_static/svg/overview.svg" alt="RLinf-overview" width="600"/>
</div>

## What's NEW!
- [2025/08] RLinf v0.1 is released! The technical report [RLinf: Reinforcement Learning Infrastructure for Agentic AI](TODO:) is also released accordingly.


## Key Features


**RLinf is unique with:**

- Embodied Agent Support
- Native adapters for VLA models: [OpenVLA](https://github.com/openvla/openvla), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [$\pi_0$](https://github.com/Physical-Intelligence/openpi).
- Plug-and-play connectors for CPU- and GPU-based simulators: [ManiSkill](https://github.com/haosulab/ManiSkill), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).
- Enabling the first RL fine-tuning of the $\pi_0$ model family with a flow-matching action expert.

**RLinf is fast with:**

- Automatic Online-Scaling Mechanism
- Creatively achieves the first GPU switching within 1 second.
- Auto-scheduling policy: automatically selects the most suitable execution mode based on the training workload.
- Improves efficiency by 20-40% while preserving the on-policy property of RL algorithms.

**RLinf is flexible and easy to use with:**

- Flexible Execution Modes

  - Task-colocated: shares all GPUs across all workers.
  - Task-disaggregated: enables fine-grained pipelining.
  - Hybrid: combines colocated and disaggregated modes—specially designed for agent training in embodied intelligence.

- Multiple Backend Integrations

  - A single unified interface drives two complementary backends, allowing seamless switching without code changes.
  - FSDP + Hugging Face: rapid adaptation to new models and algorithms, ideal for beginners and fast prototyping.
  - Megatron + SGLang: optimized for large-scale training, delivering maximum efficiency for expert users with demanding workloads.

- Built-in support for popular RL methods, including PPO, GRPO, DAPO, Reinforce++, and more.

- Support for SFT.

## Roadmap

### 1. System-Level Enhancements
- [ ] Support for domestic GPUs (China-made GPU hardware)
- [ ] Asynchronous pipeline execution
- [ ] Mixture of Experts (MoE) support

### 2. Application-Level Extensions
- [ ] Vision-Language Model (VLM) training support
- [ ] GUI/Web agent training support
- [ ] Multi-agent training support
- [ ] Integration with more embodied simulators (e.g., Meta-World, GENESIS)
- [ ] Support for more Vision-Language Agents (VLAs), such as GR00T
- [ ] World model training support
- [ ] Real-world RL deployment support


## Getting Started

The overall [**Documentation**]()

**Quickstart**

  - [Installation]()
  - [Quickstart 1: PPO Training of VLA on Maniskill3]()
  - [Quickstart 2: GRPO Training of LLMs on MATH]()
  - [PPO in RLinf]()
  - [GRPO in RLinf]()
  - [Basic YAML Configuaration]()

**Key Design**

  - [Hybrid mode with fine-grained pipelining]()
  - [Enable Online-scaling Mechanism]()
  - [Execute Auto-scheduling Policy]()

**Example Gallery**

  - [Embodied Intelligence VLA Training]()
  - [Math Reasoning Model Training]()

**Advanced Features**

  - [5D Parallelism Configuration for Megatron-LM]()
  - [LoRA Integration for efficient fine-tuning]()
  - [Switch between different versions of SGLang]()
  - [Checkpoint Resume and Recovery Support]()

**Extending The Framework:**

  - [Adding new Environments]()
  - [Adding new Models with FSDP+Huggingface backend]()
  - [Adding new Models with Megatron+SGLang backend]()

**Blogs**

  - [Compare with VeRL]()

## Experiment Result

| Type | Status |
| :---: | :---: |
| Reasoning RL-MATH | [![Build Status]()]() |
| Agentic RL-VLA | [![Build Status]()]() |

## Contribution Guidelines
We welcome contributions to RLinf. Please read [contribution guide](TODO:) before taking action.

## Citation and Acknowledgement
