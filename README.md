<div align="center">
  <img src="docs/source/_static/svg/logo_white.svg" alt="RLinf-logo" width="600"/>
</div>


<div align="center">


<!-- <a href="TODO"><img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv"></a> -->
<a href="https://huggingface.co/RLinf"><img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
<a href="TODO:"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&logo=readthedocs"></a>
<a href="https://deepwiki.com/RLinf/RLinf"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="Ask DeepWiki.com" style="height:20px;"></a>
<a href="https://github.com/user-attachments/assets/e4443113-73e5-4b28-aaa7-7af61172eddd"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>

</div>

<h1 align="center">RLinf: Reinforcement Learning Infrastructure for Agentic AI</h1>

RLinf is a flexible and scalable open-source infrastructure designed for post-training foundation models via reinforcement learning. The 'inf' in RLinf stands for Infrastructure, highlighting its role as a robust backbone for next-generation training. It also stands for Infinite, symbolizing the system’s support for open-ended learning, continuous generalization, and limitless possibilities in intelligence development.


<div align="center">
  <img src="docs/source/_static/svg/overview.svg" alt="RLinf-overview"/>
</div>

## What's NEW!
- [2025/08] RLinf is open-sourced. The formal v0.1 will be released soon. The technical report [RLinf: Reinforcement Learning Infrastructure for Agentic AI]() will also be released accordingly. 

## Key Features


**RLinf is unique with:**
- Model-to-Model Flow: a new paradigm M2Flow, which executes macro-level logical flows through micro-level execution flows, decoupling logical communication (programmable) from physical communication (fine-grained and schedulable).

- Embodied Agent Support
  - Fast adaptation support for mainstream VLA models: [OpenVLA](https://github.com/openvla/openvla), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [π₀](https://github.com/Physical-Intelligence/openpi).
  - Support for mainstream CPU & GPU-based simulators via standardized RL interfaces: [ManiSkill3](https://github.com/haosulab/ManiSkill), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).
  - Enabling the first RL fine-tuning of the $\pi_0$ model family with a flow-matching action expert.

**RLinf is fast with:**

- Online Scaling Strategy: dynamically scales training resources, with GPU switching performed within seconds.
- Auto-scheduling Strategy: automatically selects the most suitable execution mode based on the training workload.
- Improves efficiency by 20-40% while preserving the on-policy property of RL algorithms.

**RLinf is flexible and easy to use with:**

- Flexible Execution Modes

  - Collocated mode: shares all GPUs across all workers.
  - Disaggregated mode: enables fine-grained pipelining.
  - Hybrid mode: combines collocated and disaggregated modes—specially designed for VLA training in embodied intelligence.

- Multiple Backend Integrations

  - FSDP + Hugging Face: rapid adaptation to new models and algorithms, ideal for beginners and fast prototyping.
  - Megatron + SGLang: optimized for large-scale training, delivering maximum efficiency for expert users with demanding workloads.

- Adaptive communication via the asynchronous communication channel

- Built-in support for popular RL methods, including [PPO](https://arxiv.org/abs/1707.06347), [GRPO](https://arxiv.org/abs/2402.03300), [DAPO](https://arxiv.org/abs/2503.14476), [Reinforce++](https://arxiv.org/abs/2501.03262), and more.


## Roadmap

### 1. System-Level Enhancements
- [ ] Support for heterogeneous GPUs  
- [ ] Support for asynchronous pipeline execution  
- [ ] Support for Mixture of Experts (MoE)  

### 2. Application-Level Extensions
- [ ] Support for Vision-Language Model (VLM) training  
- [ ] Support for deep searcher agent training  
- [ ] Support for multi-agent training  
- [ ] Support for integration with more embodied simulators (e.g., [Meta-World](https://github.com/Farama-Foundation/Metaworld), [GENESIS](https://github.com/Genesis-Embodied-AI/Genesis))  
- [ ] Support for more Vision-Language Agents (VLAs), such as [GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- [ ] Support for world model   
- [ ] Support for real-world RL embodied intelligence


## Getting Started 

Complete documentation for RLinf can be found [**Here**](https://github.com/RLinf/Documentation).

**Quickstart**

  - [Installation]()
  - [Quickstart 1: PPO Training of VLAs on Maniskill3]()
  - [Quickstart 2: GRPO Training of LLMs on MATH]()
  - [Multi-node Training]()
  - [Model Evaluation]()

**Key Design**
  - [Unified User Interface Usage]()
  - [Flexible Execution Modes]()
  - [Enable Automatic Scheduling]()
  - [Elastic Communication]()

**Example Gallery**

  - [Embodied Intelligence Vision-Language-Action Model training]()
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

  - [Comparison with VeRL]()

## Build Status

| Type | Status |
| :---: | :---: |
| Reasoning RL-MATH | [![Build Status]()]() |
| Embodied RL-VLA | [![Build Status]()]() |

## Contribution Guidelines
We welcome contributions to RLinf. Please read [contribution guide](TODO:) before taking action.

## Citation and Acknowledgement
