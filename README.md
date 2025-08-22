<div align="center">
  <img src="docs/source/_static/svg/logo_white.svg" alt="RLinf-logo" width="600"/>
</div>


<div align="center">


<!-- <a href="TODO"><img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv"></a> -->
<a href="https://huggingface.co/RLinf"><img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
<a href="TODO:"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&logo=readthedocs"></a>
<a href="TODO:"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="Ask DeepWiki.com" style="height:20px;"></a>
<a href="TODO:"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>

</div>

<h1 align="center">RLinf: Reinforcement Learning Infrastructure for Agentic AI</h1>

RLinf is a flexible and scalable open-source infrastructure designed for post-training foundation models via reinforcement learning. The 'inf' in RLinf stands for Infrastructure, highlighting its role as a robust backbone for next-generation training. It also stands for Infinite, symbolizing the system’s support for open-ended learning, continuous generalization, and limitless possibilities in intelligence development.


<div align="center">
  <img src="docs/source/_static/svg/overview.svg" alt="RLinf-overview" width="600"/>
</div>

## What's NEW!
- [2025/08] RLinf v0.1 is released! The technical report [RLinf: Reinforcement Learning Infrastructure for Agentic AI](TODO:) is also released accordingly.


## Key Features


**RLinf is unique with:**

- Embodied Agent Support
- Fast adaptation support for mainstream VLA models: [OpenVLA](https://github.com/openvla/openvla), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [π₀](https://github.com/Physical-Intelligence/openpi).
- Support for mainstream CPU & GPU-based simulators via standardized RL interfaces: [ManiSkill3](https://github.com/haosulab/ManiSkill), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).
- Enabling the first RL fine-tuning of the $\pi_0$ model family with a flow-matching action expert.

**RLinf is fast with:**

- Automatic online-scaling mechanism: GPU switching within 1 second.
- Auto-scheduling Strategy: automatically selects the most suitable execution mode based on the training workload.
- Improves efficiency by 20-40% while preserving the on-policy property of RL algorithms.

**RLinf is flexible and easy to use with:**

- Flexible Execution Modes

  - Collocated mode: shares all GPUs across all workers.
  - Disaggregated mode: enables fine-grained pipelining.
  - Hybrid mode: combines collocated and disaggregated modes—specially designed for agent training in embodied intelligence.

- Multiple Backend Integrations

  - A single unified interface drives two complementary backends, allowing seamless switching without code changes.
  - FSDP + Hugging Face: rapid adaptation to new models and algorithms, ideal for beginners and fast prototyping.
  - Megatron + SGLang: optimized for large-scale training, delivering maximum efficiency for expert users with demanding workloads.

- Built-in support for popular RL methods, including PPO, GRPO, DAPO, Reinforce++, and more.

- Support for SFT.

## Roadmap

### 1. System-Level Enhancements
- [ ] Support for heterogeneous GPUs  
- [ ] Support for asynchronous pipeline execution  
- [ ] Support for Mixture of Experts (MoE)  

### 2. Application-Level Extensions
- [ ] Support for Vision-Language Model (VLM) training  
- [ ] Support for GUI/Web agent training  
- [ ] Support for multi-agent training  
- [ ] Support for integration with more embodied simulators (e.g., Meta-World, GENESIS)  
- [ ] Support for more Vision-Language Agents (VLAs), such as GR00T  
- [ ] Support for world model training  
- [ ] Support for real-world RL deployment  


## Getting Started 

TODO: fill

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

  - [Comparison with VeRL]()

## Experiment Result

| Type | Status |
| :---: | :---: |
| Reasoning RL-MATH | [![Build Status]()]() |
| Agentic RL-VLA | [![Build Status]()]() |

## Contribution Guidelines
We welcome contributions to RLinf. Please read [contribution guide](TODO:) before taking action.

## Citation and Acknowledgement

TODO: Chinese version