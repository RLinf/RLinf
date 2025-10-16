<div align="center">
  <img src="docs/source-en/_static/svg/logo_white.svg" alt="RLinf-logo" width="600"/>
</div>

<div align="center">
<a href="https://arxiv.org/abs/2509.15965"><img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv"></a>
<a href="https://huggingface.co/RLinf"><img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
<a href="https://rlinf.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&logo=readthedocs"></a>
<a href="https://rlinf.readthedocs.io/zh-cn/latest/"><img src="https://img.shields.io/badge/中文文档-red?logo=readthedocs"></a>
<a href="https://deepwiki.com/RLinf/RLinf"><img src="https://img.shields.io/badge/Ask%20DeepWiki-1DA1F2?logo=databricks&logoColor=white&color=00ADEF" alt="Ask DeepWiki"></a>
<a href="https://github.com/RLinf/misc/blob/main/pic/wechat.jpg?raw=true"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>
</div>

<div align="center">

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![简体中文](https://img.shields.io/badge/语言-简体中文-red.svg)](README.zh-CN.md)

</div>

<h1 align="center">
  <sub>RLinf: 为Agentic AI而生的强化学习框架</sub>
</h1>

RLinf 是一个灵活且可扩展的开源框架，专为利用强化学习进行基础模型的后训练而设计。名称中的 “inf” 既代表 `Infrastructure`，强调其作为新一代训练坚实基础的作用；也代表 `Infinite`，寓意其支持开放式学习、持续泛化以及智能发展的无限可能。

<div align="center">
  <img src="docs/source-en/_static/svg/overview.svg" alt="RLinf-overview"/>
</div>


## 最新动态
- [2025/10] Pi0的强化学习微调已经上线! 文档：[π₀模型强化学习训练](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/pi0.html)
- [2025/10] RLinf 正式支持在线强化学习！文档：[coding_online_rl](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/coding_online_rl.html)，同时发布文章 [《首个开源的Agent在线强化学习框架RLinf-Online！让你的Agent今天比昨天更聪明》](https://mp.weixin.qq.com/s/jmohmDokuWLhQHFueSHZIQ)。
- [2025/10] RLinf算法技术报告 [《RLinf-VLA：一个统一且高效的VLA+RL训练框架》](https://arxiv.org/abs/2510.06710) 已正式发布。
- [2025/09] <img src="https://github.githubassets.com/images/icons/emoji/unicode/1f525.png" width="18" /> [示例库](https://rlinf.readthedocs.io/en/latest/rst_source/examples/index.html) 已更新，用户可以在其中找到多种可直接使用的示例！
- [2025/09] 我们的论文 [《RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation》](https://arxiv.org/abs/2509.15965)已正式发布。
- [2025/09] 机器之心关于 RLinf 的报道[《首个为具身智能而生的大规模强化学习框架RLinf！清华、北京中关村学院、无问芯穹等重磅开源》](https://mp.weixin.qq.com/s/Xtv4gDu3lhDDGadLrzt6Aw)已经发布。
- [2025/08] RLinf 已经开源，正式的 v0.1 版本即将发布。


## ✨ 核心特性


**RLinf 的独特之处在于：**
- 宏工作流到微执行流的映射机制（Macro-to-Micro Flow）：一种全新的 M2Flow 范式，通过微观层次的执行流来驱动宏观层次的逻辑流，实现逻辑工作流构建（可编程）与物理通信和调度（高效性）的解耦。

- 灵活的执行模式

  - 共享式（Collocated Mode）：用户可以配置组件是否同时常驻于 GPU 内存，或通过卸载 / 重新加载机制交替使用 GPU。
  - 分离式（Disaggregated Mode）：组件既可以顺序运行（可能导致 GPU 空闲），也可以以流水线方式执行，从而确保所有 GPU 都处于忙碌状态。
  - 混合式（Hybrid Mode）：进一步扩展了灵活性，支持自定义组合不同的放置形式。典型案例是 Generator 和 GPU-based Simulator 执行分离式细粒度流水，二者与 Inference 和 Trainer 执行共享式。

- 自动调度策略： 根据训练任务自动选择最合适的执行模式，无需手动分配资源。
  
- 具身智能体支持
  - 主流 VLA 模型的快速自适应支持: [OpenVLA](https://github.com/openvla/openvla), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), [π₀](https://github.com/Physical-Intelligence/openpi) 和 [π₀.₅](https://github.com/Physical-Intelligence/openpi).
  - 支持主流基于 CPU 与 GPU 的模拟器（通过标准化 RL 接口）： [ManiSkill3](https://github.com/haosulab/ManiSkill), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).
  - 首次实现对带有 flow-matching action expert 的 $\pi_0$ 和 $\pi_{0.5}$ 模型家族的 RL 微调。

**RLinf 的高效性体现在：**

- 细粒度流水化的混合式模式： 相较于其他框架，实现了 120%+ 的吞吐量提升。
- 秒级显卡自动扩缩： 可动态扩展训练资源，支持在数秒内完成 GPU 切换，在保持 RL 算法 on-policy 特性的同时，进一步提升 20–40% 的效率。

**RLinf 的灵活性与易用性体现在：**

- 多后端集成

  - FSDP + Hugging Face： 快速适配新模型与新算法，非常适合初学者和快速原型开发。
  - Megatron + SGLang： 针对大规模训练进行了优化，为专家用户提供最大化效率。

- 自适应通信： 通过异步通信通道实现高效交互。

- 内置支持主流 RL 方法： 包括 [PPO](https://arxiv.org/abs/1707.06347), [GRPO](https://arxiv.org/abs/2402.03300), [DAPO](https://arxiv.org/abs/2503.14476), [Reinforce++](https://arxiv.org/abs/2501.03262) 等。

## 主要成果
### 具身智能


<div align="center">
<table border="0">
  <tr>
    <td align="center">
      <img src="https://github.com/RLinf/misc/raw/main/pic/mani_openvla.png" alt="mani_openvla" width="350"/>
      <br/>
      <strong>OpenVLA</strong>
    </td>
    <td align="center">
      <img src="https://github.com/RLinf/misc/raw/main/pic/mani_openvlaoft.png" alt="mani_openvlaoft" width="350"/>
      <br/>
      <strong>OpenVLA-OFT</strong>
    </td>
  </tr>
</table>
</div>

- 在 ManiSkill 环境 “PutOnPlateInScene25Mani-v3” 上，使用 OpenVLA 与 OpenVLA-OFT 模型进行训练。结果显示，在 PPO 与 GRPO 算法的对比中，PPO 始终表现优于 GRPO，且训练过程更加稳定。

<div align="center">
<table style="text-align:center;">
  <tr>
    <th colspan="6" style="text-align:center;"> <strong>在 ManiSkill 上的评测结果。表中数值表示任务的成功率（Success Rate）</strong></th>
  </tr>
  <tr>
    <td style="text-align:center;"></td>
    <th rowspan="2" colspan="1" style="text-align:center;">In-Distribution</th>
    <td colspan="4" style="text-align:center;"><strong>Out-Of-Distribution<strong></td>
  
  </tr>
  <tr>
    <th style="text-align:center;"></th>
    <th style="text-align:center;">Vision</th>
    <th style="text-align:center;">Semantic</th>
    <th style="text-align:center;">Execution</th>
    <th style="text-align:center;">Avg.</th>
  </tr>
  <tr>
    <td style="text-align:center;">OpenVLA (Base)</td>
    <td style="text-align:center;">53.91%</td>
    <td style="text-align:center;">38.75%</td>
    <td style="text-align:center;">35.94%</td>
    <td style="text-align:center;">42.11%</td>
    <td style="text-align:center;">39.10%</td>
  </tr>
  <tr>
    <td style="text-align:center;"><a href="https://huggingface.co/gen-robot/openvla-7b-rlvla-warmup"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">RL4VLA (PPO)</td>
    <td style="text-align:center;">93.75%</td>
    <td style="text-align:center;">80.47%</td>
    <td style="text-align:center;">75.00%</td>
    <td style="text-align:center;">81.77%</td>
    <td style="text-align:center;">79.15%</td>
  </tr>
  <tr>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLA-GRPO-ManiSkill3-25ood"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">OpenVLA (RLinf-GRPO)</td>
    <td style="text-align:center;">84.38%</td>
    <td style="text-align:center;">74.69%</td>
    <td style="text-align:center;">72.99%</td>
    <td style="text-align:center;">77.86%</td>
    <td style="text-align:center;">75.15%</td>
  </tr>
  <tr>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLA-PPO-ManiSkill3-25ood"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">OpenVLA (RLinf-PPO)</td>
    <td style="text-align:center;"><strong>96.09%</strong></td>
    <td style="text-align:center;">82.03%</td>
    <td style="text-align:center;"><strong>78.35%</strong></td>
    <td style="text-align:center;"><strong>85.42%</strong></td>
    <td style="text-align:center;"><strong>81.93%</strong></td>
  </tr>
  <tr>
    <th colspan="6" style="text-align:center;"></th>
  </tr>
  <tr>
    <td style="text-align:center;">OpenVLA-OFT (Base)</td>
    <td style="text-align:center;">28.13%</td>
    <td style="text-align:center;">27.73%</td>
    <td style="text-align:center;">12.95%</td>
    <td style="text-align:center;">11.72%</td>
    <td style="text-align:center;">18.29%</td>
  </tr>
  <tr>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-ManiSkill3-25ood"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">OpenVLA-OFT (RLinf-GRPO)</td>
    <td style="text-align:center;">94.14%</td>
    <td style="text-align:center;">84.69%</td>
    <td style="text-align:center;">45.54%</td>
    <td style="text-align:center;">44.66%</td>
    <td style="text-align:center;">60.64%</td>
  </tr>
  <tr>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">OpenVLA-OFT (RLinf-PPO)</td>
    <td style="text-align:center;"><strong>97.66%</strong></td>
    <td style="text-align:center;"><strong>92.11%</strong></td>
    <td style="text-align:center;">64.84%</td>
    <td style="text-align:center;">73.57%</td>
    <td style="text-align:center;">77.05%</td>
  </tr>
</table>
</div>


<div align="center">
<table style="text-align:center;">
  <tr>
    <th colspan="7" style="text-align:center;"><strong>统一模型在五个 LIBERO 任务组上的评测结果</strong></th>
  </tr>
  <tr>
    <th style="text-align:center;">Model</th>
    <th style="text-align:center;">Spatial</th>
    <th style="text-align:center;">Object</th>
    <th style="text-align:center;">Goal</th>
    <th style="text-align:center;">10</th>
    <th style="text-align:center;">90</th>
    <th style="text-align:center;">Avg.</th>
  </tr>
  <tr>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">OpenVLA-OFT (Base)</td>
    <td style="text-align:center;">72.18%</td>
    <td style="text-align:center;">71.48%</td>
    <td style="text-align:center;">64.06%</td>
    <td style="text-align:center;">48.44%</td>
    <td style="text-align:center;">70.97%</td>
    <td style="text-align:center;">65.43%</td>
  </tr>
  <tr>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">OpenVLA-OFT (RLinf-GRPO)</td>
    <td style="text-align:center;"><strong>99.40%<strong></td>
    <td style="text-align:center;"><strong>99.80%<strong></td>
    <td style="text-align:center;"><strong>98.79%<strong></td>
    <td style="text-align:center;"><strong>93.95%<strong></td>
    <td style="text-align:center;"><strong>98.59%<strong></td>
    <td style="text-align:center;"><strong>98.11%<strong></td>
  </tr>
  <tr>
    <td style="text-align:center;">Δ Improvement</td>
    <td style="text-align:center;">+27.22</td>
    <td style="text-align:center;">+28.32</td>
    <td style="text-align:center;">+34.73</td>
    <td style="text-align:center;">+45.51</td>
    <td style="text-align:center;">+27.62</td>
    <td style="text-align:center;">+32.68</td>
  </tr>
</table>
</div>


- RLinf 同时支持 PPO 与 GRPO 算法，为视觉-语言-动作（Vision-Language-Action, VLA）模型提供最先进的训练能力。
- 该框架与主流具身智能基准测试（如 ManiSkill3 与 LIBERO）无缝集成，并在多样化的评测指标上均取得了优异表现。


### 数学推理

<div align="center">
<table>
  <tr>
    <th colspan="5" style="text-align:center;"><strong>1.5B model results</strong></th>
  </tr>
  <tr>
    <th>Model</th>
    <th>AIME 24</a></th>
    <th>AIME 25</a></th>
    <th>GPQA-diamond</a></th>
    <th>Average</th>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">DeepSeek-R1-Distill-Qwen-1.5B (base model)</a></td>
    <td>28.33</td><td>24.90</td><td>27.45</td><td>26.89</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/zwhe99/DeepMath-1.5B"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">DeepMath-1.5B</a></td>
    <td>37.80</td><td>30.42</td><td>32.11</td><td>33.44</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">DeepScaleR-1.5B-Preview</a></td>
    <td>40.41</td><td>30.93</td><td>27.54</td><td>32.96</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/inclusionAI/AReaL-1.5B-Preview-Stage-3"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">AReaL-1.5B-Preview-Stage-3</a></td>
    <td>40.73</td><td>31.56</td><td>28.10</td><td>33.46</td>
  </tr>
  <tr>
    <td>AReaL-1.5B-retrain*</td>
    <td>44.42</td><td>34.27</td><td>33.81</td><td>37.50</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/Nickyang/FastCuRL-1.5B-V3"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">FastCuRL-1.5B-V3</a></td>
    <td>43.65</td><td>32.49</td><td>35.00</td><td>37.05</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/RLinf/RLinf-math-1.5B"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;"><strong>RLinf-math-1.5B</strong></a></td>
    <td><strong>48.44</strong></td><td><strong>35.63</strong></td><td><strong>38.46</strong></td><td><strong>40.84</strong></td>
  </tr>
</table>
</div>

\* 我们使用默认设置对模型进行了 600 步的重新训练。

<div align="center">
<table>
  <tr>
    <th colspan="5" style="text-align:center;"><strong>7B model results</strong></th>
  </tr>
  <tr>
    <th>Model</th>
    <th>AIME 24</a></th>
    <th>AIME 25</a></th>
    <th>GPQA-diamond</a></th>
    <th>Average</th>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">DeepSeek-R1-Distill-Qwen-7B (base model)</a></td>
    <td>54.90</td><td>40.20</td><td>45.48</td><td>46.86</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/inclusionAI/AReaL-boba-RL-7B"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">AReaL-boba-RL-7B</a></td>
    <td>61.66</td><td>49.38</td><td>46.93</td><td>52.66</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/Skywork/Skywork-OR1-7B"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">Skywork-OR1-7B</a></td>
    <td>66.87</td><td>52.49</td><td>44.43</td><td>54.60</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/POLARIS-Project/Polaris-7B-Preview"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">Polaris-7B-Preview</a></td>
    <td><strong>68.55</strong></td><td>51.24</td><td>43.88</td><td>54.56</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/nvidia/AceMath-RL-Nemotron-7B"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">AceMath-RL-Nemotron-7B</a></td>
    <td>67.30</td><td><strong>55.00</strong></td><td>45.57</td><td>55.96</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/RLinf/RLinf-math-7B"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;"><strong>RLinf-math-7B</strong></a></td>
    <td>68.33</td><td>52.19</td><td><strong>48.18</strong></td><td><strong>56.23</strong></td>
  </tr>
</table>
</div>

- RLinf 在数学推理任务上实现了当前最先进的性能，在多个基准测试（AIME 24、AIME 25、GPQA-diamond）中，1.5B 与 7B 规模的模型均稳定超越现有方法。

## 路线图

### 1. 系统级增强
- [ ] 支持异构 GPU

- [ ] 支持异步流水线执行

- [ ] 支持专家混合（Mixture of Experts, MoE）

- [ ] 支持 vLLM 推理后端

### 2. 应用级扩展
- [ ] 支持视觉-语言模型（VLMs）训练

- [ ] 支持深度搜索智能体训练

- [ ] 支持多智能体训练
- [ ] 支持更多具身模拟器的集成 (如 [Meta-World](https://github.com/Farama-Foundation/Metaworld), [GENESIS](https://github.com/Genesis-Embodied-AI/Genesis), [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin))  
- [ ] 支持更多VLA模型，比如 [GR00T](https://github.com/NVIDIA/Isaac-GR00T), [WALL-OSS](https://huggingface.co/x-square-robot/wall-oss-flow)
- [ ] 支持世界模型（World Model）

- [ ] 支持真实世界的具身智能强化学习


## 快速开始 

完整的 RLinf 文档请见[**这里**](https://rlinf.readthedocs.io/en/latest/).

**快速上手**

  - [安装指南](https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html)
  - [快速上手 1：在 ManiSkill3 上进行 VLA 的 PPO 训练](https://rlinf.readthedocs.io/en/latest/rst_source/start/vla.html)
  - [快速上手 2：在 MATH 上进行 LLM 的 GRPO 训练](https://rlinf.readthedocs.io/en/latest/rst_source/start/llm.html)
  - [多节点训练](https://rlinf.readthedocs.io/en/latest/rst_source/start/distribute.html)
  - [模型评估](https://rlinf.readthedocs.io/en/latest/rst_source/start/eval.html)

**关键设计**
  - [统一用户接口使用](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/user/index.html)
  - [灵活的执行模式](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/mode/index.html)
  - [自动调度支持](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/scheduler/index.html)
  - [弹性通信](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/communication/index.html)

**示例库**

  - [具身智能 VLA 模型训练](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied.html)
  - [数学推理模型训练](https://rlinf.readthedocs.io/en/latest/rst_source/examples/reasoning.html)

**高级特性**

  - [Megatron-LM 的 5D 并行配置](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/5D.html)
  - [LoRA 集成以实现高效微调](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/lora.html)
  - [在不同版本的 SGLang 之间切换](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/version.html)
  - [检查点恢复与重启支持](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/resume.html)

**框架扩展**

  - [添加新环境](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/extend/new_env.html)
  - [基于 FSDP+Hugging Face 后端添加新模型](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/extend/new_model_fsdp.html)
  - [基于 Megatron+SGLang 后端添加新模型](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/extend/new_model_megatron.html)

**博客**

  - [与 VeRL 的对比](https://rlinf.readthedocs.io/en/latest/rst_source/blog/compare_with_verl.html)

## 构建状态

| Type             | Status |
| :--------------: | :----: |
| 推理 RL-MATH | [![Build Status](https://github.com/RLinf/RLinf/actions/workflows/math_e2e.yml/badge.svg)](https://github.com/RLinf/RLinf/actions/workflows/math_e2e.yml) |
| 具身 RL-VLA   | [![Build Status](https://github.com/RLinf/RLinf/actions/workflows/embodied_e2e.yml/badge.svg)](https://github.com/RLinf/RLinf/actions/workflows/embodied_e2e.yml) |


## 贡献指南
我们欢迎对 RLinf 的贡献。在参与之前，请先阅读 [贡献指南](https://rlinf.readthedocs.io/en/latest/index.html#contribution-guidelines)。

## 引用与致谢

如果您觉得 **RLinf** 对您的研究或工作有所帮助，请引用以下论文：

```bibtex
@misc{yu2025rlinfflexibleefficientlargescale,
  title={RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation}, 
  author={Chao Yu and Yuanqing Wang and Zhen Guo and Hao Lin and Si Xu and Hongzhi Zang and Quanlu Zhang and Yongji Wu and Chunyang Zhu and Junhao Hu and Zixiao Huang and Mingjie Wei and Yuqing Xie and Ke Yang and Bo Dai and Zhexuan Xu and Xiangyuan Wang and Xu Fu and Zhihao Liu and Kang Chen and Weilin Liu and Gang Liu and Boxun Li and Jianlei Yang and Zhi Yang and Guohao Dai and Yu Wang},
  year={2025},
  eprint={2509.15965},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2509.15965}, 
}
```

如果你在 RLinf 中使用了 RL+VLA，欢迎引用我们的算法技术报告和实证研究论文：

```bibtex
@misc{zang2025rlinfvlaunifiedefficientframework,
      title={RLinf-VLA: A Unified and Efficient Framework for VLA+RL Training}, 
      author={Hongzhi Zang and Mingjie Wei and Si Xu and Yongji Wu and Zhen Guo and Yuanqing Wang and Hao Lin and Liangzhi Shi and Yuqing Xie and Zhexuan Xu and Zhihao Liu and Kang Chen and Wenhao Tang and Quanlu Zhang and Weinan Zhang and Chao Yu and Yu Wang},
      year={2025},
      eprint={2510.06710},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.06710}, 
}
```

```bibtex
@misc{liu2025rlbringvlageneralization,
  title={What Can RL Bring to VLA Generalization? An Empirical Study}, 
  author={Jijia Liu and Feng Gao and Bingwen Wei and Xinlei Chen and Qingmin Liao and Yi Wu and Chao Yu and Yu Wang},
  year={2025},
  eprint={2505.19789},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2505.19789}, 
}
```

**致谢**
RLinf 的灵感来源并受益于更广泛开源社区的思想与工具。
我们特别感谢 VeRL、AReaL、Megatron-LM、SGLang 和 PyTorch Fully Sharded Data Parallel (FSDP) 的团队与贡献者。
如果我们不慎遗漏了您的项目或贡献，请提交 issue 或 pull request，以便我们能够给予您应有的致谢。

**联系方式：**
我们欢迎博士后、博士/硕士研究生以及实习生的加入。
诚邀您共同塑造强化学习基础设施与具身智能的未来！
- Chao Yu: zoeyuchao@gmail.com
- Yu Wang: yu-wang@tsinghua.edu.cn