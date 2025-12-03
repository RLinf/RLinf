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
  <sub>RLinf: Reinforcement Learning Infrastructure for Post-training</sub>
</h1>

RLinf is a flexible and scalable open-source infrastructure designed for post-training foundation models via reinforcement learning. The 'inf' in RLinf stands for `Infrastructure`, highlighting its role as a robust backbone for next-generation training. It also stands for `Infinite`, symbolizing the system’s support for open-ended learning, continuous generalization, and limitless possibilities in intelligence development.

<div align="center">
  <img src="docs/source-en/_static/svg/overview.svg" alt="RLinf-overview"/>
</div>


## What's NEW!
- [2025/11] 🔥 RLinf supports reinforcement learning fine-tuning for [IsaacLab](https://github.com/isaac-sim/IsaacLab). Doc: [RL on IsaacLab](https://rlinf.readthedocs.io/en/latest/rst_source/examples/isaaclab.html) 
- [2025/11] 🔥 RLinf supports reinforcement learning fine-tuning for [Behavior 1k](https://github.com/StanfordVL/BEHAVIOR-1K). Doc: [RL on Behavior 1k](https://rlinf.readthedocs.io/en/latest/rst_source/examples/behavior.html) 
- [2025/11] 🔥 RLinf supports reinforcement learning fine-tuning for [GR00T-N1.5](https://github.com/NVIDIA/Isaac-GR00T). Doc: [RL on GR00T-N1.5](https://rlinf.readthedocs.io/en/latest/rst_source/examples/gr00t.html).
- [2025/11] 🔥 RLinf supports reinforcement learning fine-tuning for [Metaworld](https://github.com/Farama-Foundation/Metaworld). Doc: [RL on Metaworld](https://rlinf.readthedocs.io/en/latest/rst_source/examples/metaworld.html).
- [2025/11] 🔥 RLinf supports reinforcement learning fine-tuning for [Behavior 1k](https://github.com/StanfordVL/BEHAVIOR-1K). Doc: [RL on Behavior 1k](https://rlinf.readthedocs.io/en/latest/rst_source/examples/behavior.html).
- [2025/11] Add lora support to π₀ and π₀.₅.
- [2025/10] 🔥 RLinf supports reinforcement learning fine-tuning for π₀ and π₀.₅! Doc: [RL on π₀ and π₀.₅ Models](https://rlinf.readthedocs.io/en/latest/rst_source/examples/pi0.html). For more technical details, refer to the [RL fine-tuning for π₀ and π₀.₅ technical report](https://arxiv.org/abs/2510.25889). The report on πRL by [Machine Heart](https://mp.weixin.qq.com/s/dFlpmqmE0qfhOQmGG25X9g) and [RoboTech](https://mp.weixin.qq.com/s/S51P-Y1UYXzumnZzon2N1g) are also released.
- [2025/10] 🔥 RLinf now officially supports online reinforcement learning! Doc: [coding_online_rl](https://rlinf.readthedocs.io/en/latest/rst_source/examples/coding_online_rl.html), Blog post: [The first open-source agent online RL framework RLinf-Online](https://mp.weixin.qq.com/s/jmohmDokuWLhQHFueSHZIQ).
- [2025/10] 🔥 The RLinf Algorithm Technical Report [RLinf-VLA: A Unified and Efficient Framework for VLA+RL Training](https://arxiv.org/abs/2510.06710) is released.
- [2025/09] 🔥 [Example Gallery](https://rlinf.readthedocs.io/en/latest/rst_source/examples/index.html) is updated, users can find various off-the-shelf examples!
- [2025/09] The paper [RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation](https://arxiv.org/abs/2509.15965) is released.
- [2025/09] The [report on RLinf by Machine Heart](https://mp.weixin.qq.com/s/Xtv4gDu3lhDDGadLrzt6Aw)  is released. 
- [2025/08] RLinf is open-sourced. The formal v0.1 will be released soon.

## Key Features


### Embodied Intelligence

<table style="width: 100%; table-layout: auto; border-collapse: collapse;">
  <thead align="center" valign="bottom">
    <tr>
      <th style="min-width: 120px; text-align: left;">Simulators</th>
      <th style="min-width: 120px;">Real-world Robotics</th>
      <th style="min-width: 120px;">Models</th>
      <th style="min-width: 120px;">Algorithms</th>
    </tr>
  </thead>
  <tbody valign="top">
    <tr>
      <td style="text-align: left; padding-left: 8px;">
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/maniskill.html">ManiSkill</a> ✅</li>
          <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/libero.html">LIBERO</a> ✅</li>
          <li>RoboTwin</li>
          <li>RoboVerse</li>
          <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/behavior.html">BEHAVIOR</a> ✅</li>
          <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/metaworld.html">MetaWorld</a> ✅</li>
          <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/isaaclab.html">IsaacLab</a> ✅</li>
          <li>RoboCasa</li>
          <li>More...</li>
        </ul>
      </td>
      <td>
        <ul style="margin-left: 0; padding-left: 16px;">
          <li>Franka Arm</li>
          <li>More...</li>
        </ul>
      </td>
      <td>
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><b>VLA</b></li>
          <ul>
            <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/pi0.html">π₀</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/pi0.html">π₀.₅</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/maniskill.html">OpenVLA</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/libero.html">OpenVLA-OFT</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/gr00t.html">GR00T</a> ✅</li>
          </ul>
          <li><b>VLM</b></li>
          <ul>
            <li>Qwen2.5-VL</li>
          </ul>
          <li><b>Custom Models</b></li>
          <ul>
            <li>MLP-Policy ✅</li>
          </ul>
        </ul>
      </td>
      <td>
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><b>RL Algos</b></li>
          <ul>
            <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/rlalg/grpo.html">GRPO</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/rlalg/ppo.html">PPO</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/rlalg/dapo.html">DAPO</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/rlalg/reinforce.html">Reinforce++</a> ✅</li>
            <li>SAC</li>
          </ul>
          <li><b>SFT</b></li>
          <ul>
            <li>Full-parameter SFT</li>
            <li>LoRA SFT</li>
          </ul>
        </ul>
      </td>
    </tr>
  </tbody>
</table>
  
RLinf supports mainstream VLA models, mainstream CPU & GPU-parallel simulators via standardized Worker interfaces, and enables the first RL fine-tuning of the $\pi_{0}$ and $\pi_{0.5}$ model family with a flow-matching action expert, as shown in the above table.

### Agentic RL

Agentic RL includes both RL training for improving LLM reasoning ability, such as [Math Reasoning](https://rlinf.readthedocs.io/en/latest/rst_source/examples/reasoning.html), and RL training for Agents, for example, [RL training of coding agent](https://rlinf.readthedocs.io/en/latest/rst_source/examples/coding_online_rl.html). We believe embodied intelligence will also integrate the ability of agents in the future to complete complex tasks.

### High flexibility, efficiency, and scalability

Besides the rich functionalities introduced above, RLinf has high flexibility to support diverse RL training workflows (PPO, GRPO, SAC and so on), while hiding the complexity of distributed programming. Users can easily scale RL training to a large number of GPU nodes without modifying code, meeting the increasing demand of computation for RL training.

The high flexibility allows RLinf to explore more efficient scheduling and execution. The hybrid execution mode for embodied RL achieves a **100%+** throughput improvement compared to baseline solutions.

Multiple Backend Integrations

- FSDP + HuggingFace/SGLang/vLLM: rapid adaptation to new models and algorithms, ideal for beginners and fast prototyping.
- Megatron + SGLang/vLLM: optimized for large-scale training, delivering maximum efficiency for expert users with demanding workloads.

## Quick Start
**Installation:** Users can refer to our [installation guide](https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html) to install RLinf. We recommend users to use our provided docker image (i.e., [Installation Method 1](https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html#installation-method-1-docker-image)), as the environment and dependencies of embodied RL are complex.

**Run a simple example:** After setting up the environment, users can run a simple example of embodied RL with ManiSkill3 simulator following [this document](https://rlinf.readthedocs.io/en/latest/rst_source/start/vla.html).

For more tutorials of RLinf and application examples, checkout our [documentation](https://rlinf.readthedocs.io/en/latest/index.html) and [example gallery](https://rlinf.readthedocs.io/en/latest/rst_source/examples/index.html).


## Main Results
### Embodied Intelligence

- RLinf supports both PPO and GRPO algorithms, enabling state-of-the-art training for Vision-Language-Action models.
- The framework provides seamless integration with mainstream embodied intelligence benchmarks, and achieves strong performance across diverse evaluation metrics.

#### OpenVLA and OpenVLA-OFT Results

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

- Training curves on ManiSkill “PutOnPlateInScene25Mani-v3” with OpenVLA and
OpenVLA-OFT models, using PPO and GRPO algorithms. PPO consistently outperforms GRPO
and exhibits greater stability.

<div align="center">
<table style="text-align:center;">
  <tr>
    <th colspan="6" style="text-align:center;"> <strong>Evaluation results on ManiSkill. Values denote success rates</strong></th>
  </tr>
  <tr>
    <td style="text-align:center;"></td>
    <th rowspan="2" colspan="1" style="text-align:center;">In-Distribution</th>
    <td colspan="4" style="text-align:center;"><strong>Out-Of-Distribution</strong></td>
  
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
    <td style="text-align:center;"><a href="https://huggingface.co/gen-robot/openvla-7b-rlvla-rl"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">RL4VLA (PPO)</a></td>
    <td style="text-align:center;">93.75%</td>
    <td style="text-align:center;">80.47%</td>
    <td style="text-align:center;">75.00%</td>
    <td style="text-align:center;">81.77%</td>
    <td style="text-align:center;">79.15%</td>
  </tr>
  <tr>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLA-GRPO-ManiSkill3-25ood"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">OpenVLA (RLinf-GRPO)</a></td>
    <td style="text-align:center;">84.38%</td>
    <td style="text-align:center;">74.69%</td>
    <td style="text-align:center;">72.99%</td>
    <td style="text-align:center;">77.86%</td>
    <td style="text-align:center;">75.15%</td>
  </tr>
  <tr>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLA-PPO-ManiSkill3-25ood"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">OpenVLA (RLinf-PPO)</a></td>
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
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-ManiSkill3-25ood"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">OpenVLA-OFT (RLinf-GRPO)</a></td>
    <td style="text-align:center;">94.14%</td>
    <td style="text-align:center;">84.69%</td>
    <td style="text-align:center;">45.54%</td>
    <td style="text-align:center;">44.66%</td>
    <td style="text-align:center;">60.64%</td>
  </tr>
  <tr>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">OpenVLA-OFT (RLinf-PPO)</a></td>
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
    <th colspan="7" style="text-align:center;"><strong>Evaluation results of the unified model on the five LIBERO task groups</strong></th>
  </tr>
  <tr>
    <th style="text-align:center;">Model</th>
    <th style="text-align:center;">Spatial</th>
    <th style="text-align:center;">Object</th>
    <th style="text-align:center;">Goal</th>
    <th style="text-align:center;">Long</th>
    <th style="text-align:center;">90</th>
    <th style="text-align:center;">Avg.</th>
  </tr>
  <tr>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">OpenVLA-OFT (Base)</a></td>
    <td style="text-align:center;">72.18%</td>
    <td style="text-align:center;">71.48%</td>
    <td style="text-align:center;">64.06%</td>
    <td style="text-align:center;">48.44%</td>
    <td style="text-align:center;">70.97%</td>
    <td style="text-align:center;">65.43%</td>
  </tr>
  <tr>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130"><img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">OpenVLA-OFT (RLinf-GRPO)</a></td>
    <td style="text-align:center;"><strong>99.40%</strong></td>
    <td style="text-align:center;"><strong>99.80%</strong></td>
    <td style="text-align:center;"><strong>98.79%</strong></td>
    <td style="text-align:center;"><strong>93.95%</strong></td>
    <td style="text-align:center;"><strong>98.59%</strong></td>
    <td style="text-align:center;"><strong>98.11%</strong></td>
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

#### &pi;<sub>0</sub> and &pi;<sub>0.5</sub> Results

<div align="center">
<table style="text-align:center; width:80%; margin:0 auto;">
  <tr>
    <th colspan="8" style="text-align:center;"><strong>Evaluation results on the four LIBERO task groups</strong></th>
  </tr>
  <tr>
    <th rowspan="2" colspan="2" style="text-align:center;">Model</th>
    <th colspan="6" style="text-align:center;">LIBERO</th>
  </tr>
  <tr>
    <th style="text-align:center;">Spatial</th>
    <th style="text-align:center;">Object</th>
    <th style="text-align:center;">Goal</th>
    <th style="text-align:center;">Long</th>
    <th style="text-align:center;">Avg.</th>
    <th style="text-align:center;">&Delta; Avg.</th>
  </tr>

  <!-- Full Dataset SFT (6 rows) -->
  <tr>
    <td colspan="8" style="text-align:center; font-style:italic;"><strong>Full Dataset SFT</strong></td>
  </tr>
  <tr>
    <td colspan="2" style="text-align:center;">Octo</td>
    <td style="text-align:center;">78.9%</td>
    <td style="text-align:center;">85.7%</td>
    <td style="text-align:center;">84.6%</td>
    <td style="text-align:center;">51.1%</td>
    <td style="text-align:center;">75.1%</td>
    <td style="text-align:center;">—</td>
  </tr>
  <tr>
    <td colspan="2" style="text-align:center;">OpenVLA</td>
    <td style="text-align:center;">84.7%</td>
    <td style="text-align:center;">88.4%</td>
    <td style="text-align:center;">79.2%</td>
    <td style="text-align:center;">53.7%</td>
    <td style="text-align:center;">76.5%</td>
    <td style="text-align:center;">—</td>
  </tr>
  <tr>
    <td colspan="2" style="text-align:center;">&pi;<sub>fast</sub></td>
    <td style="text-align:center;">96.4%</td>
    <td style="text-align:center;">96.8%</td>
    <td style="text-align:center;">88.6%</td>
    <td style="text-align:center;">60.2%</td>
    <td style="text-align:center;">85.5%</td>
    <td style="text-align:center;">—</td>
  </tr>
  <tr>
    <td colspan="2" style="text-align:center;">OpenVLA-OFT</td>
    <td style="text-align:center;">91.6%</td>
    <td style="text-align:center;">95.3%</td>
    <td style="text-align:center;">90.6%</td>
    <td style="text-align:center;">86.5%</td>
    <td style="text-align:center;">91.0%</td>
    <td style="text-align:center;">—</td>
  </tr>
  <tr>
    <td colspan="2" style="text-align:center;">&pi;<sub>0</sub></td>
    <td style="text-align:center;">96.8%</td>
    <td style="text-align:center;">98.8%</td>
    <td style="text-align:center;">95.8%</td>
    <td style="text-align:center;">85.2%</td>
    <td style="text-align:center;">94.2%</td>
    <td style="text-align:center;">—</td>
  </tr>
  <tr>
    <td colspan="2" style="text-align:center;">&pi;<sub>0.5</sub></td>
    <td style="text-align:center;">98.8%</td>
    <td style="text-align:center;">98.2%</td>
    <td style="text-align:center;">98.0%</td>
    <td style="text-align:center;">92.4%</td>
    <td style="text-align:center;">96.9%</td>
    <td style="text-align:center;">—</td>
  </tr>

  <!-- Few-shot SFT + RL: pi_0 -->
  <tr>
    <td colspan="8" style="text-align:center; font-style:italic;"><strong>Few-shot Dataset SFT + RL</strong></td>
  </tr>
  <tr>
    <td rowspan="3" style="text-align:center;">&pi;<sub>0</sub></td>
    <td style="text-align:center;">
      <a href="https://www.modelscope.cn/models/RLinf/RLinf-Pi0-SFT-Spatial-Object-Goal">
        <img src="docs/source-en/_static/svg/modelscope-logo.svg" alt="ModelScope" width="16" height="16" style="vertical-align: middle;">
      </a>
      <a href="https://huggingface.co/RLinf/RLinf-Pi0-SFT-Spatial-Object-Goal">
        <img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">SFT
      </a>
    </td>
    <td style="text-align:center;">65.3%</td>
    <td style="text-align:center;">64.4%</td>
    <td style="text-align:center;">49.8%</td>
    <td style="text-align:center;">51.2%</td>
    <td style="text-align:center;">57.6%</td>
    <td style="text-align:center;">—</td>
  </tr>
  <tr>
    <td style="text-align:center;">Flow-SDE</td>
    <td style="text-align:center;">98.4%</td>
    <td style="text-align:center;">99.4%</td>
    <td style="text-align:center;">96.2%</td>
    <td style="text-align:center;">90.2%</td>
    <td style="text-align:center;">96.1%</td>
    <td style="text-align:center;">+38.5</td>
  </tr>
  <tr>
    <td style="text-align:center;">Flow-Noise</td>
    <td style="text-align:center;">99.0%</td>
    <td style="text-align:center;">99.2%</td>
    <td style="text-align:center;">98.2%</td>
    <td style="text-align:center;">93.8%</td>
    <td style="text-align:center;">97.6%</td>
    <td style="text-align:center;"><b>+40.0</b></td>
  </tr>

  <!-- Few-shot SFT + RL: pi_0.5 -->
  <tr>
    <td colspan="8" style="text-align:center; font-style:italic;"><strong>Few-shot Dataset SFT + RL</strong></td>
  </tr>
  <tr>
    <td rowspan="3" style="text-align:center;">&pi;<sub>0.5</sub></td>
    <td style="text-align:center;">
      <a href="https://www.modelscope.cn/models/RLinf/RLinf-Pi05-SFT">
        <img src="docs/source-en/_static/svg/modelscope-logo.svg" alt="ModelScope" width="16" height="16" style="vertical-align: middle;">
      </a>
      <a href="https://huggingface.co/RLinf/RLinf-Pi05-SFT">
        <img src="docs/source-en/_static/svg/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">SFT
      </a>
    </td>
    <td style="text-align:center;">84.6%</td>
    <td style="text-align:center;">95.4%</td>
    <td style="text-align:center;">84.6%</td>
    <td style="text-align:center;">43.9%</td>
    <td style="text-align:center;">77.1%</td>
    <td style="text-align:center;">—</td>
  </tr>
  <tr>
    <td style="text-align:center;">Flow-SDE</td>
    <td style="text-align:center;">99.6%</td>
    <td style="text-align:center;">100%</td>
    <td style="text-align:center;">98.8%</td>
    <td style="text-align:center;">93.0%</td>
    <td style="text-align:center;">97.9%</td>
    <td style="text-align:center;">+20.8</td>
  </tr>
  <tr>
    <td style="text-align:center;">Flow-Noise</td>
    <td style="text-align:center;"><b>99.6%</b></td>
    <td style="text-align:center;"><b>100%</b></td>
    <td style="text-align:center;"><b>99.6%</b></td>
    <td style="text-align:center;"><b>94.0%</b></td>
    <td style="text-align:center;"><b>98.3%</b></td>
    <td style="text-align:center;">+21.2</td>
  </tr>
</table>
</div>


### Math Reasoning

<div align="center">
<table style="text-align:center;">
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

\* We retrain the model using the default settings for 600 steps.

<div align="center">
<table style="text-align:center;">
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

- RLinf achieves state-of-the-art performance on math reasoning tasks, consistently outperforming existing models across multiple benchmarks (AIME 24, AIME 25, GPQA-diamond) for both 1.5B and 7B model sizes.

## Roadmap

### 1. System-Level Enhancements
- [X] Support for heterogeneous GPUs  
- [ ] Support for asynchronous pipeline execution  
- [X] Support for Mixture of Experts (MoE)  
- [X] Support for vLLM inference backend

### 2. Application-Level Extensions
- [X] Support for Vision-Language Models (VLMs) training  
- [ ] Support for deep searcher agent training  
- [ ] Support for multi-agent training  
- [ ] Support for integration with more embodied simulators (e.g., [RoboCasa](https://github.com/robocasa/robocasa), [GENESIS](https://github.com/Genesis-Embodied-AI/Genesis), [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin))  
- [ ] Support for more Vision Language Action models (VLAs), such as [WALL-OSS](https://huggingface.co/x-square-robot/wall-oss-flow)
- [ ] Support for world model   
- [ ] Support for real-world RL embodied intelligence

# CI Test Status
RLinf has comprehensive CI tests for both the core components (via unit tests) and end-to-end RL training workflows of embodied, agent, and reasoning scenarios.
Below is the summary of the CI test status of the main branch:

| Test Name | Status |
| -------- | ------ |
| unit-tests | <img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/RLinf/RLinf/ci-tests.yml?label=Status"> |
| agent-reason-e2e-tests | <img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/RLinf/RLinf/ci-tests.yml?label=Status"> |
| embodied-e2e-tests | <img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/RLinf/RLinf/ci-tests.yml?label=Status"> |
| scheduler-tests | <img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/RLinf/RLinf/ci-tests.yml?label=Status"> |

## Contribution Guidelines
We welcome contributions to RLinf. Please read [contribution guide](https://github.com/RLinf/RLinf?tab=contributing-ov-file#contributing-to-rlinf) before taking action. Thank the following contributors and welcome more developers to join us on this open source project.

<a href="https://github.com/RLinf/RLinf/graphs/contributors"><img src="https://stg.contrib.rocks/image?repo=RLinf/RLinf&max=240&columns=18" /></a>

## Citation and Acknowledgement

If you find **RLinf** helpful, please cite the paper:

```bibtex
@article{yu2025rlinf,
  title={RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation},
  author={Yu, Chao and Wang, Yuanqing and Guo, Zhen and Lin, Hao and Xu, Si and Zang, Hongzhi and Zhang, Quanlu and Wu, Yongji and Zhu, Chunyang and Hu, Junhao and others},
  journal={arXiv preprint arXiv:2509.15965},
  year={2025}
}
```

If you use RL+VLA in RLinf, you can also cite our technical report and empirical study paper:

```bibtex
@article{zang2025rlinf,
  title={RLinf-VLA: A Unified and Efficient Framework for VLA+ RL Training},
  author={Zang, Hongzhi and Wei, Mingjie and Xu, Si and Wu, Yongji and Guo, Zhen and Wang, Yuanqing and Lin, Hao and Shi, Liangzhi and Xie, Yuqing and Xu, Zhexuan and others},
  journal={arXiv preprint arXiv:2510.06710},
  year={2025}
}
```

```bibtex
@article{liu2025can,
  title={What can rl bring to vla generalization? an empirical study},
  author={Liu, Jijia and Gao, Feng and Wei, Bingwen and Chen, Xinlei and Liao, Qingmin and Wu, Yi and Yu, Chao and Wang, Yu},
  journal={arXiv preprint arXiv:2505.19789},
  year={2025}
}
```

```bibtex
@article{chen2025pi_,
  title={$$\backslash$pi\_$\backslash$texttt $\{$RL$\}$ $: Online RL Fine-tuning for Flow-based Vision-Language-Action Models},
  author={Chen, Kang and Liu, Zhihao and Zhang, Tonghe and Guo, Zhen and Xu, Si and Lin, Hao and Zang, Hongzhi and Zhang, Quanlu and Yu, Zhaofei and Fan, Guoliang and others},
  journal={arXiv preprint arXiv:2510.25889},
  year={2025}
}
```

**Acknowledgements**
RLinf has been inspired by, and benefits from, the ideas and tooling of the broader open-source community.
In particular, we would like to thank the teams and contributors behind VeRL, AReaL, Megatron-LM, SGLang, and PyTorch Fully Sharded Data Parallel (FSDP), and if we have inadvertently missed your project or contribution, please open an issue or a pull request so we can properly credit you.

**Contact:**
We welcome applications from Postdocs, PhD/Master's students, and interns. Join us in shaping the future of RL infrastructure and embodied AI!
- Chao Yu: zoeyuchao@gmail.com
- Yu Wang: yu-wang@tsinghua.edu.cn