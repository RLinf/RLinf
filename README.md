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
  <sub>RLinf: Reinforcement Learning Infrastructure for Embodied and Agentic AI</sub>
</h1>

RLinf is a flexible and scalable open-source RL infrastructure designed for Embodied and Agentic AI. The 'inf' in RLinf stands for `Infrastructure`, highlighting its role as a robust backbone for next-generation training. It also stands for `Infinite`, symbolizing the system’s support for open-ended learning, continuous generalization, and limitless possibilities in intelligence development.

<div align="center">
  <img src="docs/source-en/_static/svg/overview.svg" alt="RLinf-overview"/>
</div>


## What's NEW!
- [2026/02] 🔥 The Technical Report of our realworld online learning system [RLinf-USER: A Unified and Extensible System for Real-World Online Policy Learning in Embodied AI](https://arxiv.org/abs/2602.07837) is released. Doc: [RLinf-USER](https://rlinf.readthedocs.io/en/latest/rst_source/publications/rlinf_user.html).
- [2026/02] 🔥 RLinf supports reinforcement learning fine-tuning for [Dexbotic](https://github.com/dexmal/dexbotic). Doc: [RL on Dexbotic Model](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dexbotic.html).
- [2026/02] 🔥 RLinf supports reinforcement learning with [GSEnv](https://github.com/chenkang455/ManiSkill-GS) for Real2Sim2Real. Doc: [RL with GSEnv](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/gsenv.html).
- [2026/01] 🔥 RLinf supports reinforcement learning fine-tuning for [OpenSora World Model](https://github.com/hpcaitech/Open-Sora). Doc: [RL on OpenSora World Model](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/opensora.html).
- [2026/01] 🔥 RLinf supports reinforcement learning fine-tuning for [RoboTwin](https://github.com/robotwin-Platform/RoboTwin). Doc: [RL on RoboTwin](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html).
- [2026/01] 🔥 RLinf supports SAC training for flow matching policy. Doc: [SAC-Flow](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/sac_flow.html), paper: [SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling](https://arxiv.org/abs/2509.25756).
-- [2025/12] 🔥 RLinf supports agentic reinforcement learning on [Search-R1](https://github.com/PeterGriffinJin/Search-R1). Doc: [Search-R1](https://rlinf.readthedocs.io/en/latest/rst_source/examples/agentic/searchr1.html).
- [2025/12] 🔥 RLinf v0.2-pre is open-sourced. We support real-world RL with Franka. Doc: [RL on Franka in the RealWorld](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/franka.html).
- [2025/12] 🔥 RLinf supports reinforcement learning fine-tuning for [RoboCasa](https://github.com/robocasa/robocasa). Doc: [RL on Robocasa](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robocasa.html).
- [2025/12] 🎉 RLinf official release of [v0.1](https://github.com/RLinf/RLinf/releases/tag/v0.1).
- [2025/11] 🔥 RLinf supports reinforcement learning fine-tuning for [CALVIN](https://github.com/mees/calvin). Doc: [RL on CALVIN](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/calvin.html).
- [2025/11] 🔥 RLinf supports reinforcement learning fine-tuning for [IsaacLab](https://github.com/isaac-sim/IsaacLab). Doc: [RL on IsaacLab](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/isaaclab.html). 
- [2025/11] 🔥 RLinf supports reinforcement learning fine-tuning for [GR00T-N1.5](https://github.com/NVIDIA/Isaac-GR00T). Doc: [RL on GR00T-N1.5](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/gr00t.html).
- [2025/11] 🔥 RLinf supports reinforcement learning fine-tuning for [Metaworld](https://github.com/Farama-Foundation/Metaworld). Doc: [RL on Metaworld](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/metaworld.html).
- [2025/11] 🔥 RLinf supports reinforcement learning fine-tuning for [Behavior 1k](https://github.com/StanfordVL/BEHAVIOR-1K). Doc: [RL on Behavior 1k](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/behavior.html).
- [2025/11] Add lora support to π₀ and π₀.₅.
- [2025/10] 🔥 RLinf supports reinforcement learning fine-tuning for π₀ and π₀.₅! Doc: [RL on π₀ and π₀.₅ Models](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html). For more technical details, refer to the [RL fine-tuning for π₀ and π₀.₅ technical report](https://arxiv.org/abs/2510.25889). The report on πRL by [Machine Heart](https://mp.weixin.qq.com/s/dFlpmqmE0qfhOQmGG25X9g) and [RoboTech](https://mp.weixin.qq.com/s/S51P-Y1UYXzumnZzon2N1g) are also released.
- [2025/10] 🔥 RLinf now officially supports online reinforcement learning! Doc: [coding_online_rl](https://rlinf.readthedocs.io/en/latest/rst_source/examples/agentic/coding_online_rl.html), Blog post: [The first open-source agent online RL framework RLinf-Online](https://mp.weixin.qq.com/s/jmohmDokuWLhQHFueSHZIQ).
- [2025/10] 🔥 The RLinf Algorithm Technical Report [RLinf-VLA: A Unified and Efficient Framework for VLA+RL Training](https://arxiv.org/abs/2510.06710) is released.
- [2025/09] 🔥 [Example Gallery](https://rlinf.readthedocs.io/en/latest/rst_source/examples/index.html) is updated, users can find various off-the-shelf examples!
- [2025/09] The paper [RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation](https://arxiv.org/abs/2509.15965) is released.
- [2025/09] The [report on RLinf by Machine Heart](https://mp.weixin.qq.com/s/Xtv4gDu3lhDDGadLrzt6Aw)  is released. 
- [2025/08] RLinf is open-sourced. The formal v0.1 will be released soon.

## Key Features

RLinf has high flexibility to support diverse RL training workflows (PPO, GRPO, SAC and so on), while hiding the complexity of distributed programming. Users can easily scale RL training to a large number of GPU nodes without modifying code, meeting the increasing demand of computation for RL training.

The high flexibility allows RLinf to explore more efficient scheduling and execution. The hybrid execution mode for embodied RL achieves up to **2.434×** throughput compared to existing frameworks.

Multiple Backend Integrations

- FSDP + HuggingFace/SGLang/vLLM: rapid adaptation to new models and algorithms, ideal for beginners and fast prototyping.
- Megatron + SGLang/vLLM: optimized for large-scale training, delivering maximum efficiency for expert users with demanding workloads.

## Examples

### Embodied AI

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
          <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/maniskill.html">ManiSkill</a> ✅</li>
          <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/libero.html">LIBERO</a> ✅</li>
          <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html">RoboTwin</a> ✅</li>
          <li>RoboVerse</li>
          <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/behavior.html">BEHAVIOR</a> ✅</li>
          <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/metaworld.html">MetaWorld</a> ✅</li>
          <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/isaaclab.html">IsaacLab</a> ✅</li>
          <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/calvin.html">CALVIN</a> ✅</li>
          <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robocasa.html">RoboCasa</a> ✅</li>
          <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/frankasim.html">Franka-Sim</a> ✅</li>
          <li>More...</li>
        </ul>
      </td>
      <td>
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/franka.html">Franka Arm</a> ✅</li>
          <li>More...</li>
        </ul>
      </td>
      <td>
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><b>VLA</b></li>
          <ul>
            <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html">π₀</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html">π₀.₅</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/maniskill.html">OpenVLA</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/libero.html">OpenVLA-OFT</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/gr00t.html">GR00T</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dexbotic.html">Dexbotic</a> ✅</li>
          </ul>
          <li><b>VLM</b></li>
          <ul>
            <li>Qwen2.5-VL</li>
          </ul>
          <li><b>World Model</b></li>
          <ul>
            <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/opensora.html">OpenSora</a> ✅</li>
          </ul>
          <li><b>Custom Models</b></li>
          <ul>
            <li><a href="https://github.com/RLinf/RLinf/blob/main/docs/source-en/rst_source/examples/embodied/mlp.rst">MLP-Policy</a> ✅</li>
            <li>CNN-Policy ✅</li>
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
            <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/rlalg/sac.html">SAC</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/rlalg/crossq.html">CrossQ</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/rlalg/rlpd.html">RLPD</a> ✅</li>
            <li><a href="https://arxiv.org/abs/2509.25756">SAC-Flow</a> ✅</li>
          </ul>
          <li><b>SFT</b></li>
          <ul>
            <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/sft.html">Full-parameter SFT</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/sft.html">LoRA SFT</a> ✅</li>
          </ul>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

### Agentic AI

<table style="width: 100%; table-layout: auto; border-collapse: collapse;">
  <thead align="center" valign="bottom">
    <tr>
      <th style="min-width: 200px;">Single-Agent</th>
      <th style="min-width: 200px;">Multi-Agent</th>
    </tr>
  </thead>
  <tbody valign="top">
    <tr>
      <td>
        <ul style="margin-left: 0; padding-left: 16px;">
          <li>
            <a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/agentic/searchr1.html">
              SearchR1
            </a> ✅
          </li>
          <li>
            <a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/agentic/coding_online_rl.html">
              Online Coder
            </a> ✅
          </li>
          <li>
            <a href="https://rlinf.readthedocs.io/en/latest/rst_source/examples/agentic/reasoning.html">
              Math Reasoning RL
            </a> ✅
          </li>
        </ul>
      </td>
      <td>
        <ul style="margin-left: 0; padding-left: 16px;">
          <li>
            WideSeek-R1
          </li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>


## Quick Start
**Installation:** Users can refer to our [installation guide](https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html) to install RLinf. We recommend users to use our provided docker image (i.e., [Installation Method 1](https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html#installation-method-1-docker-image)), as the environment and dependencies of embodied RL are complex.

**Run a simple example:** After setting up the environment, users can run a simple example of embodied RL with ManiSkill3 simulator following [this document](https://rlinf.readthedocs.io/en/latest/rst_source/start/vla.html).

**SOTA RL Training Reproduction:** RLinf provides end-to-end recipes that reproduce or match **state-of-the-art (SOTA) RL results** out of the box—users can directly run our configs and scripts to obtain SOTA performance without custom engineering. Check out our [example gallery](https://rlinf.readthedocs.io/en/latest/rst_source/examples/index.html) for more details.


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

If you train your policies in physical world with RLinf, you can cite our paper:
```bibtex
@article{zang2026rlinfuser,
  title={RLinf-USER: A Unified and Extensible System for Real-World Online Policy Learning in Embodied AI}, 
  author={Hongzhi Zang and Shu'ang Yu and Hao Lin and Tianxing Zhou and Zefang Huang and Zhen Guo and Xin Xu and Jiakai Zhou and Yuze Sheng and Shizhe Zhang and Feng Gao and Wenhao Tang and Yufeng Yue and Quanlu Zhang and Xinlei Chen and Chao Yu and Yu Wang},
  year={2026},
  journal={arXiv preprint arXiv:2602.07837},
  url={https://arxiv.org/abs/2602.07837}, 
}
```

**Acknowledgements**
RLinf has been inspired by, and benefits from, the ideas and tooling of the broader open-source community.
In particular, we would like to thank the teams and contributors behind VeRL, AReaL, Megatron-LM, SGLang, and PyTorch Fully Sharded Data Parallel (FSDP), and if we have inadvertently missed your project or contribution, please open an issue or a pull request so we can properly credit you.

**Contact:**
We welcome applications from Postdocs, PhD/Master's students, and interns. Join us in shaping the future of RL infrastructure and embodied AI!
- Chao Yu: zoeyuchao@gmail.com
- Yu Wang: yu-wang@tsinghua.edu.cn