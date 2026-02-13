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
  <sub>RLinf: 为具身智能和智能体而生的强化学习框架</sub>
</h1>

RLinf 是一个灵活且可扩展的开源框架，专为具身智能和智能体而设计。名称中的 “inf” 既代表 `Infrastructure`，强调其作为新一代训练坚实基础的作用；也代表 `Infinite`，寓意其支持开放式学习、持续泛化以及智能发展的无限可能。

<div align="center">
  <img src="docs/source-en/_static/svg/overview.svg" alt="RLinf-overview"/>
</div>


## 最新动态
- [2026/02] 🔥 RLinf真机在线学习系统的论文 [RLinf-USER: A Unified and Extensible System for Real-World Online Policy Learning in Embodied AI](https://arxiv.org/abs/2602.07837) 发布了！文档：[RLinf-USER](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/publications/rlinf_user.html)。
- [2026/02] 🔥 RLinf 支持 [Dexbotic](https://github.com/dexmal/dexbotic) 强化学习微调。文档：[RL on Dexbotic Model](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/dexbotic.html)。
- [2026/02] 🔥 RLinf 支持基于 [GSEnv](https://github.com/chenkang455/ManiSkill-GS) 的 Real2Sim2Real 强化学习。文档：[RL with GSEnv](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/gsenv.html)。
- [2026/01] 🔥 基于[OpenSora World Model](https://github.com/hpcaitech/Open-Sora)的强化学习微调已经上线！文档：[RL on OpenSora World Model](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/opensora.html)。
- [2026/01] 🔥 基于[RoboTwin](https://github.com/robotwin-Platform/RoboTwin)的强化学习微调已经上线！文档：[RL on RoboTwin](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/robotwin.html)。
- [2026/01] 🔥 RLinf 支持流匹配策略的 SAC 训练，包含仿真和Franka真机环境。文档：[SAC-Flow](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/sac_flow.html)，论文：[SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling](https://arxiv.org/abs/2509.25756)。
- [2025/12] 🔥 RLinf支持[Search-R1](https://github.com/PeterGriffinJin/Search-R1)的强化学习微调，相比原版实现加速 55%！ 文档: [Search-R1](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/agentic/searchr1.html)。
- [2025/12] 🔥 RLinf v0.2-pre 发布！真机Franka的强化学习已经上线。 文档：[RL on Franka in the Real World](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/franka.html)。
- [2025/12] 🔥 基于[RoboCasa](https://github.com/robocasa/robocasa)的强化学习微调已经上线! 文档：[RL on RoboCasa](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/robocasa.html)。
- [2025/12] 🎉 RLinf正式发布[v0.1](https://github.com/RLinf/RLinf/releases/tag/v0.1)版本。
- [2025/11] 🔥 基于[CALVIN](https://github.com/mees/calvin)的强化学习微调已经上线! 文档：[RL on CALVIN](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/calvin.html)。
- [2025/11] 🔥 基于[IsaacLab](https://github.com/isaac-sim/IsaacLab)的强化学习微调已经上线! 文档：[RL on IsaacLab](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/isaaclab.html)。 
- [2025/11] 🔥 RLinf现在已经支持强化学习微调[GR00T-N1.5](https://github.com/NVIDIA/Isaac-GR00T)！文档：[RL on GR00T-N1.5](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/gr00t.html)。
- [2025/11] 🔥 基于[Metaworld](https://github.com/Farama-Foundation/Metaworld)的强化学习微调已经上线! 文档：[RL on Metaworld](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/metaworld.html)。
- [2025/11] 🔥 基于[Behavior 1k](https://github.com/StanfordVL/BEHAVIOR-1K)的强化学习微调已经上线! 文档：[RL on Behavior 1k](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/behavior.html) 。
- [2025/11] lora微调支持π₀和π₀.₅模型。
- [2025/10] 🔥 π₀和π₀.₅模型的强化学习微调已经上线! 文档：[π₀和π₀.₅模型强化学习训练](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/pi0.html)。更多技术细节请参考：[π₀ 与 π₀.₅ 模型强化学习微调技术报告](https://arxiv.org/abs/2510.25889)。机器之心与具身智能之心报道：[《RLinf上新πRL：在线强化学习微调π₀ 和 π₀.₅》](https://mp.weixin.qq.com/s/dFlpmqmE0qfhOQmGG25X9g), [《清华大学最新！πRL：用在线强化学习让机器人 “边学边做” 的通用方案》](https://mp.weixin.qq.com/s/S51P-Y1UYXzumnZzon2N1g)。
- [2025/10] 🔥 RLinf 正式支持在线强化学习！文档：[coding_online_rl](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/agentic/coding_online_rl.html)，同时发布文章 [《首个开源的Agent在线强化学习框架RLinf-Online！让你的Agent今天比昨天更聪明》](https://mp.weixin.qq.com/s/jmohmDokuWLhQHFueSHZIQ)。
- [2025/10] 🔥 RLinf算法技术报告 [《RLinf-VLA：一个统一且高效的VLA+RL训练框架》](https://arxiv.org/abs/2510.06710) 已正式发布。
- [2025/09] 🔥 [示例库](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/index.html) 已更新，用户可以在其中找到多种可直接使用的示例！
- [2025/09] 🔥 我们的论文 [《RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation》](https://arxiv.org/abs/2509.15965)已正式发布。
- [2025/09] 🔥 机器之心关于 RLinf 的报道[《首个为具身智能而生的大规模强化学习框架RLinf！清华、北京中关村学院、无问芯穹等重磅开源》](https://mp.weixin.qq.com/s/Xtv4gDu3lhDDGadLrzt6Aw)已经发布。
- [2025/08] RLinf 已经开源，正式的 v0.1 版本即将发布。


## 核心特性

RLinf具有高度灵活性，可支持多种强化学习训练工作流（PPO、GRPO、SAC等），同时隐藏了分布式编程的复杂性。用户无需修改代码即可轻松将强化学习训练扩展至大量GPU节点，满足强化学习训练日益增长的计算需求。

这种高灵活性使 RLinf 能够探索更高效的调度与执行模式。在具身强化学习中，混合执行模式的吞吐量可达现有框架的 **2.434** 倍。

多后端集成支持

- FSDP + HuggingFace/SGLang/vLLM: 快速适配新模型与新算法，非常适合初学者和快速原型验证。
- Megatron + SGLang/vLLM: 针对大规模训练进行了优化，为专家用户提供最大化效率。

### 具身智能

<table style="width: 100%; table-layout: auto; border-collapse: collapse;">
  <thead align="center" valign="bottom">
    <tr>
      <th style="min-width: 120px; text-align: left;">模拟器</th>
      <th style="min-width: 120px;">真机</th>
      <th style="min-width: 120px;">模型</th>
      <th style="min-width: 120px;">算法</th>
    </tr>
  </thead>
  <tbody valign="top">
    <tr>
      <td style="text-align: left; padding-left: 8px;">
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/maniskill.html">ManiSkill</a> ✅</li>
          <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/libero.html">LIBERO</a> ✅</li>
          <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/robotwin.html">RoboTwin</a> ✅</li>
          <li>RoboVerse</li>
          <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/behavior.html">BEHAVIOR</a> ✅</li>
          <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/metaworld.html">MetaWorld</a> ✅</li>
          <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/isaaclab.html">IsaacLab</a> ✅</li>
          <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/robocasa.html">RoboCasa</a> ✅</li>
          <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/frankasim.html">Franka-Sim</a> ✅</li>
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
          <li><b>VLA 模型</b></li>
          <ul>
            <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/pi0.html">π₀</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/pi0.html">π₀.₅</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/maniskill.html">OpenVLA</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/libero.html">OpenVLA-OFT</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/gr00t.html">GR00T</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/dexbotic.html">Dexbotic</a> ✅</li>
          </ul>
          <li><b>VLM 模型</b></li>
          <ul>
            <li>Qwen2.5-VL</li>
          </ul>
          <li><b>世界模型</b></li>
          <ul>
            <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/opensora.html">OpenSora</a> ✅</li>
          </ul>
          <li><b>自定义模型</b></li>
          <ul>
            <li><a href="https://github.com/RLinf/RLinf/blob/main/docs/source-en/rst_source/examples/embodied/mlp.rst">MLP-Policy</a> ✅</li>
            <li>CNN-Policy ✅</li>
          </ul>
        </ul>
      </td>
      <td>
        <ul style="margin-left: 0; padding-left: 16px;">
          <li><b>RL 算法</b></li>
          <ul>
            <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/rlalg/grpo.html">GRPO</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/rlalg/ppo.html">PPO</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/rlalg/dapo.html">DAPO</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/rlalg/reinforce.html">Reinforce++</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/rlalg/sac.html">SAC</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/rlalg/crossq.html">CrossQ</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/rlalg/rlpd.html">RLPD</a> ✅</li>
            <li><a href="https://arxiv.org/abs/2509.25756">SAC-Flow</a> ✅</li>
          </ul>
          <li><b>SFT</b></li>
          <ul>
            <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/sft.html">全量微调</a> ✅</li>
            <li><a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/sft.html">LoRA微调</a> ✅</li>
          </ul>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

### 智能体强化学习

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
            <a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/agentic/searchr1.html">
              SearchR1
            </a> ✅
          </li>
          <li>
            <a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/agentic/coding_online_rl.html">
              Online Coder
            </a> ✅
          </li>
          <li>
            <a href="https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/agentic/reasoning.html">
              Math推理强化学习
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

## 快速开始
**安装步骤：** 请参考我们的[安装指南](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/start/installation.html)安装RLinf。鉴于具身强化学习的环境配置较为复杂，我们推荐直接使用我们提供的Docker镜像（即[安装方法一：Docker镜像](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/start/installation.html#installation-method-1-docker-image)）。

**运行简单示例：** 环境配置完成后，用户可以参照[该文档](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/start/vla.html)的内容，运行基于ManiSkill3模拟器的具身强化学习基础示例。

**SOTA RL 训练复现：** RLinf 提供了端到端的配置和脚本，可以直接运行，无需额外工程改造，即可复现业界领先的训练效果。请参考[示例库](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/index.html)了解更多细节。


# 持续集成测试状态
RLinf 具有全面的 CI 测试，涵盖核心组件（通过单元测试）和具身、智能体和推理场景的端到端 RL 训练工作流。
以下是主分支 CI 测试状态的摘要：

| 测试名 | 状态 |
| -------- | ------ |
| 单元测试 | <img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/RLinf/RLinf/ci-tests.yml?label=Status"> |
| 智能体/推理端到端测试 | <img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/RLinf/RLinf/ci-tests.yml?label=Status"> |
| 具身智能端到端测试 | <img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/RLinf/RLinf/ci-tests.yml?label=Status"> |
| 调度器测试 | <img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/RLinf/RLinf/ci-tests.yml?label=Status"> |

## 贡献指南
我们欢迎对 RLinf 的贡献。在参与之前，请先阅读 [贡献指南](https://github.com/RLinf/RLinf?tab=contributing-ov-file#contributing-to-rlinf)。感谢以下贡献者，并诚邀更多开发者加入我们的开源项目，共建具身智能与强化学习系统。

<a href="https://github.com/RLinf/RLinf/graphs/contributors"><img src="https://stg.contrib.rocks/image?repo=RLinf/RLinf&max=240&columns=18" /></a>

## 引用与致谢

如果您觉得 **RLinf** 对您的研究或工作有所帮助，请引用以下论文：

```bibtex
@article{yu2025rlinf,
  title={RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation},
  author={Yu, Chao and Wang, Yuanqing and Guo, Zhen and Lin, Hao and Xu, Si and Zang, Hongzhi and Zhang, Quanlu and Wu, Yongji and Zhu, Chunyang and Hu, Junhao and others},
  journal={arXiv preprint arXiv:2509.15965},
  year={2025}
}
```

如果你在 RLinf 中使用了 RL+VLA，欢迎引用我们的算法技术报告和实证研究论文：

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

如果您使用了RLinf的真机在线学习系统，欢迎引用我们的文章：
```bibtex
@article{zang2026rlinfuser,
  title={RLinf-USER: A Unified and Extensible System for Real-World Online Policy Learning in Embodied AI}, 
  author={Hongzhi Zang and Shu'ang Yu and Hao Lin and Tianxing Zhou and Zefang Huang and Zhen Guo and Xin Xu and Jiakai Zhou and Yuze Sheng and Shizhe Zhang and Feng Gao and Wenhao Tang and Yufeng Yue and Quanlu Zhang and Xinlei Chen and Chao Yu and Yu Wang},
  year={2026},
  journal={arXiv preprint arXiv:2602.07837},
  url={https://arxiv.org/abs/2602.07837}, 
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