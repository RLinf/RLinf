<h1 align="center">WideSeek-R1: Exploring Width Scaling for Broad Information Seeking via MARL</h1>

<div align="center">

[![paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/xxx)
&nbsp;
[![Model](https://img.shields.io/badge/Hugging%20Face-Model-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/RLinf/WideSeek-R1-4b)
&nbsp;
[![Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/RLinf/WideSeek-R1-train-data)
&nbsp;
[![Website](https://img.shields.io/badge/Website-1A73E8?style=for-the-badge&logo=googledocs&logoColor=white)](https://wideseek-r1.github.io)

</div>

## 📝 Overview

![overview](https://github.com/RLinf/misc/raw/main/pic/wideseek_r1/teaser.jpg)


we propose WideSeek-R1, a lead-agent-subagent framework trained via multi-agent reinforcement learning (MARL) to synergize scalable orchestration and parallel execution. By utilizing a shared LLM with isolated contexts and specialized tools, WideSeek-R1 jointly optimizes the lead agent and parallel subagents on a curated dataset of 20k broad information-seeking tasks. Extensive experiments show that WideSeek-R1-4B achieves an item F1 score of 40.0% on the WideSearch benchmark, which is comparable to the performance of single-agent DeepSeek-R1-671B. Furthermore, WideSeek-R1-4B exhibits consistent performance gains as the number of parallel subagents increases, highlighting the effectiveness of width scaling.


## 🏆 Results


### Single Agent

| Model            | Item F1 Avg@4 (%) | Item F1 Max@4 (%) | Row F1 Avg@4 (%) | Row F1 Max@4 (%) | Success Avg@4 (%) | Success Pass@4 (%) |
| ---------------- | ----------------: | ----------------: | ---------------: | ---------------: | ----------------: | -----------------: |
| SingleSeek-R1-4B |              28.1 |              39.2 |              6.5 |             12.5 |               0.3 |                1.0 |
| Qwen3-4B         |              20.1 |              30.2 |              3.0 |              4.8 |               0.0 |                0.0 |
| Search-R1-7B     |              15.5 |              24.4 |              2.0 |              4.4 |               0.0 |                0.0 |
| ASearcher-7B     |              16.5 |              26.0 |              2.8 |              5.8 |               0.0 |                0.0 |
| DeepSeek-R1-671B |              41.3 |              55.1 |             20.7 |             31.7 |               0.4 |                1.5 |

### Multi-Agent System

| Model        | Item F1 Avg@4 (%) | Item F1 Max@4 (%) | Row F1 Avg@4 (%) | Row F1 Max@4 (%) | Success Avg@4 (%) | Success Pass@4 (%) |
| ------------ | ----------------: | ----------------: | ---------------: | ---------------: | ----------------: | -----------------: |
| **WideSeek-R1-4B**     |              **40.0** |              **51.8** |             **15.3** |             **24.4** |               **0.4** |                **1.0** |
| Qwen3-4B     |              31.2 |              42.3 |              8.4 |             15.5 |               0.0 |                0.0 |
| AgentFlow-7B |              28.7 |              45.4 |              9.0 |             20.2 |               0.4 |                1.5 |
| OWL-8B       |              20.2 |              29.3 |              3.1 |              5.8 |               0.0 |                0.0 |
| MiroFlow-8B  |              23.7 |              37.7 |              5.8 |             12.7 |               0.4 |                1.0 |

---

## Installation

我们目前支持两种安装方式，基于docker安装和基于uv安装。两种安装方式的详情均参考 [RLinf docs](https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html)
其中基于docker安装，请使用镜像

```bash
docker pull rlinf/rlinf:math-rlinf0.1-torch2.6.0-sglang0.4.6.post5-vllm0.8.5-megatron0.13.0-te2.1
```

基于uv的安装，请遵循`reasoning`的环境，也就是基于

```bash
bash requirements/install.sh reason
```


## QuickStart

### Evluation
0. 下载数据集（ws的eval的，需要push到hf）
1. 下载模型
2. 启动eval
3. 用self contain的ws的代码eval，提pr到rlinf org下（顺带把dataproducer提了）

### Training
0. 下载数据集
1. 启动tool
2. 启动qwen3 30b的llm judge
3. 启动training


## Data producer

See more details in xxx

## Acknowledgement

## 📚 Citation
```
@article{xu2026wideseek,
  title   = {WideSeek-R1: Exploring Width Scaling for Broad Information Seeking via Multi-Agent Reinforcement Learning},
  author  = {Xu, Zelai and Xu, Zhexuan and Zhang, Ruize and Zhu, Chunyang and Yu, Shi and Liu, Weilin and Zhang, Quanlu and Ding, Wenbo and Yu, Chao and Wang, Yu},
  journal = {arXiv preprint arXiv:2602.04634},
  year    = {2026},
}
```