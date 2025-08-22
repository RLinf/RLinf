## Dependency Installation Guide

This installation is divided into two steps depending on the experiments you wish to run.

First, for all experiments, follow the [Common Dependencies](#common-dependencies) section to install common dependencies.

Second, for experiments depending on Megatron and SGLang/vLLM like Math, follow the [Megatron and SGLang/vLLM Dependencies](#megatron-and-sglangvllm-dependencies) section to install Megatron-related dependencies.
For embodied experiments, follow the [Embodied Dependencies](#embodied-dependencies) to install OpenVLA or Pi0 dependencies.

### Common Dependencies
We recommend using [`uv`](https://docs.astral.sh/uv/) to install the necessary Python dependencies.
If your are using [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html), you can also install `uv` via `pip`.
```shell
conda create -n rlinf python=3.11.10 -y
conda activate rlinf
pip install --upgrade uv
```

After installing `uv`, create a virtual environment and install Pytorch as well as the common dependencies.
```shell
uv venv
source .venv/bin/activate
UV_TORCH_BACKEND=auto uv sync
```

### Megatron and SGLang/vLLM Dependencies
Run the following to install Megatron, SGLang/vLLM and their dependencies.
```shell
uv sync --extra sgl_vllm
mkdir -p /opt && git clone https://github.com/NVIDIA/Megatron-LM.git -b core_r0.11.0 /opt/Megatron-LM
uv pip install -r requirements/megatron.txt --no-build-isolation
```
Before using Megatron, make sure it's path is added to the `PYTHONPATH` environment variables.
```shell
export PYTHONPATH=/opt/Megatron-LM:$PYTHONPATH
```

### Embodied Dependencies
For embodied experiments, first install the necessary system dependencies (currently only Debian/Ubuntu `apt` package management is supported).
```shell
bash requirements/install_embodied_deps.sh
uv sync --extra embodied
```
Next, depending on the experiment types, install the `openvla` or `pi0` dependencies.
```shell
# For OpenVLA/OpenVLA-oft experiments
UV_TORCH_BACKEND=auto uv pip install -r requirements/openvla.txt

# For Pi0 experiment
UV_TORCH_BACKEND=auto uv pip install -r requirements/pi0.txt
```