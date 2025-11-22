# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Override Ray's NvidiaGPUAcceleratorManager
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py

import os
import warnings

from ray._private.accelerators.nvidia_gpu import NvidiaGPUAcceleratorManager

from .accelerator import AcceleratorManager, AcceleratorType


@AcceleratorManager.register_manager(AcceleratorType.NV_GPU)
class NvidiaGPUManager(AcceleratorManager):
    """Utility Class for NVIDIA GPU."""

    @staticmethod
    def _parse_nvidia_gpu_model(model_str: str) -> str:
        """Parse the NVIDIA GPU model from the full name string.

        Args:
            model_str (str): The full name string of the NVIDIA GPU.

        Returns:
            str: The parsed model of the NVIDIA GPU.
        """
        # Example model_str: "NVIDIA GeForce RTX 3090, "NVIDIA A100-SXM4-40GB"
        UNRELATED_KEYWORDS = {"NVIDIA", "GeForce"}

        if model_str is None:
            return None

        parts = model_str.split()
        # Filter out unrelated keywords
        filtered_parts = [part for part in parts if part not in UNRELATED_KEYWORDS]
        if filtered_parts:
            return " ".join(filtered_parts)
        return "UNKNOWN"

    @staticmethod
    def get_num_devices():
        """Get the number of NVIDIA GPU devices on the node."""
        return NvidiaGPUAcceleratorManager.get_current_node_num_accelerators()

    @staticmethod
    def get_accelerator_type():
        """Get the type of the accelerator."""
        return AcceleratorType.NV_GPU

    @staticmethod
    def get_accelerator_model():
        """Get the model of the NVIDIA GPU."""
        import ray._private.thirdparty.pynvml as pynvml

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            model = pynvml.nvmlDeviceGetName(handle)
            if isinstance(model, bytes):
                model = model.decode("utf-8")
            model = NvidiaGPUManager._parse_nvidia_gpu_model(model)
            pynvml.nvmlShutdown()
            return model
        except pynvml.NVMLError as _:
            return "UNKNOWN"

    @staticmethod
    def get_accelerator_env_var(visible_accelerators: list[str]) -> dict[str, str]:
        """Get the environment variables related to the accelerator.

        Args:
            visible_accelerators (List[str]): A list of visible accelerator IDs.

        Returns:
            Dict[str, str]: A dictionary containing the accelerator environment variables.
        """
        env_vars = {}
        visible_accelerators_str = ",".join(visible_accelerators)

        # All the three types of GPU can be set together
        env_vars["CUDA_VISIBLE_DEVICES"] = visible_accelerators_str
        # Override Ray's control over GPU assignment
        env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
        # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96

        # Simulator env vars
        if len(visible_accelerators) > 0:
            env_vars["MUJOCO_EGL_DEVICE_ID"] = str(visible_accelerators[0])

        # NCCL env vars
        env_vars["NCCL_CUMEM_ENABLE"] = "0"
        env_vars["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        if os.environ.get("NCCL_CUMEM_ENABLE", "0") != "0":
            warnings.warn(
                f"NCCL_CUMEM_ENABLE is set to {os.environ['NCCL_CUMEM_ENABLE']}. However, "
                "This may increase memory overhead with cudagraph+allreduce: "
                "https://github.com/NVIDIA/nccl/issues/1234, and thus set to 0 by both vLLM and SGLang, see https://github.com/vllm-project/vllm/pull/24141.",
            )
            env_vars["NCCL_CUMEM_ENABLE"] = os.environ["NCCL_CUMEM_ENABLE"]

        return env_vars

    @staticmethod
    def get_visible_devices():
        """Get the visible device IDs."""
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)

        if visible_devices is None or visible_devices == "":
            return []
        else:
            try:
                visible_devices = [int(v.strip()) for v in visible_devices.split(",")]
            except ValueError:
                raise ValueError(
                    f"Invalid visible device IDs: {visible_devices}. "
                    "Please ensure they are integers separated by commas."
                )
            return visible_devices

    @staticmethod
    def get_ccl_backend():
        """Get the CCL backend."""
        return "nccl"

    @staticmethod
    def get_ccl_socket_ifname_env_var() -> str:
        """Get the network socket interface name environment variable.

        Returns:
            str: The network socket interface name environment variable.
        """
        return "NCCL_SOCKET_IFNAME"

    @staticmethod
    def get_torch_platform():
        """Get the PyTorch platform module."""
        import torch

        return torch.cuda

    @staticmethod
    def get_device_type() -> str:
        """Get the device type."""
        return "cuda"
