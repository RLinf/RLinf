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

import os
import warnings
from enum import Enum


class AcceleratorType(Enum):
    """Enum representing different types of accelerators."""

    NV_GPU = "NV_GPU"
    AMD_GPU = "AMD_GPU"
    INTEL_GPU = "INTEL_GPU"
    NPU = "NPU"  # Huawei Ascend
    NO_ACCEL = "NO_ACCEL"


class Accelerator:
    """Utility class representing an accelerator and abstracting device operations."""

    SUPPORTED_LIST = ["GPU", "NPU"]
    UNSUPPORTED_LIST = ["neuron_cores", "TPU", "HPU", "RBLN"]

    # To support an accelerator's CCL,
    # the `_new_process_group_helper` functions of `mult_channel_pg` need to be implemented
    CCL_SUPPORT_LIST = [AcceleratorType.NV_GPU, AcceleratorType.AMD_GPU]

    @staticmethod
    def get_node_accelerator_type_and_num(node_info: dict[str, int]):
        """Get the type of accelerator and num of accelerators available on the current node.

        Args:
            node_info (Dict[str, int]): A dictionary containing the resources of the node. This dictionary is obtained via `ray.nodes()`.
            The resource keys are documented at https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html.

        Returns:
            Tuple[AcceleratorType, int]: A tuple containing the type of accelerator and num of accelerators.
        """
        node_resources: dict[str, str] = node_info.get("Resources", {})

        for unsupported in Accelerator.UNSUPPORTED_LIST:
            if unsupported in node_resources and node_resources[unsupported] > 0:
                warnings.warn(
                    f"Unsupported accelerator type {unsupported} detected on node {node_info}"
                )

        for supported in Accelerator.SUPPORTED_LIST:
            if supported in node_resources and node_resources[supported] > 0:
                if supported == "GPU":
                    # Find the accelerator_type to distinguish NV/AMD/Intel GPUs
                    # accelerator_type is a key in node_resources which starts with "accelerator_type"
                    accelerator_type = AcceleratorType.NV_GPU
                    for key in node_resources.keys():
                        if key.startswith("accelerator_type"):
                            if "AMD" in key:
                                # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L12
                                accelerator_type = AcceleratorType.AMD_GPU
                            elif "INTEL" in key:
                                # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L82
                                accelerator_type = AcceleratorType.INTEL_GPU
                            break
                elif supported == "NPU":
                    accelerator_type = AcceleratorType.NPU
                return accelerator_type, int(node_resources[supported])
        return AcceleratorType.NO_ACCEL, 0

    @staticmethod
    def get_accelerator_env_var(
        accelerator_type: AcceleratorType, visible_accelerators: list[str]
    ) -> dict[str, str]:
        """Get the environment variables related to the accelerator.

        Args:
            accelerator_type (AcceleratorType): The type of the accelerator.
            visible_accelerators (List[str]): A list of visible accelerator IDs.

        Returns:
            Dict[str, str]: A dictionary containing the accelerator environment variables.
        """
        env_vars = {}
        visible_accelerators = ",".join(visible_accelerators)
        if accelerator_type == AcceleratorType.NV_GPU:
            # All the three types of GPU can be set together
            env_vars["CUDA_VISIBLE_DEVICES"] = visible_accelerators
            # Override Ray's control over GPU assignment
            env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
            # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96

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

        elif accelerator_type == AcceleratorType.AMD_GPU:
            env_vars["ROCR_VISIBLE_DEVICES"] = visible_accelerators
            env_vars["RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES"] = "1"
            # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L99

        elif accelerator_type == AcceleratorType.INTEL_GPU:
            env_vars["ONEAPI_DEVICE_SELECTOR"] = visible_accelerators
            env_vars["RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR"] = "1"
            # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L94

        elif accelerator_type == AcceleratorType.NPU:
            env_vars["ASCEND_RT_VISIBLE_DEVICES"] = visible_accelerators
            env_vars["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"
            # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L91

        elif accelerator_type != AcceleratorType.NO_ACCEL:
            raise ValueError(f"Unsupported accelerator type: {accelerator_type}")

        return env_vars

    @staticmethod
    def get_visible_devices(accelerator_type: AcceleratorType) -> list[int]:
        """Get the visible device environment variable based on accelerator type.

        Args:
            accelerator_type (AcceleratorType): The type of the accelerator.

        Returns:
            List[int]: A list of visible device IDs.

        """
        if accelerator_type == AcceleratorType.NV_GPU:
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        elif accelerator_type == AcceleratorType.AMD_GPU:
            visible_devices = os.environ.get("ROCR_VISIBLE_DEVICES")
        elif accelerator_type == AcceleratorType.INTEL_GPU:
            visible_devices = os.environ.get("ONEAPI_DEVICE_SELECTOR")
        elif accelerator_type == AcceleratorType.NPU:
            visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        elif accelerator_type == AcceleratorType.NO_ACCEL:
            visible_devices = None

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
    def get_ccl_backend(accelerator_type: AcceleratorType) -> str:
        """Get the CCL backend based on the accelerator type.

        Args:
            accelerator_type (AcceleratorType): The type of the accelerator.

        Returns:
            str: The CCL backend.
        """
        if accelerator_type == AcceleratorType.NV_GPU:
            return "nccl"
        elif accelerator_type == AcceleratorType.AMD_GPU:
            return "nccl"
        elif accelerator_type == AcceleratorType.INTEL_GPU:
            return "ccl"
        elif accelerator_type == AcceleratorType.NPU:
            return "hccl"
        elif accelerator_type == AcceleratorType.NO_ACCEL:
            return None
        raise ValueError(f"Unsupported accelerator type: {accelerator_type}")

    @staticmethod
    def get_torch_platform(accelerator_type: AcceleratorType):
        """Get the PyTorch platform module based on the accelerator type."""
        import torch

        if accelerator_type == AcceleratorType.NV_GPU:
            return torch.cuda
        elif accelerator_type == AcceleratorType.AMD_GPU:
            return torch.cuda
        elif accelerator_type == AcceleratorType.INTEL_GPU:
            return torch.xpu
        elif accelerator_type == AcceleratorType.NPU:
            return torch.npu
        elif accelerator_type == AcceleratorType.NO_ACCEL:
            return None
        raise ValueError(f"Unsupported accelerator type: {accelerator_type}")

    @staticmethod
    def get_device_type(accelerator_type: AcceleratorType) -> str:
        """Get the device type based on the accelerator type."""
        if accelerator_type == AcceleratorType.NV_GPU:
            return "cuda"
        elif accelerator_type == AcceleratorType.AMD_GPU:
            return "cuda"
        elif accelerator_type == AcceleratorType.INTEL_GPU:
            return "xpu"
        elif accelerator_type == AcceleratorType.NPU:
            return "npu"
        elif accelerator_type == AcceleratorType.NO_ACCEL:
            return None
        raise ValueError(f"Unsupported accelerator type: {accelerator_type}")
