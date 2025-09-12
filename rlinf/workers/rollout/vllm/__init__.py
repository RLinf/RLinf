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


from importlib.metadata import PackageNotFoundError, version

from packaging.version import parse


def get_version(pkg):
    try:
        return parse(version(pkg))
    except PackageNotFoundError:
        return None


package_name = "vllm"
package_version = get_version(package_name)


if package_version is None:
    raise ValueError(
        "vllm package is not installed or its version could not be determined."
    )
elif package_version >= parse("0.8.5") and package_version < parse("0.9.0"):
    from rlinf.hybrid_engines.vllm.vllm_0_8_5.vllm_engine import VLLMEngine
else:
    raise ValueError(f"vllm version {package_version} not supported")

__all__ = ["VLLMEngine"]
