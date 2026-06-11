// Copyright (c) 2024 Alibaba PAI, ColossalAI and Nvidia Megatron-LM Team.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// modified from
// https://github.com/hpcaitech/ColossalAI/blob/main/extensions/pybind/optimizer/optimizer.cpp
#include <torch/extension.h>

void multi_tensor_adam_cuda(int chunk_size, at::Tensor noop_flag,
                            std::vector<std::vector<at::Tensor>> tensor_lists,
                            const float lr, const float beta1,
                            const float beta2, const float epsilon,
                            const int step, const int mode,
                            const int bias_correction, const float weight_decay,
                            const float div_scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multi_tensor_adam", &multi_tensor_adam_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer");
}
