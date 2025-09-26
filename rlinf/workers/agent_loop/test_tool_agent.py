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

import asyncio
import logging
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from rlinf.workers.agent_loop.tool_agent_loop import ToolAgentLoop
from rlinf.tools.code_judge_tool import CodeJudgeTool

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_tool_agent():
    """测试简单的tool agent loop"""
    
    # 创建配置
    config = OmegaConf.create({
        "rollout": {
            "max_user_turns": 3,
            "max_assistant_turns": 3,
            "max_parallel_calls": 2,
            "max_tool_response_length": 200,
            "tool_response_truncate_side": "right",
            "response_length": 512
        },
        "data": {
            "apply_chat_template_kwargs": {}
        }
    })
    
    # 初始化tokenizer (使用一个简单的模型)
    try:
        tokenizer = AutoTokenizer.from_pretrained("/mnt/mnt/public/xyq_ckpt/DeepSeek-R1-Distill-Qwen-7B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"无法加载tokenizer: {e}")
        return
    
    # 创建并注入 rstar2 风格工具（调用外部 code-judge 服务）
    python_tool = CodeJudgeTool(
        name="python_code_with_standard_io",
        host_addr="localhost",
        host_port="8088",
        batch_size=4,
        concurrency=2,
        batch_timeout_seconds=0.2,
    )
    tools = {"python_code_with_standard_io": python_tool}

    # 创建tool agent（注入工具）
    agent = ToolAgentLoop(config, tokenizer, tools=tools)
    
    # 测试数据
    test_messages = [
        {"role": "user", "content": "请帮我计算 2+3 的结果"}
    ]
    
    sampling_params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 100
    }
    
    try:
        # 运行agent loop
        result = await agent.run(
            sampling_params=sampling_params,
            raw_prompt=test_messages
        )
        
        # 打印结果
        logger.info("=== Tool Agent Loop 测试结果 ===")
        logger.info(f"Prompt IDs 长度: {len(result.prompt_ids)}")
        logger.info(f"Response IDs 长度: {len(result.response_ids)}")
        logger.info(f"Response Mask: {result.response_mask}")
        logger.info(f"对话轮数: {result.num_turns}")
        logger.info(f"Prompt 文本: {result.prompt_text}")
        logger.info(f"Response 文本: {result.response_text}")
        
        # 验证结果
        assert len(result.prompt_ids) > 0, "Prompt IDs 不应为空"
        assert len(result.response_ids) > 0, "Response IDs 不应为空"
        assert len(result.response_mask) == len(result.response_ids), "Response mask 长度应与 response IDs 一致"
        assert result.num_turns > 0, "对话轮数应大于0"
        
        logger.info("✅ 所有测试通过!")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(test_tool_agent())
