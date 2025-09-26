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

"""
简单的Tool Agent Loop使用示例

这个示例展示了如何在rlinf中使用简单的tool agent loop。
"""

import asyncio
import logging
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from .tool_agent_loop import ToolAgentLoop

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """主函数 - 演示如何使用Tool Agent Loop"""
    
    # 1. 创建配置
    config = OmegaConf.create({
        "rollout": {
            "max_user_turns": 5,           # 最大用户轮数
            "max_assistant_turns": 5,      # 最大助手轮数
            "max_parallel_calls": 3,       # 最大并行工具调用数
            "max_tool_response_length": 500,  # 工具响应最大长度
            "tool_response_truncate_side": "right",  # 截断方向
            "response_length": 1024        # 响应最大长度
        },
        "data": {
            "apply_chat_template_kwargs": {}  # 聊天模板参数
        }
    })
    
    # 2. 初始化tokenizer
    try:
        # 使用一个轻量级的模型进行演示
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("✅ Tokenizer 初始化成功")
    except Exception as e:
        logger.error(f"❌ Tokenizer 初始化失败: {e}")
        return
    
    # 3. 创建Tool Agent Loop
    agent = ToolAgentLoop(config, tokenizer)
    logger.info("✅ Tool Agent Loop 创建成功")
    
    # 4. 准备测试用例
    test_cases = [
        {
            "name": "计算器测试",
            "messages": [
                {"role": "user", "content": "请帮我计算 15 * 8 + 3 的结果"}
            ]
        },
        {
            "name": "搜索测试", 
            "messages": [
                {"role": "user", "content": "请搜索关于人工智能的最新信息"}
            ]
        },
        {
            "name": "天气查询测试",
            "messages": [
                {"role": "user", "content": "请查询北京的天气情况"}
            ]
        }
    ]
    
    # 5. 运行测试用例
    sampling_params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 150
    }
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"测试用例 {i}: {test_case['name']}")
        logger.info(f"{'='*50}")
        
        try:
            # 运行agent loop
            result = await agent.run(
                sampling_params=sampling_params,
                raw_prompt=test_case['messages']
            )
            
            # 显示结果
            logger.info(f"📝 用户输入: {test_case['messages'][0]['content']}")
            logger.info(f"🤖 助手响应: {result.response_text}")
            logger.info(f"🔄 对话轮数: {result.num_turns}")
            logger.info(f"📊 Token统计: Prompt={len(result.prompt_ids)}, Response={len(result.response_ids)}")
            logger.info(f"🎯 响应掩码: {result.response_mask[:10]}..." if len(result.response_mask) > 10 else f"🎯 响应掩码: {result.response_mask}")
            
        except Exception as e:
            logger.error(f"❌ 测试用例 {i} 失败: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info("🎉 所有测试完成!")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())
