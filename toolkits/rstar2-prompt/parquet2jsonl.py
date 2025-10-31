#!/usr/bin/env python3
"""
Parquet to JSONL converter with complete ninja template application
将parquet文件转换为应用了完整ninja template的JSONL格式
"""

import json
import yaml
import pandas as pd
import numpy as np
import argparse
import ast
from pathlib import Path
from typing import Dict, List, Optional, Any

def load_tools_from_yaml(yaml_file: str) -> List[Dict]:
    """
    从YAML文件加载工具定义
    
    Args:
        yaml_file: YAML配置文件路径
        
    Returns:
        工具定义列表
    """
    with open(yaml_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    tools = []
    if 'tools' in config:
        for tool_config in config['tools']:
            if 'tool_schema' in tool_config:
                tools.append(tool_config['tool_schema'])
    
    return tools

def extract_content_from_source_prompt(source_prompt: Any) -> str:
    """
    从source_prompt中提取content字段
    
    Args:
        source_prompt: 原始的source_prompt，可能是字符串、列表、字典或numpy array
        
    Returns:
        提取的content内容
    """
    # 处理 numpy array - 先转换为Python原生类型
    if isinstance(source_prompt, np.ndarray):
        # 如果是numpy array，转换为字符串
        source_prompt = str(source_prompt)
    
    # 如果已经是列表或字典，直接处理
    if isinstance(source_prompt, list):
        for message in source_prompt:
            if isinstance(message, dict) and 'content' in message:
                if message.get('role') == 'user':
                    return message['content']
        # 如果没找到user消息，返回第一个有content的消息
        for message in source_prompt:
            if isinstance(message, dict) and 'content' in message:
                return message['content']
    
    if isinstance(source_prompt, dict) and 'content' in source_prompt:
        return source_prompt['content']
    
    # 如果是字符串，尝试解析
    if isinstance(source_prompt, str):
        # 尝试用json解析
        try:
            prompt_data = json.loads(source_prompt)
            # 递归调用处理解析后的数据
            return extract_content_from_source_prompt(prompt_data)
        except json.JSONDecodeError:
            pass
        
        # 尝试用ast.literal_eval解析（处理Python字面量）
        try:
            prompt_data = ast.literal_eval(source_prompt)
            # 递归调用处理解析后的数据
            return extract_content_from_source_prompt(prompt_data)
        except (ValueError, SyntaxError):
            pass
    
    # 如果所有解析都失败，返回原字符串
    return str(source_prompt)

def apply_ninja_template(source_prompt: Any, solution: str, tools: Optional[List[Dict]] = None) -> str:
    """
    应用完整的ninja template到source_prompt
    
    Args:
        source_prompt: 原始问题（可能包含JSON格式的消息）
        solution: 解答
        tools: 工具定义列表（可选）
        
    Returns:
        格式化后的prompt
    """
    # 从source_prompt中提取实际的问题内容
    content = extract_content_from_source_prompt(source_prompt)
    
    # 构建消息列表
    messages = []
    
    # System message - 根据是否有tools决定内容
    if tools:
        # 有tools时的system消息
        system_content = 'A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <reason> </reason> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <reason> reasoning process here </reason> <answer> answer here </answer>.'
        
        # 添加tools信息
        tools_section = "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        
        for tool in tools:
            tools_section += "\n" + json.dumps(tool)
        
        tools_section += "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
        
        system_message = f"<|im_start|>system\n{system_content}{tools_section}<|im_end|>\n"
    else:
        # 没有tools时的system消息
        system_message = '<|im_start|>system\nA conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <reason> </reason> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <reason> reasoning process here </reason> <answer> answer here </answer>.<|im_end|>\n'
    
    # User message - 添加特定的前缀和问题内容
    user_prefix = 'You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.\nThis is the problem:\n'
    user_message = f'<|im_start|>user\n{user_prefix}{content}<|im_end|>\n'
    
    # Assistant start - 以<reason>标记开始
    assistant_start = '<|im_start|>assistant\n<reason>'
    
    # 组合完整的prompt
    formatted_prompt = system_message + user_message + assistant_start
    
    return formatted_prompt

def convert_parquet_to_jsonl(input_file: str, output_file: Optional[str] = None, 
                             tools_yaml: Optional[str] = None, verbose: bool = False):
    """
    将parquet文件转换为JSONL格式，应用完整的ninja template
    
    Args:
        input_file: 输入的parquet文件路径
        output_file: 输出的jsonl文件路径（可选）
        tools_yaml: 工具定义的YAML文件路径（可选）
        verbose: 是否显示详细信息
    """
    # 加载工具定义（如果提供）
    tools = None
    if tools_yaml:
        try:
            tools = load_tools_from_yaml(tools_yaml)
            print(f"从 {tools_yaml} 加载了 {len(tools)} 个工具定义")
        except Exception as e:
            print(f"警告：无法加载工具定义文件 {tools_yaml}: {e}")
            print("将继续处理，但不包含工具信息")
    
    # 读取parquet文件
    try:
        df = pd.read_parquet(input_file)
    except Exception as e:
        print(f"错误：无法读取parquet文件 {input_file}")
        print(f"错误详情：{e}")
        return
    
    # 检查必需的列
    required_columns = ['source_prompt', 'solution']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"错误：parquet文件缺少必需的列：{missing_columns}")
        print(f"文件中的列：{list(df.columns)}")
        return
    
    # 如果没有指定输出文件，自动生成
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.with_suffix('.jsonl')
    
    # 转换并写入JSONL
    successful_count = 0
    failed_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            try:
                # 提取content用于调试
                if verbose and idx < 3:
                    print(f"\n=== 示例 {idx + 1} ===")
                    print(f"原始 source_prompt 类型: {type(row['source_prompt'])}")
                    
                    # 显示原始内容
                    if isinstance(row['source_prompt'], np.ndarray):
                        print(f"numpy array 内容: {str(row['source_prompt'])[:200]}")
                    
                    content = extract_content_from_source_prompt(row['source_prompt'])
                    print(f"提取的 content 前200字符: {content[:200]}...")
                
                
                # 创建JSON对象
                json_obj = {
                    "prompt": list(row['source_prompt']),
                    "task": row.get('task', 'math'),
                    "query_id": row.get('query_id', f"{idx:08d}"),
                    "solutions": [row['solution']] if isinstance(row['solution'], str) else row['solution']
                }
                
                # 如果有工具信息，也可以添加到JSON对象中（可选）
                if tools:
                    json_obj["tools_available"] = True
                    json_obj["tools_count"] = len(tools)
                
                # 写入一行JSONL
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                successful_count += 1
                
            except Exception as e:
                print(f"警告：处理第 {idx} 行时出错: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                failed_count += 1
    
    print(f"\n转换完成！")
    print(f"输入文件：{input_file}")
    print(f"输出文件：{output_file}")
    print(f"成功转换：{successful_count} 条记录")
    if failed_count > 0:
        print(f"失败：{failed_count} 条记录")
    if tools:
        print(f"应用了 {len(tools)} 个工具定义")

def main():
    parser = argparse.ArgumentParser(
        description='将parquet文件转换为应用了完整ninja template的JSONL格式'
    )
    parser.add_argument('input_file', help='输入的parquet文件路径')
    parser.add_argument('-o', '--output', help='输出的jsonl文件路径（可选，默认为同名.jsonl文件）')

    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    
    args = parser.parse_args()
    
    convert_parquet_to_jsonl(args.input_file, args.output, args.verbose)

if __name__ == "__main__":
    main()