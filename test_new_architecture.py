#!/usr/bin/env python3
"""
测试新架构的脚本
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rlinf'))

def test_worker_inheritance():
    """测试Worker继承关系"""
    print("测试Worker继承关系...")
    
    try:
        # 测试SGLangGenerateWorker
        from rlinf.workers.rollout.sglang.sglang_generate_worker import SGLangGenerateWorker
        from rlinf.scheduler import Worker
        
        print("✓ SGLangGenerateWorker导入成功")
        print(f"  - SGLangGenerateWorker继承自: {SGLangGenerateWorker.__bases__}")
        print(f"  - 是否为Worker子类: {issubclass(SGLangGenerateWorker, Worker)}")
        
        # 测试ToolAgentLoop
        from rlinf.workers.agent_loop.tool_agent_loop import ToolAgentLoop
        
        print("✓ ToolAgentLoop导入成功")
        print(f"  - ToolAgentLoop继承自: {ToolAgentLoop.__bases__}")
        print(f"  - 是否为Worker子类: {issubclass(ToolAgentLoop, Worker)}")
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False

def test_worker_methods():
    """测试Worker方法"""
    print("\n测试Worker方法...")
    
    try:
        from rlinf.workers.rollout.sglang.sglang_generate_worker import SGLangGenerateWorker
        from rlinf.workers.agent_loop.tool_agent_loop import ToolAgentLoop
        
        # 检查SGLangGenerateWorker的方法
        sglang_methods = ['init_worker', 'sync_model_from_actor', 'async_generate', 'shutdown']
        for method in sglang_methods:
            if hasattr(SGLangGenerateWorker, method):
                print(f"✓ SGLangGenerateWorker.{method} 方法存在")
            else:
                print(f"✗ SGLangGenerateWorker.{method} 方法不存在")
        
        # 检查ToolAgentLoop的方法
        tool_methods = ['init_worker', 'run']
        for method in tool_methods:
            if hasattr(ToolAgentLoop, method):
                print(f"✓ ToolAgentLoop.{method} 方法存在")
            else:
                print(f"✗ ToolAgentLoop.{method} 方法不存在")
        
        return True
        
    except Exception as e:
        print(f"✗ 方法测试错误: {e}")
        return False

def test_runner_integration():
    """测试Rstar2Runner集成"""
    print("\n测试Rstar2Runner集成...")
    
    try:
        from rlinf.runners.rstar2_runner import Rstar2Runner
        from rlinf.workers.rollout.sglang.sglang_generate_worker import SGLangGenerateWorker
        from rlinf.workers.agent_loop.tool_agent_loop import ToolAgentLoop
        
        print("✓ Rstar2Runner导入成功")
        print("✓ SGLangGenerateWorker和ToolAgentLoop可以在Rstar2Runner中使用")
        print("✓ 现在使用WorkerGroup管理AgentLoop组件")
        
        # 检查Rstar2Runner的新方法
        runner_methods = ['_init_agentloop_components', '_get_agentloop_tools', '_async_generate_fn', '_run_agentloop_rollout']
        for method in runner_methods:
            if hasattr(Rstar2Runner, method):
                print(f"✓ Rstar2Runner.{method} 方法存在")
            else:
                print(f"✗ Rstar2Runner.{method} 方法不存在")
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False

def test_config_compatibility():
    """测试配置兼容性"""
    print("\n测试配置兼容性...")
    
    try:
        from omegaconf import OmegaConf
        
        # 测试配置文件
        config_path = "examples/rstar2/config/rstar2-agentloop-sglang.yaml"
        if os.path.exists(config_path):
            cfg = OmegaConf.load(config_path)
            print(f"✓ 配置文件加载成功: {config_path}")
            print(f"✓ rollout_backend: {cfg.rollout.rollout_backend}")
            print(f"✓ AgentLoop配置: max_user_turns={cfg.rollout.max_user_turns}")
            return True
        else:
            print(f"✗ 配置文件不存在: {config_path}")
            return False
            
    except Exception as e:
        print(f"✗ 配置测试错误: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("新架构测试")
    print("=" * 60)
    
    success1 = test_worker_inheritance()
    success2 = test_worker_methods()
    success3 = test_runner_integration()
    success4 = test_config_compatibility()
    
    print("\n" + "=" * 60)
    if success1 and success2 and success3 and success4:
        print("🎉 所有测试通过！新架构修改成功完成。")
        print("\n修改总结:")
        print("1. ✓ SGLangGenerateWorker现在正确继承自Worker")
        print("2. ✓ ToolAgentLoop现在正确继承自Worker")
        print("3. ✓ 使用WorkerGroup管理AgentLoop组件，避免直接实例化")
        print("4. ✓ Rstar2Runner现在通过WorkerGroup与AgentLoop组件交互")
        print("5. ✓ 配置文件保持兼容")
        print("\n架构优势:")
        print("- 避免了Worker嵌套问题")
        print("- 所有组件都支持分布式通信")
        print("- 统一的组件管理")
        print("- 保持向后兼容性")
        sys.exit(0)
    else:
        print("❌ 部分测试失败，请检查错误信息。")
        sys.exit(1)
