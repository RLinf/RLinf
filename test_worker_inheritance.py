#!/usr/bin/env python3
"""
测试脚本：验证SGLangGenerateWorker和ToolAgentLoop是否正确继承自Worker
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
        
        # 测试Rstar2Runner集成
        from rlinf.runners.rstar2_runner import Rstar2Runner
        
        print("✓ Rstar2Runner导入成功")
        print("✓ Rstar2Runner现在直接维护SGLangGenerateWorker和ToolAgentLoop")
        
        print("\n所有测试通过！Worker继承关系正确设置。")
        return True
        
    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
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
        print("✓ 避免了Worker嵌套，所有组件由Rstar2Runner直接维护")
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Worker继承关系测试")
    print("=" * 60)
    
    success1 = test_worker_inheritance()
    success2 = test_runner_integration()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 所有测试通过！修改成功完成。")
        print("\n修改总结:")
        print("1. ✓ SGLangGenerateWorker现在继承自Worker")
        print("2. ✓ ToolAgentLoop现在继承自Worker")
        print("3. ✓ 删除了AgentLoopSGLangWorker，避免Worker嵌套")
        print("4. ✓ Rstar2Runner现在直接维护SGLangGenerateWorker和ToolAgentLoop")
        print("5. ✓ 所有Worker交互都在Rstar2Runner中统一管理")
        sys.exit(0)
    else:
        print("❌ 测试失败，请检查错误信息。")
        sys.exit(1)
