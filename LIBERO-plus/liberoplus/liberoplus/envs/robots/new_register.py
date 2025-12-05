# 生成批量注册代码的脚本

# 生成MountedPanda的导入和注册代码
mounted_imports = ["MountedPanda"]
mounted_registry = ['"MountedPanda": SingleArm']

# 生成OnTheGroundPanda的导入和注册代码
ground_imports = ["OnTheGroundPanda"]
ground_registry = ['"OnTheGroundPanda": SingleArm']

# 为1到49号变体生成代码
for i in range(1, 50):
    mounted_imports.append(f"MountedPanda{i}")
    mounted_registry.append(f'"MountedPanda{i}": SingleArm')
    
    ground_imports.append(f"OnTheGroundPanda{i}")
    ground_registry.append(f'"OnTheGroundPanda{i}": SingleArm')

# 生成完整的导入语句
import_statement = f"from .mounted_panda import {', '.join(mounted_imports)}\n"
import_statement += f"from .on_the_ground_panda import {', '.join(ground_imports)}\n\n"

# 生成注册代码 - 修复f-string中的反斜杠问题
registry_dict_content = ',\n        '.join(mounted_registry + ground_registry)
registry_code = f"""from robosuite.robots.single_arm import SingleArm
from robosuite.robots import ROBOT_CLASS_MAPPING

ROBOT_CLASS_MAPPING.update(
    {{
        {registry_dict_content}
    }}
)"""

# 组合完整代码
full_code = import_statement + registry_code

# 写入文件
with open('robot_registry_code.txt', 'w', encoding='utf-8') as f:
    f.write(full_code)

print("已生成 robot_registry_code.txt 文件，包含批量注册代码")
