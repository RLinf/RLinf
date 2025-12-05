import numpy as np

# 设置随机种子以确保可重复性
np.random.seed(42)

# 生成49个具有不同初始状态的MountedPanda类定义
mounted_class_definitions = []
ground_class_definitions = []

for i in range(1, 101):
    # 生成唯一的初始位置
    original_qpos = np.array([0.0, -1.61037389e-01, 0.00, -2.44459747e00, 0.00, 2.22675220e00, np.pi / 4])
    perturbation = np.random.randn(len(original_qpos))
    perturbation = perturbation / np.linalg.norm(perturbation)
    perturbed_qpos = original_qpos + perturbation * 0.1
    
    # 格式化qpos数组为字符串
    qpos_str = np.array2string(perturbed_qpos, separator=', ', precision=8, suppress_small=True)
    
    # 创建MountedPanda类定义
    mounted_class_def = f"""
class MountedPanda{i}(MountedPanda):
    \"\"\"
    Panda Robot New Init State {i}
    \"\"\"

    @property
    def init_qpos(self):
        return np.array({qpos_str})
"""
    mounted_class_definitions.append(mounted_class_def)
    
    # 创建OnTheGroundPanda类定义
    ground_class_def = f"""
class OnTheGroundPanda{i}(OnTheGroundPanda):
    \"\"\"
    Panda Robot New Init State {i}
    \"\"\"

    @property
    def init_qpos(self):
        return np.array({qpos_str})
"""
    ground_class_definitions.append(ground_class_def)

for i in range(101, 201):
    # 生成唯一的初始位置
    original_qpos = np.array([0.0, -1.61037389e-01, 0.00, -2.44459747e00, 0.00, 2.22675220e00, np.pi / 4])
    perturbation = np.random.randn(len(original_qpos))
    perturbation = perturbation / np.linalg.norm(perturbation)
    perturbed_qpos = original_qpos + perturbation * 0.2
    
    # 格式化qpos数组为字符串
    qpos_str = np.array2string(perturbed_qpos, separator=', ', precision=8, suppress_small=True)
    
    # 创建MountedPanda类定义
    mounted_class_def = f"""
class MountedPanda{i}(MountedPanda):
    \"\"\"
    Panda Robot New Init State {i}
    \"\"\"

    @property
    def init_qpos(self):
        return np.array({qpos_str})
"""
    mounted_class_definitions.append(mounted_class_def)
    
    # 创建OnTheGroundPanda类定义
    ground_class_def = f"""
class OnTheGroundPanda{i}(OnTheGroundPanda):
    \"\"\"
    Panda Robot New Init State {i}
    \"\"\"

    @property
    def init_qpos(self):
        return np.array({qpos_str})
"""
    ground_class_definitions.append(ground_class_def)


for i in range(201, 301):
    # 生成唯一的初始位置
    original_qpos = np.array([0.0, -1.61037389e-01, 0.00, -2.44459747e00, 0.00, 2.22675220e00, np.pi / 4])
    perturbation = np.random.randn(len(original_qpos))
    perturbation = perturbation / np.linalg.norm(perturbation)
    perturbed_qpos = original_qpos + perturbation * 0.3
    
    # 格式化qpos数组为字符串
    qpos_str = np.array2string(perturbed_qpos, separator=', ', precision=8, suppress_small=True)
    
    # 创建MountedPanda类定义
    mounted_class_def = f"""
class MountedPanda{i}(MountedPanda):
    \"\"\"
    Panda Robot New Init State {i}
    \"\"\"

    @property
    def init_qpos(self):
        return np.array({qpos_str})
"""
    mounted_class_definitions.append(mounted_class_def)
    
    # 创建OnTheGroundPanda类定义
    ground_class_def = f"""
class OnTheGroundPanda{i}(OnTheGroundPanda):
    \"\"\"
    Panda Robot New Init State {i}
    \"\"\"

    @property
    def init_qpos(self):
        return np.array({qpos_str})
"""
    ground_class_definitions.append(ground_class_def)

for i in range(301, 401):
    # 生成唯一的初始位置
    original_qpos = np.array([0.0, -1.61037389e-01, 0.00, -2.44459747e00, 0.00, 2.22675220e00, np.pi / 4])
    perturbation = np.random.randn(len(original_qpos))
    perturbation = perturbation / np.linalg.norm(perturbation)
    perturbed_qpos = original_qpos + perturbation * 0.4
    
    # 格式化qpos数组为字符串
    qpos_str = np.array2string(perturbed_qpos, separator=', ', precision=8, suppress_small=True)
    
    # 创建MountedPanda类定义
    mounted_class_def = f"""
class MountedPanda{i}(MountedPanda):
    \"\"\"
    Panda Robot New Init State {i}
    \"\"\"

    @property
    def init_qpos(self):
        return np.array({qpos_str})
"""
    mounted_class_definitions.append(mounted_class_def)
    
    # 创建OnTheGroundPanda类定义
    ground_class_def = f"""
class OnTheGroundPanda{i}(OnTheGroundPanda):
    \"\"\"
    Panda Robot New Init State {i}
    \"\"\"

    @property
    def init_qpos(self):
        return np.array({qpos_str})
"""
    ground_class_definitions.append(ground_class_def)

for i in range(401, 501):
    # 生成唯一的初始位置
    original_qpos = np.array([0.0, -1.61037389e-01, 0.00, -2.44459747e00, 0.00, 2.22675220e00, np.pi / 4])
    perturbation = np.random.randn(len(original_qpos))
    perturbation = perturbation / np.linalg.norm(perturbation)
    perturbed_qpos = original_qpos + perturbation * 0.5
    
    # 格式化qpos数组为字符串
    qpos_str = np.array2string(perturbed_qpos, separator=', ', precision=8, suppress_small=True)
    
    # 创建MountedPanda类定义
    mounted_class_def = f"""
class MountedPanda{i}(MountedPanda):
    \"\"\"
    Panda Robot New Init State {i}
    \"\"\"

    @property
    def init_qpos(self):
        return np.array({qpos_str})
"""
    mounted_class_definitions.append(mounted_class_def)
    
    # 创建OnTheGroundPanda类定义
    ground_class_def = f"""
class OnTheGroundPanda{i}(OnTheGroundPanda):
    \"\"\"
    Panda Robot New Init State {i}
    \"\"\"

    @property
    def init_qpos(self):
        return np.array({qpos_str})
"""
    ground_class_definitions.append(ground_class_def)



# 将MountedPanda类定义写入txt文件
with open('mounted_panda_variants.txt', 'w', encoding='utf-8') as f:
    f.write("import numpy as np\n")
    f.write("from robosuite.models.robots.manipulators.panda import MountedPanda\n\n")
    for class_def in mounted_class_definitions:
        f.write(class_def)
        f.write("\n")

print("已生成 mounted_panda_variants.txt 文件，包含500个MountedPanda变体类")

# 将OnTheGroundPanda类定义写入txt文件
with open('ground_panda_variants.txt', 'w', encoding='utf-8') as f:
    f.write("import numpy as np\n")
    f.write("from robosuite.models.robots.manipulators.panda import OnTheGroundPanda\n\n")
    for class_def in ground_class_definitions:
        f.write(class_def)
        f.write("\n")

print("已生成 ground_panda_variants.txt 文件，包含500个OnTheGroundPanda变体类")