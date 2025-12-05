import os

def rename_items_recursive(root_path):
    # 先处理子目录，再处理当前目录的文件，避免路径问题
    # 首先收集所有子目录
    subdirectories = []
    for item in os.listdir(root_path):
        item_path = os.path.join(root_path, item)
        if os.path.isdir(item_path):
            subdirectories.append(item)
    
    # 递归处理子目录
    for subdir in subdirectories:
        subdir_path = os.path.join(root_path, subdir)
        rename_items_recursive(subdir_path)
    
    # 处理当前目录中的所有项目（文件和文件夹）
    for item in os.listdir(root_path):
        old_path = os.path.join(root_path, item)
        
        # 检查名称中是否包含"wooden"
        if "wooden" in item:
            # 替换名称中的"wooden"为"yellow"
            new_name = item.replace("wooden", "yellow")
            new_path = os.path.join(root_path, new_name)
            
            # 避免重命名到相同的名称
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"已重命名: {old_path} -> {new_path}")

if __name__ == "__main__":
    # 指定要处理的根文件夹路径
    root_folder = "./yellow_cabinet"
    
    # 检查根文件夹是否存在
    if not os.path.exists(root_folder):
        print(f"错误：根文件夹 '{root_folder}' 不存在")
    else:
        # 调用递归重命名函数
        rename_items_recursive(root_folder)
        print("所有重命名操作完成")
