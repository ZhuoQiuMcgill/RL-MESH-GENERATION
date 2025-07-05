#!/usr/bin/env python3
"""
项目设置脚本

该脚本用于初始化项目目录结构，创建必要的文件夹和示例数据文件。
根据config.yaml中的路径配置来创建目录结构。
"""

import os
import sys
from typing import List, Tuple

# 添加项目根目录到路径
sys.path.insert(0, os.getcwd())

from src.rl.config import load_config
from src.utils import MeshImporter, create_default_importer


def create_directory_structure():
    """
    根据配置文件创建项目目录结构
    """
    print("正在设置项目目录结构...")

    # 加载配置
    config = load_config()
    paths_config = config.get("paths", {})

    # 获取项目根目录
    root_dir = os.getcwd()

    # 创建主要目录
    main_dirs = [
        "data_root",
        "cache_dir",
        "temp_dir",
        "tools_dir",
        "config_dir"
    ]

    created_dirs = []

    for dir_key in main_dirs:
        if dir_key in paths_config:
            dir_path = os.path.join(root_dir, paths_config[dir_key])
            os.makedirs(dir_path, exist_ok=True)
            created_dirs.append(dir_path)
            print(f"✓ 创建目录: {dir_path}")

    # 创建数据子目录
    data_root = os.path.join(root_dir, paths_config.get("data_root", "data"))
    data_subdirs = [
        "mesh_dir",
        "logs_dir",
        "models_dir",
        "results_dir",
    ]

    for subdir_key in data_subdirs:
        if subdir_key in paths_config:
            subdir_path = os.path.join(data_root, paths_config[subdir_key])
            os.makedirs(subdir_path, exist_ok=True)
            created_dirs.append(subdir_path)
            print(f"✓ 创建数据子目录: {subdir_path}")

    print(f"\n总共创建了 {len(created_dirs)} 个目录")
    return created_dirs


def create_sample_mesh_files():
    """
    创建示例mesh文件
    """
    print("\n正在创建示例mesh文件...")

    # 创建导入器
    importer = create_default_importer()

    # 定义示例mesh数据
    sample_meshes = [
        ("simple_square", [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]),
        ("triangle", [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]),
        ("rectangle", [(0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0)]),
        ("pentagon", [
            (0.0, 1.0), (0.95, 0.31), (0.59, -0.81),
            (-0.59, -0.81), (-0.95, 0.31)
        ]),
        ("hexagon", [
            (1.0, 0.0), (0.5, 0.87), (-0.5, 0.87),
            (-1.0, 0.0), (-0.5, -0.87), (0.5, -0.87)
        ])
    ]

    created_files = []

    for mesh_name, vertices in sample_meshes:
        try:
            file_path = importer.create_sample_mesh_file(
                mesh_name=mesh_name,
                vertices=vertices,
                subfolder="mesh",
                overwrite=True
            )
            created_files.append(file_path)
            print(f"✓ 创建mesh文件: {mesh_name}.txt ({len(vertices)} 个顶点)")
        except Exception as e:
            print(f"✗ 创建 {mesh_name}.txt 失败: {e}")

    # 创建一些复杂示例到examples目录
    complex_examples = [
        ("l_shape", [
            (0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0),
            (1.0, 2.0), (0.0, 2.0)
        ]),
        ("star_shape", [
            (0.0, 1.0), (0.22, 0.31), (0.95, 0.31), (0.36, -0.12),
            (0.59, -0.81), (0.0, -0.5), (-0.59, -0.81), (-0.36, -0.12),
            (-0.95, 0.31), (-0.22, 0.31)
        ])
    ]

    for mesh_name, vertices in complex_examples:
        try:
            file_path = importer.create_sample_mesh_file(
                mesh_name=mesh_name,
                vertices=vertices,
                subfolder="examples",
                overwrite=True
            )
            created_files.append(file_path)
            print(f"✓ 创建示例文件: {mesh_name}.txt ({len(vertices)} 个顶点)")
        except Exception as e:
            print(f"✗ 创建 {mesh_name}.txt 失败: {e}")

    print(f"\n总共创建了 {len(created_files)} 个mesh文件")
    return created_files


def verify_setup():
    """
    验证设置是否成功
    """
    print("\n正在验证项目设置...")

    try:
        # 验证导入器
        importer = create_default_importer()

        # 验证数据目录结构
        if importer.validate_data_structure():
            print("✓ 数据目录结构验证成功")
        else:
            print("✗ 数据目录结构验证失败")
            return False

        # 列出可用的mesh文件
        available_meshes = importer.list_available_meshes()
        print(f"✓ 在mesh目录中找到 {len(available_meshes)} 个文件:")
        for mesh in available_meshes:
            print(f"  - {mesh}")

        # 检查examples目录
        example_meshes = importer.list_available_meshes("examples")
        if example_meshes:
            print(f"✓ 在examples目录中找到 {len(example_meshes)} 个文件:")
            for mesh in example_meshes:
                print(f"  - {mesh}")

        # 测试加载一个mesh文件
        if available_meshes:
            test_mesh = available_meshes[0]
            try:
                boundary = importer.load_boundary_by_name(test_mesh)
                print(f"✓ 成功加载测试mesh: {test_mesh} ({len(boundary.get_vertices())} 个顶点)")
            except Exception as e:
                print(f"✗ 加载测试mesh失败: {e}")
                return False

        print("\n✓ 项目设置验证成功！")
        return True

    except Exception as e:
        print(f"✗ 验证过程中发生错误: {e}")
        return False


def print_usage_examples():
    """
    打印使用示例
    """
    print("\n" + "=" * 60)
    print("项目设置完成！以下是使用示例:")
    print("=" * 60)

    print("""
# 1. 基本使用示例
from src.rl.trainer import MeshTrainer

# 使用默认边界创建训练器
trainer = MeshTrainer()

# 从mesh名称创建训练器
trainer = MeshTrainer.from_mesh_name("simple_square")

# 从文件路径创建训练器
trainer = MeshTrainer.from_file("data/mesh/pentagon.txt")

# 2. 查看可用的mesh文件
from src.utils import create_default_importer
importer = create_default_importer()
available_meshes = importer.list_available_meshes()
print("可用的mesh文件:", available_meshes)

# 3. 获取mesh文件信息
mesh_info = importer.get_mesh_info("simple_square")
print("Mesh信息:", mesh_info)

# 4. 开始训练
trainer = MeshTrainer.from_mesh_name("simple_square")
stats = trainer.train(max_episodes=10)
    """)

    print("=" * 60)


def main():
    """
    主函数
    """
    print("强化学习网格生成项目设置")
    print("=" * 40)

    try:
        # 创建目录结构
        create_directory_structure()

        # 创建示例文件
        # create_sample_mesh_files()

        # 验证设置
        if verify_setup():
            print_usage_examples()
        else:
            print("\n⚠️  设置过程中发现问题，请检查错误信息")
            return 1

        return 0

    except Exception as e:
        print(f"\n❌ 设置过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
