import os
from typing import List, Tuple, Optional, Dict, Any
import logging

from src.geometry import Boundary, Mesh


class MeshImporter:
    """
    网格数据导入器

    该类用于从txt文件中读取边界点数据，创建Boundary对象并生成Mesh对象。
    支持读取顺时针排列的封闭图形边界点数据。
    所有路径配置都从config.yaml中获取。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化导入器

        Args:
            config: 配置字典，如果为None则从config.yaml加载
        """
        # 导入config加载函数
        from src.rl.config import load_config

        self.config = config if config is not None else load_config()
        self.paths_config = self.config.get("paths", {})

        # 设置数据根目录
        self.data_root = self._get_absolute_path(self.paths_config.get("data_root", "data"))

        # 设置日志
        self.logger = logging.getLogger(__name__)

        # 确保必要的目录存在
        self._ensure_directories()

    def _get_absolute_path(self, relative_path: str) -> str:
        """
        将相对路径转换为绝对路径（基于项目根目录）

        Args:
            relative_path: 相对于项目根目录的路径

        Returns:
            str: 绝对路径
        """
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(os.getcwd(), relative_path)

    def _get_subfolder_path(self, subfolder: str) -> str:
        """
        获取子文件夹的完整路径

        Args:
            subfolder: 子文件夹名称

        Returns:
            str: 子文件夹的完整路径
        """
        # 首先检查是否在paths配置中有对应的目录配置
        folder_key = f"{subfolder}_dir"
        if folder_key in self.paths_config:
            subfolder_path = self.paths_config[folder_key]
        else:
            # 如果没有配置，直接使用subfolder名称
            subfolder_path = subfolder

        return os.path.join(self.data_root, subfolder_path)

    def _ensure_directories(self):
        """
        确保必要的目录存在
        """
        try:
            # 确保数据根目录存在
            os.makedirs(self.data_root, exist_ok=True)

            # 确保常用的子目录存在
            common_dirs = ["mesh_dir", "custom_dir", "examples_dir"]
            for dir_key in common_dirs:
                if dir_key in self.paths_config:
                    dir_path = os.path.join(self.data_root, self.paths_config[dir_key])
                    os.makedirs(dir_path, exist_ok=True)

        except Exception as e:
            self.logger.warning(f"创建目录时发生错误: {e}")

    def get_data_root(self) -> str:
        """
        获取数据根目录路径

        Returns:
            str: 数据根目录的绝对路径
        """
        return self.data_root

    def get_subfolder_path(self, subfolder: str) -> str:
        """
        获取指定子文件夹的完整路径（公共方法）

        Args:
            subfolder: 子文件夹名称

        Returns:
            str: 子文件夹的完整路径
        """
        return self._get_subfolder_path(subfolder)

    def load_boundary_from_file(self, file_path: str) -> Boundary:
        """
        从指定文件路径加载边界数据

        Args:
            file_path: txt文件的完整路径

        Returns:
            Boundary: 创建的边界对象

        Raises:
            FileNotFoundError: 当文件不存在时
            ValueError: 当文件格式错误时
            IOError: 当文件读取失败时
        """
        # 如果是相对路径，转换为基于项目根目录的绝对路径
        if not os.path.isabs(file_path):
            file_path = self._get_absolute_path(file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"边界数据文件不存在: {file_path}")

        try:
            vertices = self._parse_vertices_from_file(file_path)
            boundary = Boundary(vertices)
            self.logger.info(f"成功从文件 {file_path} 加载边界，包含 {len(vertices)} 个顶点")
            return boundary

        except Exception as e:
            self.logger.error(f"加载边界数据失败: {file_path}, 错误: {str(e)}")
            raise

    def load_boundary_by_name(self, mesh_name: str, subfolder: str = "mesh") -> Boundary:
        """
        根据网格名称加载边界数据

        Args:
            mesh_name: 网格文件名（不含扩展名），如 "1", "simple_square"
            subfolder: 子文件夹名称，默认为 "mesh"，会从配置中查找对应的目录

        Returns:
            Boundary: 创建的边界对象

        Example:
            # 加载 data/mesh/1.txt（假设mesh_dir配置为"mesh"）
            boundary = importer.load_boundary_by_name("1")

            # 加载 data/custom/square.txt（假设custom_dir配置为"custom"）
            boundary = importer.load_boundary_by_name("square", "custom")
        """
        subfolder_path = self._get_subfolder_path(subfolder)
        file_path = os.path.join(subfolder_path, f"{mesh_name}.txt")
        return self.load_boundary_from_file(file_path)

    def create_mesh_from_file(self, file_path: str) -> Mesh:
        """
        从指定文件创建完整的网格对象

        Args:
            file_path: txt文件的完整路径

        Returns:
            Mesh: 创建的网格对象
        """
        boundary = self.load_boundary_from_file(file_path)
        mesh = Mesh(boundary)
        self.logger.info(f"成功创建网格对象，文件: {file_path}")
        return mesh

    def create_mesh_by_name(self, mesh_name: str, subfolder: str = "mesh") -> Mesh:
        """
        根据网格名称创建完整的网格对象

        Args:
            mesh_name: 网格文件名（不含扩展名）
            subfolder: 子文件夹名称，默认为 "mesh"

        Returns:
            Mesh: 创建的网格对象

        Example:
            # 创建来自配置路径下的网格
            mesh = importer.create_mesh_by_name("1")
        """
        boundary = self.load_boundary_by_name(mesh_name, subfolder)
        mesh = Mesh(boundary)
        self.logger.info(f"成功创建网格对象，名称: {mesh_name}")
        return mesh

    def _parse_vertices_from_file(self, file_path: str) -> List[Tuple[float, float]]:
        """
        从文件中解析顶点坐标

        Args:
            file_path: 文件路径

        Returns:
            List[Tuple[float, float]]: 顶点坐标列表

        Raises:
            ValueError: 当文件格式错误时
        """
        vertices = []

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()

                    # 跳过空行和注释行
                    if not line or line.startswith('#'):
                        continue

                    # 解析坐标
                    parts = line.split()
                    if len(parts) != 2:
                        raise ValueError(
                            f"文件 {file_path} 第 {line_num} 行格式错误: "
                            f"每行应包含两个由空格分隔的数字，实际内容: '{line}'"
                        )

                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        vertices.append((x, y))
                    except ValueError as e:
                        raise ValueError(
                            f"文件 {file_path} 第 {line_num} 行坐标解析失败: "
                            f"无法将 '{parts[0]}' 和 '{parts[1]}' 转换为数字"
                        ) from e

        except IOError as e:
            raise IOError(f"读取文件失败: {file_path}") from e

        if len(vertices) < 3:
            raise ValueError(
                f"边界点数量不足: 文件 {file_path} 只包含 {len(vertices)} 个点，"
                f"至少需要 3 个点才能构成有效的边界"
            )

        return vertices

    def validate_data_structure(self) -> bool:
        """
        验证数据目录结构是否正确

        Returns:
            bool: 如果数据目录结构正确返回True，否则返回False
        """
        try:
            # 检查数据根目录
            if not os.path.exists(self.data_root):
                self.logger.warning(f"数据根目录不存在: {self.data_root}")
                return False

            # 检查主要的子目录
            mesh_dir = self._get_subfolder_path("mesh")
            if not os.path.exists(mesh_dir):
                self.logger.warning(f"mesh目录不存在: {mesh_dir}")
                return False

            # 检查是否有txt文件
            txt_files = [f for f in os.listdir(mesh_dir) if f.endswith('.txt')]
            if not txt_files:
                self.logger.warning(f"mesh目录中没有找到txt文件: {mesh_dir}")
                return False

            self.logger.info(f"数据目录结构正确，找到 {len(txt_files)} 个txt文件")
            return True

        except Exception as e:
            self.logger.error(f"验证数据目录结构时发生错误: {str(e)}")
            return False

    def list_available_meshes(self, subfolder: str = "mesh") -> List[str]:
        """
        列出可用的网格文件

        Args:
            subfolder: 子文件夹名称，默认为 "mesh"

        Returns:
            List[str]: 可用的网格文件名列表（不含扩展名）
        """
        subfolder_path = self._get_subfolder_path(subfolder)

        if not os.path.exists(subfolder_path):
            self.logger.warning(f"目录不存在: {subfolder_path}")
            return []

        try:
            txt_files = [
                os.path.splitext(f)[0]  # 移除扩展名
                for f in os.listdir(subfolder_path)
                if f.endswith('.txt')
            ]

            self.logger.info(f"在 {subfolder_path} 中找到 {len(txt_files)} 个网格文件")
            return sorted(txt_files)

        except Exception as e:
            self.logger.error(f"列出网格文件时发生错误: {str(e)}")
            return []

    def get_mesh_info(self, mesh_name: str, subfolder: str = "mesh") -> dict:
        """
        获取网格文件的基本信息

        Args:
            mesh_name: 网格文件名（不含扩展名）
            subfolder: 子文件夹名称，默认为 "mesh"

        Returns:
            dict: 包含网格信息的字典
        """
        subfolder_path = self._get_subfolder_path(subfolder)
        file_path = os.path.join(subfolder_path, f"{mesh_name}.txt")

        info = {
            "name": mesh_name,
            "subfolder": subfolder,
            "configured_path": subfolder_path,
            "file_path": file_path,
            "exists": os.path.exists(file_path),
            "vertex_count": 0,
            "file_size": 0,
            "error": None
        }

        if not info["exists"]:
            info["error"] = "文件不存在"
            return info

        try:
            # 获取文件大小
            info["file_size"] = os.path.getsize(file_path)

            # 获取顶点数量
            vertices = self._parse_vertices_from_file(file_path)
            info["vertex_count"] = len(vertices)

        except Exception as e:
            info["error"] = str(e)

        return info

    def get_default_meshes(self) -> List[str]:
        """
        获取配置中定义的默认mesh列表

        Returns:
            List[str]: 默认mesh名称列表
        """
        return self.paths_config.get("default_meshes", [])

    def get_available_subfolders(self) -> Dict[str, str]:
        """
        获取所有配置的子文件夹及其路径

        Returns:
            Dict[str, str]: 子文件夹名称到路径的映射
        """
        subfolders = {}

        # 查找所有以_dir结尾的配置项
        for key, value in self.paths_config.items():
            if key.endswith('_dir'):
                folder_name = key[:-4]  # 移除'_dir'后缀
                folder_path = os.path.join(self.data_root, value)
                subfolders[folder_name] = folder_path

        return subfolders

    def create_sample_mesh_file(self, mesh_name: str, vertices: List[Tuple[float, float]],
                                subfolder: str = "mesh", overwrite: bool = False) -> str:
        """
        创建示例mesh文件

        Args:
            mesh_name: mesh文件名（不含扩展名）
            vertices: 顶点坐标列表
            subfolder: 子文件夹名称，默认为 "mesh"
            overwrite: 是否覆盖已存在的文件

        Returns:
            str: 创建的文件路径

        Raises:
            FileExistsError: 当文件已存在且overwrite为False时
            IOError: 当文件创建失败时
        """
        subfolder_path = self._get_subfolder_path(subfolder)
        os.makedirs(subfolder_path, exist_ok=True)

        file_path = os.path.join(subfolder_path, f"{mesh_name}.txt")

        if os.path.exists(file_path) and not overwrite:
            raise FileExistsError(f"文件已存在: {file_path}")

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Mesh file: {mesh_name}\n")
                f.write(f"# Generated automatically\n")
                f.write(f"# Vertices: {len(vertices)}\n")
                f.write("#\n")

                for x, y in vertices:
                    f.write(f"{x:.6f} {y:.6f}\n")

            self.logger.info(f"成功创建mesh文件: {file_path}")
            return file_path

        except Exception as e:
            raise IOError(f"创建文件失败: {file_path}, 错误: {str(e)}") from e


def create_default_importer(config: Optional[Dict[str, Any]] = None) -> MeshImporter:
    """
    创建默认的网格导入器实例

    Args:
        config: 配置字典，如果为None则从config.yaml加载

    Returns:
        MeshImporter: 使用配置的导入器实例
    """
    return MeshImporter(config)
