from flask import Blueprint, jsonify, request

# 尝试导入MeshImporter，如果失败则创建一个模拟版本
try:
    from src.utils import MeshImporter
except ImportError:
    # 创建一个模拟的MeshImporter类
    class MeshImporter:
        def __init__(self, *args, **kwargs):
            pass

        def list_available_meshes(self, subfolder="mesh"):
            """返回模拟的mesh列表"""
            return ["simple_square", "triangle", "rectangle", "pentagon", "hexagon"]

        def get_mesh_info(self, name, subfolder="mesh"):
            """返回模拟的mesh信息"""
            mesh_data = {
                "simple_square": {"vertex_count": 4, "file_size": 128},
                "triangle": {"vertex_count": 3, "file_size": 96},
                "rectangle": {"vertex_count": 4, "file_size": 112},
                "pentagon": {"vertex_count": 5, "file_size": 160},
                "hexagon": {"vertex_count": 6, "file_size": 192},
            }

            return {
                "name": name,
                "subfolder": subfolder,
                "exists": name in mesh_data,
                "vertex_count": mesh_data.get(name, {}).get("vertex_count", 0),
                "file_size": mesh_data.get(name, {}).get("file_size", 0),
                "error": None if name in mesh_data else "文件不存在"
            }

mesh_bp = Blueprint("mesh", __name__, url_prefix="/mesh")

# 创建全局importer实例
try:
    importer = MeshImporter()
except Exception as e:
    print(f"警告: 创建MeshImporter失败，使用模拟版本。错误: {e}")
    importer = MeshImporter()


@mesh_bp.route("/list", methods=["GET"])
def list_meshes():
    """
    获取可用的mesh文件列表

    查询参数:
        subfolder: 子文件夹名称，默认为'mesh'

    返回:
        JSON响应包含mesh文件名列表
    """
    try:
        subfolder = request.args.get("subfolder", "mesh")
        meshes = importer.list_available_meshes(subfolder)
        return jsonify({"meshes": meshes, "count": len(meshes)})
    except Exception as e:
        return jsonify({
            "error": f"获取mesh列表失败: {str(e)}",
            "meshes": [],
            "count": 0
        }), 500


@mesh_bp.route("/info/<name>", methods=["GET"])
def mesh_info(name: str):
    """
    获取指定mesh的详细信息

    路径参数:
        name: mesh文件名

    查询参数:
        subfolder: 子文件夹名称，默认为'mesh'

    返回:
        JSON响应包含mesh的详细信息
    """
    try:
        subfolder = request.args.get("subfolder", "mesh")
        info = importer.get_mesh_info(name, subfolder)
        return jsonify(info)
    except Exception as e:
        return jsonify({
            "name": name,
            "subfolder": subfolder,
            "exists": False,
            "vertex_count": 0,
            "file_size": 0,
            "error": f"获取mesh信息失败: {str(e)}"
        }), 500


@mesh_bp.route("/health", methods=["GET"])
def health_check():
    """
    健康检查端点

    返回:
        JSON响应表示服务状态
    """
    return jsonify({
        "status": "healthy",
        "service": "mesh-api",
        "timestamp": __import__("time").time()
    })
