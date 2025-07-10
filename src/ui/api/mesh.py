from flask import Blueprint, jsonify, request
from src.utils import MeshImporter

mesh_bp = Blueprint("mesh", __name__, url_prefix="/mesh")

# 创建全局importer实例 - 移除所有mock机制
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
    subfolder = request.args.get("subfolder", "mesh")
    meshes = importer.list_available_meshes(subfolder)
    return jsonify({"meshes": meshes, "count": len(meshes)})


@mesh_bp.route("/info/<n>", methods=["GET"])
def mesh_info(n: str):
    """
    获取指定mesh的详细信息

    路径参数:
        n: mesh文件名

    查询参数:
        subfolder: 子文件夹名称，默认为'mesh'

    返回:
        JSON响应包含mesh的详细信息
    """
    subfolder = request.args.get("subfolder", "mesh")
    info = importer.get_mesh_info(n, subfolder)
    return jsonify(info)


@mesh_bp.route("/boundary/<n>", methods=["GET"])
def mesh_boundary(n: str):
    """
    获取指定mesh的边界顶点数据（新增接口）

    路径参数:
        n: mesh文件名

    查询参数:
        subfolder: 子文件夹名称，默认为'mesh'

    返回:
        JSON响应包含mesh的边界顶点坐标列表
    """
    subfolder = request.args.get("subfolder", "mesh")

    try:
        # 加载边界对象
        boundary = importer.load_boundary_by_name(n, subfolder)

        # 获取顶点坐标列表
        vertices = boundary.get_vertices()

        return jsonify({
            "mesh_name": n,
            "subfolder": subfolder,
            "boundary_vertices": vertices,
            "vertex_count": len(vertices),
            "success": True
        })

    except FileNotFoundError:
        return jsonify({
            "error": f"Mesh文件不存在: {n}",
            "success": False
        }), 404

    except Exception as e:
        return jsonify({
            "error": f"加载边界数据失败: {str(e)}",
            "success": False
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
