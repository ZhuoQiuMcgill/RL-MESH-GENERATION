import traceback
from flask import Blueprint, jsonify, request, current_app
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
    try:
        subfolder = request.args.get("subfolder", "mesh")
        meshes = importer.list_available_meshes(subfolder)
        return jsonify({"meshes": meshes, "count": len(meshes)})
    except Exception as exc:
        current_app.logger.error(f"Exception in list_meshes: {exc}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== Exception in list_meshes ===")
        print(f"Error: {exc}")
        traceback.print_exc()
        print(f"=== End Error ===")
        return jsonify({
            "error": f"获取mesh列表失败: {str(exc)}",
            "meshes": [],
            "count": 0
        }), 500


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
    try:
        subfolder = request.args.get("subfolder", "mesh")
        info = importer.get_mesh_info(n, subfolder)
        return jsonify(info)
    except Exception as exc:
        current_app.logger.error(f"Exception in mesh_info: {exc}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== Exception in mesh_info ===")
        print(f"Error: {exc}")
        traceback.print_exc()
        print(f"=== End Error ===")
        return jsonify({
            "error": f"获取mesh信息失败: {str(exc)}",
            "name": n,
            "exists": False
        }), 500


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
    try:
        subfolder = request.args.get("subfolder", "mesh")

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

    except FileNotFoundError as exc:
        current_app.logger.error(f"FileNotFoundError in mesh_boundary: {exc}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== FileNotFoundError in mesh_boundary ===")
        print(f"Error: {exc}")
        traceback.print_exc()
        print(f"=== End Error ===")
        return jsonify({
            "error": f"Mesh文件不存在: {n}",
            "success": False
        }), 404

    except Exception as exc:
        current_app.logger.error(f"Exception in mesh_boundary: {exc}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== Exception in mesh_boundary ===")
        print(f"Error: {exc}")
        traceback.print_exc()
        print(f"=== End Error ===")
        return jsonify({
            "error": f"加载边界数据失败: {str(exc)}",
            "success": False
        }), 500


@mesh_bp.route("/health", methods=["GET"])
def health_check():
    """
    健康检查端点

    返回:
        JSON响应表示服务状态
    """
    try:
        return jsonify({
            "status": "healthy",
            "service": "mesh-api",
            "timestamp": __import__("time").time()
        })
    except Exception as exc:
        current_app.logger.error(f"Exception in health_check: {exc}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== Exception in health_check ===")
        print(f"Error: {exc}")
        traceback.print_exc()
        print(f"=== End Error ===")
        return jsonify({
            "status": "unhealthy",
            "service": "mesh-api",
            "error": str(exc),
            "timestamp": __import__("time").time()
        }), 500
