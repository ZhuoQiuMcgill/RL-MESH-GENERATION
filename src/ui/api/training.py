"""
训练API - 重构版本

提供与前端交互的训练API，支持统一的SAC训练接口
"""
import traceback
from flask import Blueprint, request, jsonify, current_app

from ..training_manager import training_manager

# 创建蓝图
training_bp = Blueprint('training', __name__, url_prefix='/training')


@training_bp.route("/health", methods=["GET"])
def health_check():
    """健康检查端点"""
    try:
        return jsonify({
            "status": "healthy",
            "service": "training-api",
            "timestamp": __import__("time").time(),
            "backend_available": True
        })
    except Exception as exc:
        current_app.logger.error(f"健康检查失败: {exc}")
        return jsonify({
            "status": "unhealthy",
            "service": "training-api",
            "error": str(exc),
            "timestamp": __import__("time").time()
        }), 500


@training_bp.route("/info", methods=["GET"])
def get_trainer_info():
    """
    获取训练器信息

    Returns:
        JSON响应包含训练器配置和状态信息
    """
    try:
        trainer_info = training_manager.get_trainer_info()
        return jsonify({
            "success": True,
            "data": trainer_info
        })
    except Exception as exc:
        current_app.logger.error(f"获取训练器信息失败: {exc}")
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 500


@training_bp.route("/initialize", methods=["POST"])
def initialize_trainer():
    """
    初始化训练器

    Body参数:
        backend (str, optional): SAC后端类型 ("custom" 或 "sb3")
        boundary_source (str, optional): 边界数据源
        device (str, optional): 训练设备

    Returns:
        JSON响应表示初始化结果
    """
    try:
        data = request.get_json() or {}

        result = training_manager.initialize_trainer(
            boundary_source=data.get('boundary_source'),
            device=data.get('device'),
            backend=data.get('backend')
        )

        if result["success"]:
            return jsonify(result)
        else:
            return jsonify(result), 400

    except Exception as exc:
        current_app.logger.error(f"初始化训练器失败: {exc}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"初始化训练器时发生错误: {str(exc)}"
        }), 500


@training_bp.route("/switch-backend", methods=["POST"])
def switch_backend():
    """
    切换SAC后端

    Body参数:
        backend_type (str): 目标后端类型 ("custom" 或 "sb3")
        boundary_source (str, optional): 边界数据源

    Returns:
        JSON响应表示切换结果
    """
    try:
        data = request.get_json() or {}
        backend_type = data.get('backend_type')

        if not backend_type:
            return jsonify({
                "success": False,
                "error": "缺少必需参数: backend_type"
            }), 400

        result = training_manager.switch_backend(
            backend_type=backend_type,
            boundary_source=data.get('boundary_source')
        )

        if result["success"]:
            return jsonify(result)
        else:
            return jsonify(result), 400

    except Exception as exc:
        current_app.logger.error(f"切换后端失败: {exc}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"切换后端时发生错误: {str(exc)}"
        }), 500


@training_bp.route("/start", methods=["POST"])
def start_training():
    """
    启动训练过程

    Body参数:
        max_timesteps (int, optional): 最大训练步数，默认100000
        max_steps (int, optional): 每episode最大步数，默认1000
        batch_size (int, optional): 批次大小
        start_training_steps (int, optional): 开始训练的步数
        description (str, optional): 训练描述
        mesh_name (str, optional): Mesh名称
        boundary_source (str, optional): 边界数据源

    Returns:
        JSON响应表示启动结果
    """
    try:
        data = request.get_json() or {}

        # 如果指定了新的边界源，先加载
        boundary_source = data.get('boundary_source')
        if boundary_source:
            load_result = training_manager.load_boundary(boundary_source)
            if not load_result["success"]:
                return jsonify({
                    "success": False,
                    "error": f"加载边界失败: {load_result['error']}"
                }), 400

        # 提取训练参数
        training_params = {
            'max_timesteps': data.get('max_timesteps', 100000),
            'max_steps': data.get('max_steps', 1000),
            'description': data.get('description', ''),
            'mesh_name': data.get('mesh_name', '')
        }

        # 添加可选参数
        if 'batch_size' in data:
            training_params['batch_size'] = data['batch_size']
        if 'start_training_steps' in data:
            training_params['start_training_steps'] = data['start_training_steps']

        # 启动训练
        result = training_manager.start_training(**training_params)

        if result["success"]:
            return jsonify(result)
        else:
            return jsonify(result), 400

    except ValueError as exc:
        current_app.logger.error(f"训练参数错误: {exc}")
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400
    except Exception as exc:
        current_app.logger.error(f"启动训练异常: {exc}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== 启动训练异常 ===")
        print(f"错误: {exc}")
        traceback.print_exc()
        print(f"=== 错误结束 ===")
        return jsonify({
            "success": False,
            "error": f"启动训练时发生未知错误: {str(exc)}"
        }), 500


@training_bp.route("/stop", methods=["POST"])
def stop_training():
    """
    停止训练过程

    Returns:
        JSON响应表示停止请求结果
    """
    try:
        result = training_manager.stop_training()

        if result["success"]:
            return jsonify({
                "success": True,
                "message": "stop_requested"
            })
        else:
            return jsonify(result), 400

    except Exception as exc:
        current_app.logger.error(f"停止训练异常: {exc}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== 停止训练异常 ===")
        print(f"错误: {exc}")
        traceback.print_exc()
        print(f"=== 错误结束 ===")
        return jsonify({
            "success": False,
            "error": f"停止训练时发生错误: {str(exc)}"
        }), 500


@training_bp.route("/status", methods=["GET"])
def get_training_status():
    """
    获取当前训练状态

    Returns:
        JSON响应包含完整的训练状态信息
    """
    try:
        status_data = training_manager.get_status()

        # 确保返回的数据结构完整
        result = {
            "running": status_data.get("running", False),
            "status": status_data.get("status", "idle"),
            "stats": status_data.get("stats"),
            "backend_type": status_data.get("backend_type"),
            "timestamp": status_data.get("timestamp", 0)
        }

        # 添加额外的统计信息
        if result["stats"] and isinstance(result["stats"], dict):
            stats = result["stats"]

            # 计算训练进度百分比（如果有最大步数信息）
            total_steps = stats.get("total_steps", 0)
            if "max_timesteps" in stats and stats["max_timesteps"] > 0:
                progress_percent = min(100.0, (total_steps / stats["max_timesteps"]) * 100)
                stats["progress_percent"] = progress_percent

            # 添加训练时长格式化
            training_time = stats.get("training_time", 0)
            if training_time > 0:
                hours = int(training_time // 3600)
                minutes = int((training_time % 3600) // 60)
                seconds = int(training_time % 60)
                stats["training_time_formatted"] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        return jsonify(result)

    except Exception as exc:
        current_app.logger.error(f"获取训练状态异常: {exc}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            "running": False,
            "status": "error",
            "stats": None,
            "backend_type": None,
            "error": str(exc),
            "timestamp": __import__("time").time()
        }), 500


@training_bp.route("/boundary/load", methods=["POST"])
def load_boundary():
    """
    加载新的边界数据

    Body参数:
        boundary_source (str): 边界数据源

    Returns:
        JSON响应表示加载结果
    """
    try:
        data = request.get_json() or {}
        boundary_source = data.get('boundary_source')

        if not boundary_source:
            return jsonify({
                "success": False,
                "error": "缺少必需参数: boundary_source"
            }), 400

        result = training_manager.load_boundary(boundary_source)

        if result["success"]:
            return jsonify(result)
        else:
            return jsonify(result), 400

    except Exception as exc:
        current_app.logger.error(f"加载边界异常: {exc}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"加载边界时发生错误: {str(exc)}"
        }), 500


@training_bp.route("/model/save", methods=["POST"])
def save_model():
    """
    保存模型

    Body参数:
        path (str): 保存路径

    Returns:
        JSON响应表示保存结果
    """
    try:
        data = request.get_json() or {}
        path = data.get('path')

        if not path:
            return jsonify({
                "success": False,
                "error": "缺少必需参数: path"
            }), 400

        result = training_manager.save_model(path)

        if result["success"]:
            return jsonify(result)
        else:
            return jsonify(result), 400

    except Exception as exc:
        current_app.logger.error(f"保存模型异常: {exc}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"保存模型时发生错误: {str(exc)}"
        }), 500


@training_bp.route("/model/load", methods=["POST"])
def load_model():
    """
    加载模型

    Body参数:
        path (str): 模型路径

    Returns:
        JSON响应表示加载结果
    """
    try:
        data = request.get_json() or {}
        path = data.get('path')

        if not path:
            return jsonify({
                "success": False,
                "error": "缺少必需参数: path"
            }), 400

        result = training_manager.load_model(path)

        if result["success"]:
            return jsonify(result)
        else:
            return jsonify(result), 400

    except Exception as exc:
        current_app.logger.error(f"加载模型异常: {exc}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"加载模型时发生错误: {str(exc)}"
        }), 500


# 错误处理器
@training_bp.errorhandler(404)
def not_found(error):
    """处理404错误"""
    return jsonify({
        "success": False,
        "error": "API端点不存在"
    }), 404


@training_bp.errorhandler(405)
def method_not_allowed(error):
    """处理405错误"""
    return jsonify({
        "success": False,
        "error": "不支持的HTTP方法"
    }), 405


@training_bp.errorhandler(500)
def internal_error(error):
    """处理500错误"""
    return jsonify({
        "success": False,
        "error": "服务器内部错误"
    }), 500
