"""
训练管理API端点

提供强化学习训练会话的REST API接口，包括启动、停止、状态查询等功能。
严格按照API文档规范返回数据格式。
"""

import traceback
import time
from flask import Blueprint, jsonify, request, current_app

from src.ui.training_manager import get_training_manager

training_bp = Blueprint("training", __name__, url_prefix="/training")


@training_bp.route("/start", methods=["POST"])
def start_training():
    """
    启动训练会话

    POST /training/start
    Content-Type: application/json

    Body:
        {
            "mesh_name": "simple_square",
            "subfolder": "mesh",
            "max_timesteps": 1000000,
            "max_steps": 1000,
            "description": "Training description"
        }

    Returns:
        JSON响应包含启动结果
    """
    try:
        # 获取请求数据
        data = request.get_json() or {}

        # 提取配置参数
        config = {
            "mesh_name": data.get("mesh_name"),
            "subfolder": data.get("subfolder", "mesh"),
            "max_timesteps": data.get("max_timesteps"),
            "max_steps": data.get("max_steps"),
            "description": data.get("description")
        }

        # 过滤None值
        config = {k: v for k, v in config.items() if v is not None}

        # 获取训练管理器
        manager = get_training_manager()

        # 启动训练
        result = manager.start_training(config)

        current_app.logger.info(f"训练启动成功: {result}")
        return jsonify(result), 200

    except RuntimeError as e:
        error_msg = str(e)
        current_app.logger.warning(f"训练启动失败 - 运行时错误: {error_msg}")
        return jsonify({
            "error": error_msg,
            "success": False
        }), 400

    except ValueError as e:
        error_msg = str(e)
        current_app.logger.warning(f"训练启动失败 - 参数错误: {error_msg}")
        return jsonify({
            "error": error_msg,
            "success": False
        }), 400

    except Exception as e:
        error_msg = f"启动训练失败: {str(e)}"
        current_app.logger.error(f"训练启动异常: {error_msg}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== 训练启动异常 ===")
        print(f"Error: {e}")
        traceback.print_exc()
        print(f"=== End Error ===")
        return jsonify({
            "error": error_msg,
            "success": False
        }), 500


@training_bp.route("/stop", methods=["POST"])
def stop_training():
    """
    停止当前训练会话

    POST /training/stop

    Returns:
        JSON响应包含停止结果
    """
    try:
        # 获取训练管理器
        manager = get_training_manager()

        # 停止训练
        result = manager.stop_training()

        current_app.logger.info(f"训练停止: {result}")
        return jsonify(result), 200

    except Exception as e:
        error_msg = f"停止训练失败: {str(e)}"
        current_app.logger.error(f"训练停止异常: {error_msg}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== 训练停止异常 ===")
        print(f"Error: {e}")
        traceback.print_exc()
        print(f"=== End Error ===")
        return jsonify({
            "error": error_msg,
            "success": False
        }), 500


@training_bp.route("/status", methods=["GET"])
def get_training_status():
    """
    获取训练状态

    GET /training/status

    Returns:
        JSON响应包含详细的训练状态信息，格式严格按照API文档规范
    """
    try:
        # 获取训练管理器
        manager = get_training_manager()

        # 获取状态
        status = manager.get_status()

        current_app.logger.debug(f"训练状态查询: running={status['running']}")
        return jsonify(status), 200

    except Exception as e:
        error_msg = f"获取训练状态失败: {str(e)}"
        current_app.logger.error(f"训练状态查询异常: {error_msg}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== 训练状态查询异常 ===")
        print(f"Error: {e}")
        traceback.print_exc()
        print(f"=== End Error ===")

        # 返回默认状态以确保前端不会因为API错误而无法工作
        return jsonify({
            "running": False,
            "status": "error",
            "stats": None,
            "progress": None,
            "timestamp": time.time(),
            "error": error_msg
        }), 200  # 仍返回200以保证前端正常工作


@training_bp.route("/health", methods=["GET"])
def health_check():
    """
    训练服务健康检查

    GET /training/health

    Returns:
        JSON响应表示服务健康状态
    """
    try:
        # 获取训练管理器
        manager = get_training_manager()

        # 获取健康状态
        health_status = manager.get_health_status()

        return jsonify(health_status), 200

    except Exception as e:
        error_msg = f"健康检查失败: {str(e)}"
        current_app.logger.error(f"训练健康检查异常: {error_msg}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== 训练健康检查异常 ===")
        print(f"Error: {e}")
        traceback.print_exc()
        print(f"=== End Error ===")

        return jsonify({
            "status": "unhealthy",
            "service": "training-api",
            "manager_running": False,
            "error": error_msg,
            "timestamp": time.time()
        }), 500


# 错误处理
@training_bp.errorhandler(404)
def not_found(error):
    """处理404错误"""
    return jsonify({
        "error": "Endpoint not found",
        "success": False
    }), 404


@training_bp.errorhandler(405)
def method_not_allowed(error):
    """处理405错误"""
    return jsonify({
        "error": "Method not allowed",
        "success": False
    }), 405


@training_bp.errorhandler(500)
def internal_error(error):
    """处理500错误"""
    return jsonify({
        "error": "Internal server error",
        "success": False
    }), 500
