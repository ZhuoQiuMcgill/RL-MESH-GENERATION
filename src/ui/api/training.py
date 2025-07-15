import traceback
from flask import Blueprint, request, jsonify, current_app
from typing import Optional
import os

from src.ui.training_manager import TrainingManager

# 设置OpenMP环境变量，解决库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 使用全局的训练管理器实例
training_manager = TrainingManager()

# 创建训练管理的蓝图
training_bp = Blueprint("training", __name__, url_prefix="/training")


@training_bp.route("/init", methods=["POST"])
def initialize_trainer():
    """
    初始化训练器

    Body参数:
        boundary_source (str, optional): 边界数据源
        backend (str, optional): SAC后端类型
        device (str, optional): 训练设备

    Returns:
        JSON响应表示初始化结果
    """
    try:
        data = request.get_json() or {}

        # 提取初始化参数
        boundary_source = data.get('boundary_source')
        backend = data.get('backend')
        device = data.get('device')

        # 修复：处理前端mesh_name到boundary_source的映射
        if boundary_source is None and 'mesh_name' in data and data['mesh_name']:
            boundary_source = data['mesh_name']
            print(f"从前端mesh_name映射boundary_source: {boundary_source}")

        # 确保boundary_source不为None
        if boundary_source is None or boundary_source == "":
            boundary_source = "simple_square"  # 使用默认mesh
            print(f"boundary_source为空，使用默认值: {boundary_source}")

        result = training_manager.initialize_trainer(
            boundary_source=boundary_source,
            backend=backend,
            device=device
        )

        if result["success"]:
            return jsonify({
                "success": True,
                "message": "trainer_initialized",
                "boundary_source": boundary_source,
                "backend": result.get("backend", "unknown"),
                "device": result.get("device", "auto")
            })
        else:
            return jsonify(result), 400

    except Exception as exc:
        current_app.logger.error(f"初始化训练器失败: {exc}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"初始化训练器时发生错误: {str(exc)}"
        }), 500


@training_bp.route("/start", methods=["POST"])
def start_training():
    """
    启动训练过程

    Body参数:
        max_timesteps (int, optional): 最大训练步数
        boundary_source (str, optional): 边界数据源
        backend (str, optional): SAC后端类型
        device (str, optional): 训练设备
        description (str, optional): 训练描述

    Returns:
        JSON响应表示启动结果
    """
    try:
        data = request.get_json() or {}

        # 提取训练参数
        max_timesteps = data.get('max_timesteps', 100000)
        boundary_source = data.get('boundary_source')
        backend = data.get('backend')
        device = data.get('device')
        description = data.get('description', '')

        # 修复：处理前端mesh_name到boundary_source的映射
        # 如果boundary_source为None但mesh_name存在，则使用mesh_name作为boundary_source
        if boundary_source is None and 'mesh_name' in data and data['mesh_name']:
            boundary_source = data['mesh_name']
            print(f"从前端mesh_name映射boundary_source: {boundary_source}")

        # 确保boundary_source不为None
        if boundary_source is None or boundary_source == "":
            boundary_source = "simple_square"  # 使用默认mesh
            print(f"boundary_source为空，使用默认值: {boundary_source}")

        # 启动训练
        result = training_manager.start_training(
            max_timesteps=max_timesteps,
            boundary_source=boundary_source,
            backend=backend,
            device=device,
            description=description
        )

        if result["success"]:
            return jsonify({
                "success": True,
                "message": "training_started",
                "training_id": result.get("training_id", ""),
                "boundary_source": boundary_source
            })
        else:
            return jsonify(result), 400

    except ValueError as exc:
        current_app.logger.error(f"训练参数错误: {exc}")
        return jsonify({
            "success": False,
            "error": f"参数验证失败: {str(exc)}"
        }), 400
    except Exception as exc:
        current_app.logger.error(f"启动训练失败: {exc}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"启动训练时发生错误: {str(exc)}"
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
        JSON响应包含完整的训练状态信息，符合前端API文档要求
    """
    try:
        status_data = training_manager.get_status()

        # 构建符合API文档的基础响应结构
        result = {
            "running": status_data.get("running", False),
            "status": status_data.get("status", "idle"),
            "stats": status_data.get("stats"),
            "backend_type": status_data.get("backend_type"),
            "timestamp": status_data.get("timestamp", __import__("time").time())
        }

        # 如果有训练统计数据，添加 progress 字段（前端期望的格式）
        if result["stats"] and isinstance(result["stats"], dict):
            stats = result["stats"]

            # 确保包含前端期望的所有字段，设置默认值
            episode = stats.get("episode", 0)
            total_steps = stats.get("total_steps", 0)
            episode_reward = stats.get("episode_reward", 0.0)
            average_reward = stats.get("average_reward", 0.0)
            buffer_size = stats.get("buffer_size", 0)

            result["progress"] = {
                "current_episode": episode,
                "total_steps": total_steps,
                "latest_reward": episode_reward,
                "average_reward": average_reward,
                "buffer_utilization": buffer_size
            }

        return jsonify(result)

    except Exception as exc:
        current_app.logger.error(f"获取训练状态失败: {exc}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== 获取训练状态异常 ===")
        print(f"错误: {exc}")
        traceback.print_exc()
        print(f"=== 错误结束 ===")
        return jsonify({
            "running": False,
            "status": "error",
            "stats": None,
            "backend_type": None,
            "error": str(exc),
            "timestamp": __import__("time").time()
        }), 500


@training_bp.route("/health", methods=["GET"])
def training_health_check():
    """
    训练管理API健康检查

    Returns:
        JSON响应表示训练服务状态
    """
    try:
        manager_running = training_manager.is_training_active()

        return jsonify({
            "status": "healthy",
            "service": "training-api",
            "manager_running": manager_running,
            "timestamp": __import__("time").time()
        })
    except Exception as exc:
        current_app.logger.error(f"训练健康检查异常: {exc}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== 训练健康检查异常 ===")
        print(f"错误: {exc}")
        traceback.print_exc()
        print(f"=== 错误结束 ===")
        return jsonify({
            "status": "unhealthy",
            "service": "training-api",
            "error": str(exc),
            "timestamp": __import__("time").time()
        }), 500
