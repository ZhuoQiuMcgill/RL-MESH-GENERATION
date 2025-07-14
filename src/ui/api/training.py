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


@training_bp.route("/start", methods=["POST"])
def start_training():
    """
    开始训练过程

    Body参数:
        max_timesteps (int): 最大训练步数，默认1000000
        max_episode_steps (int): 每episode最大步数，默认1000
        mesh_name (str): 训练用的mesh名称，可选
        description (str): 训练描述，可选
        batch_size (int): 批量大小，可选
        start_training_steps (int): 开始训练的步数，可选

    Returns:
        JSON响应表示启动结果
    """
    try:
        data = request.get_json() or {}

        # 验证必需参数
        max_timesteps = data.get('max_timesteps', 1000000)
        max_episode_steps = data.get('max_episode_steps', 1000)

        if not isinstance(max_timesteps, int) or max_timesteps <= 0:
            return jsonify({
                "success": False,
                "error": "max_timesteps必须是正整数"
            }), 400

        if not isinstance(max_episode_steps, int) or max_episode_steps <= 0:
            return jsonify({
                "success": False,
                "error": "max_episode_steps必须是正整数"
            }), 400

        # 构建训练参数
        training_params = {
            'max_timesteps': max_timesteps,
            'max_episode_steps': max_episode_steps,
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
            episode_length = stats.get("episode_length", 0)
            boundary_vertices = stats.get("boundary_vertices", 0)
            buffer_size = stats.get("buffer_size", 0)
            training_id = stats.get("training_id", "")
            online_learning_mode = stats.get("online_learning_mode", False)

            # 确保包含网格可视化数据
            mesh_data = stats.get("mesh_data", {})
            boundary_vertices_data = stats.get("boundary_vertices_data", [])
            reference_point_info = stats.get("reference_point_info", {})

            # 更新 stats 对象，确保所有字段都存在
            stats.update({
                "episode": episode,
                "total_steps": total_steps,
                "episode_reward": episode_reward,
                "average_reward": average_reward,
                "episode_length": episode_length,
                "boundary_vertices": boundary_vertices,
                "buffer_size": buffer_size,
                "training_id": training_id,
                "online_learning_mode": online_learning_mode,
                "mesh_data": mesh_data,
                "boundary_vertices_data": boundary_vertices_data,
                "reference_point_info": reference_point_info
            })

            # 添加可选的损失和alpha值
            if "recent_actor_loss" in stats:
                stats["recent_actor_loss"] = stats["recent_actor_loss"]
            if "recent_critic_loss" in stats:
                stats["recent_critic_loss"] = stats["recent_critic_loss"]
            if "current_alpha" in stats:
                stats["current_alpha"] = stats["current_alpha"]

            # 添加 progress 对象（前端期望的格式）
            result["progress"] = {
                "current_episode": episode,
                "total_steps": total_steps,
                "latest_reward": episode_reward,
                "average_reward": average_reward,
                "buffer_utilization": buffer_size
            }

            # 计算训练进度百分比（如果有最大步数信息）
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