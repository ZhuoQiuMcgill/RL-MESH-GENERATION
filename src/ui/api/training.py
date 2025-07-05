from flask import Blueprint, request, jsonify

from src.ui.training_manager import TrainingManager

training_manager = TrainingManager()

training_bp = Blueprint("training", __name__, url_prefix="/training")


@training_bp.route("/start", methods=["POST"])
def start_training():
    """
    启动训练过程

    请求体参数:
        mesh_name: 网格名称（可选）
        subfolder: 子文件夹名称，默认为'mesh'
        max_episodes: 最大训练轮数（可选）
        max_steps: 每轮最大步数（可选）

    返回:
        JSON响应表示启动结果
    """
    data = request.get_json(force=True, silent=True) or {}
    mesh_name = data.get("mesh_name")
    subfolder = data.get("subfolder", "mesh")
    episodes = data.get("max_episodes")
    steps = data.get("max_steps")

    try:
        training_manager.start_training(
            mesh_name=mesh_name,
            subfolder=subfolder,
            max_episodes=episodes,
            max_steps=steps,
        )
        return jsonify({
            "message": "training_started",
            "success": True,
            "config": {
                "mesh_name": mesh_name,
                "subfolder": subfolder,
                "max_episodes": episodes,
                "max_steps": steps
            }
        })
    except RuntimeError as exc:
        return jsonify({
            "error": str(exc),
            "success": False
        }), 400
    except Exception as exc:
        return jsonify({
            "error": f"启动训练时发生未知错误: {str(exc)}",
            "success": False
        }), 500


@training_bp.route("/stop", methods=["POST"])
def stop_training():
    """
    停止训练过程

    返回:
        JSON响应表示停止请求结果
    """
    try:
        training_manager.stop_training()
        return jsonify({
            "message": "stop_requested",
            "success": True
        })
    except Exception as exc:
        return jsonify({
            "error": f"停止训练时发生错误: {str(exc)}",
            "success": False
        }), 500


@training_bp.route("/status", methods=["GET"])
def status():
    """
    获取当前训练状态

    返回:
        JSON响应包含完整的训练状态信息
    """
    try:
        status_data = training_manager.get_status()

        # 确保返回的数据结构完整
        result = {
            "running": status_data.get("running", False),
            "status": status_data.get("status", "idle"),
            "stats": status_data.get("stats"),
            "timestamp": __import__("time").time()
        }

        # 如果有统计信息，添加一些有用的计算字段
        if result["stats"] and isinstance(result["stats"], dict):
            stats = result["stats"]
            result["progress"] = {
                "current_episode": stats.get("episode", 0),
                "total_steps": stats.get("total_steps", 0),
                "latest_reward": stats.get("episode_reward", 0.0),
                "average_reward": stats.get("average_reward", 0.0),
                "buffer_utilization": stats.get("buffer_size", 0)
            }

        return jsonify(result)
    except Exception as exc:
        return jsonify({
            "running": False,
            "status": "error",
            "stats": None,
            "error": f"获取状态时发生错误: {str(exc)}",
            "timestamp": __import__("time").time()
        }), 500


@training_bp.route("/health", methods=["GET"])
def health_check():
    """
    健康检查端点

    返回:
        JSON响应表示服务状态
    """
    return jsonify({
        "status": "healthy",
        "service": "training-api",
        "manager_running": training_manager.running,
        "timestamp": __import__("time").time()
    })
