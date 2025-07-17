"""
训练历史API端点

提供训练历史数据查询的REST API接口，包括历史列表、训练详情和episode数据的查询功能。
"""

import traceback
import time
from flask import Blueprint, jsonify, request, current_app

from src.rl.training.history_manager import HistoryManager

training_history_bp = Blueprint("training_history", __name__, url_prefix="/training/history")

# 创建全局HistoryManager实例
history_manager = HistoryManager()


@training_history_bp.route("/list", methods=["GET"])
def list_training_history():
    """
    获取所有训练历史的ID列表

    GET /training/history/list

    Returns:
        JSON响应包含training_id列表
    """
    try:
        training_ids = history_manager.list_training_id()

        current_app.logger.info(f"查询到 {len(training_ids)} 个训练历史")

        return jsonify({
            "training_ids": training_ids,
            "count": len(training_ids),
            "success": True
        }), 200

    except Exception as e:
        error_msg = f"获取训练历史列表失败: {str(e)}"
        current_app.logger.error(f"训练历史列表查询异常: {error_msg}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== 训练历史列表查询异常 ===")
        print(f"Error: {e}")
        traceback.print_exc()
        print(f"=== End Error ===")

        return jsonify({
            "error": error_msg,
            "training_ids": [],
            "count": 0,
            "success": False
        }), 500


@training_history_bp.route("/info/<training_id>", methods=["POST"])
def get_training_info(training_id: str):
    """
    获取指定训练的基本信息（detail长度和最佳episode）

    POST /training/history/info/{training_id}

    Args:
        training_id: 训练会话ID

    Returns:
        JSON响应包含训练的基本信息
    """
    try:
        # 聚焦到指定的训练ID
        history_manager.focus_on(training_id)

        # 获取基本信息
        detail_length = history_manager.size
        best_episode = history_manager.best_episode

        current_app.logger.info(f"获取训练信息: {training_id}, 长度: {detail_length}, 最佳episode: {best_episode}")

        return jsonify({
            "training_id": training_id,
            "detail_length": detail_length,
            "best_episode": best_episode,
            "success": True
        }), 200

    except FileNotFoundError as e:
        error_msg = f"训练历史不存在: {training_id}"
        current_app.logger.warning(f"训练历史查询失败: {error_msg}")

        return jsonify({
            "error": error_msg,
            "training_id": training_id,
            "success": False
        }), 404

    except Exception as e:
        error_msg = f"获取训练信息失败: {str(e)}"
        current_app.logger.error(f"训练信息查询异常: {error_msg}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== 训练信息查询异常 ===")
        print(f"Error: {e}")
        traceback.print_exc()
        print(f"=== End Error ===")

        return jsonify({
            "error": error_msg,
            "training_id": training_id,
            "success": False
        }), 500


@training_history_bp.route("/episode/<training_id>/<int:episode_index>", methods=["POST"])
def get_episode_data(training_id: str, episode_index: int):
    """
    获取指定训练的指定episode的详细数据

    POST /training/history/episode/{training_id}/{episode_index}

    Args:
        training_id: 训练会话ID
        episode_index: episode索引

    Returns:
        JSON响应包含episode的详细数据
    """
    try:
        # 聚焦到指定的训练ID
        history_manager.focus_on(training_id)

        # 获取episode数据
        episode_data = history_manager.get_episode_data(episode_index)

        current_app.logger.info(f"获取episode数据: {training_id}, episode: {episode_index}")

        return jsonify({
            "training_id": training_id,
            "episode_index": episode_index,
            "episode_data": episode_data,
            "success": True
        }), 200

    except FileNotFoundError as e:
        error_msg = f"训练历史不存在: {training_id}"
        current_app.logger.warning(f"Episode数据查询失败: {error_msg}")

        return jsonify({
            "error": error_msg,
            "training_id": training_id,
            "episode_index": episode_index,
            "success": False
        }), 404

    except IndexError as e:
        error_msg = f"Episode索引超出范围: {episode_index}"
        current_app.logger.warning(f"Episode数据查询失败: {error_msg}")

        return jsonify({
            "error": error_msg,
            "training_id": training_id,
            "episode_index": episode_index,
            "success": False
        }), 400

    except RuntimeError as e:
        error_msg = f"HistoryManager未聚焦: {str(e)}"
        current_app.logger.error(f"Episode数据查询失败: {error_msg}")

        return jsonify({
            "error": error_msg,
            "training_id": training_id,
            "episode_index": episode_index,
            "success": False
        }), 500

    except Exception as e:
        error_msg = f"获取episode数据失败: {str(e)}"
        current_app.logger.error(f"Episode数据查询异常: {error_msg}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== Episode数据查询异常 ===")
        print(f"Error: {e}")
        traceback.print_exc()
        print(f"=== End Error ===")

        return jsonify({
            "error": error_msg,
            "training_id": training_id,
            "episode_index": episode_index,
            "success": False
        }), 500


@training_history_bp.route("/health", methods=["GET"])
def health_check():
    """
    训练历史服务健康检查

    GET /training/history/health

    Returns:
        JSON响应表示服务健康状态
    """
    try:
        # 检查基本功能
        training_ids = history_manager.list_training_id()
        current_focus = history_manager.current_focus_id()

        return jsonify({
            "status": "healthy",
            "service": "training-history-api",
            "available_trainings": len(training_ids),
            "current_focus": current_focus,
            "timestamp": time.time()
        }), 200

    except Exception as e:
        error_msg = f"健康检查失败: {str(e)}"
        current_app.logger.error(f"训练历史健康检查异常: {error_msg}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== 训练历史健康检查异常 ===")
        print(f"Error: {e}")
        traceback.print_exc()
        print(f"=== End Error ===")

        return jsonify({
            "status": "unhealthy",
            "service": "training-history-api",
            "error": error_msg,
            "timestamp": time.time()
        }), 500


# 错误处理
@training_history_bp.errorhandler(404)
def not_found(error):
    """处理404错误"""
    return jsonify({
        "error": "Endpoint not found",
        "success": False
    }), 404


@training_history_bp.errorhandler(405)
def method_not_allowed(error):
    """处理405错误"""
    return jsonify({
        "error": "Method not allowed",
        "success": False
    }), 405


@training_history_bp.errorhandler(500)
def internal_error(error):
    """处理500错误"""
    return jsonify({
        "error": "Internal server error",
        "success": False
    }), 500
