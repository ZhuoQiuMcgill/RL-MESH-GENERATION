import traceback
from flask import Blueprint, request, jsonify, current_app
from typing import Optional

from src.ui.training_manager import TrainingManager

# 使用全局的训练管理器实例 - 移除所有mock机制
training_manager = TrainingManager()

# 创建历史管理的蓝图
training_history_bp = Blueprint("training_history", __name__, url_prefix="/training/history")


@training_history_bp.route("/list", methods=["GET"])
def list_training_history():
    """
    获取所有训练历史记录列表

    返回:
        JSON响应包含所有训练记录的列表
    """
    try:
        history_list = training_manager.list_all_training_history()
        return jsonify({
            "success": True,
            "trainings": history_list,
            "count": len(history_list)
        })
    except Exception as exc:
        current_app.logger.error(f"Exception in list_training_history: {exc}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== Exception in list_training_history ===")
        print(f"Error: {exc}")
        traceback.print_exc()
        print(f"=== End Error ===")
        return jsonify({
            "success": False,
            "error": f"获取训练历史列表失败: {str(exc)}",
            "trainings": [],
            "count": 0
        }), 500


@training_history_bp.route("/info/<training_id>", methods=["GET"])
def get_training_info(training_id: str):
    """
    获取指定训练的详细信息

    路径参数:
        training_id: 训练会话ID

    返回:
        JSON响应包含训练的详细信息
    """
    try:
        training_info = training_manager.get_training_history(training_id)

        if "error" in training_info:
            return jsonify({
                "success": False,
                "error": training_info["error"]
            }), 404

        return jsonify({
            "success": True,
            "training_info": training_info
        })
    except Exception as exc:
        current_app.logger.error(f"Exception in get_training_info: {exc}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== Exception in get_training_info ===")
        print(f"Error: {exc}")
        traceback.print_exc()
        print(f"=== End Error ===")
        return jsonify({
            "success": False,
            "error": f"获取训练信息失败: {str(exc)}"
        }), 500


@training_history_bp.route("/current", methods=["GET"])
def get_current_training_info():
    """
    获取当前训练会话的信息

    返回:
        JSON响应包含当前训练的信息
    """
    try:
        current_training_id = training_manager.get_current_training_id()

        if current_training_id is None:
            return jsonify({
                "success": False,
                "error": "没有活动的训练会话"
            }), 404

        training_info = training_manager.get_training_history(current_training_id)

        return jsonify({
            "success": True,
            "current_training_id": current_training_id,
            "training_info": training_info
        })
    except Exception as exc:
        current_app.logger.error(f"Exception in get_current_training_info: {exc}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== Exception in get_current_training_info ===")
        print(f"Error: {exc}")
        traceback.print_exc()
        print(f"=== End Error ===")
        return jsonify({
            "success": False,
            "error": f"获取当前训练信息失败: {str(exc)}"
        }), 500


@training_history_bp.route("/export/<training_id>", methods=["POST"])
def export_training_summary(training_id: str):
    """
    导出指定训练的摘要报告

    路径参数:
        training_id: 训练会话ID

    请求体参数:
        export_path (可选): 导出文件路径

    返回:
        JSON响应包含导出结果
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        export_path = data.get("export_path")

        result_path = training_manager.export_training_summary(training_id, export_path)

        if result_path is None:
            return jsonify({
                "success": False,
                "error": "导出失败，可能是训练记录不存在或导出过程中发生错误"
            }), 500

        return jsonify({
            "success": True,
            "export_path": result_path,
            "message": f"训练摘要已导出到: {result_path}"
        })
    except Exception as exc:
        current_app.logger.error(f"Exception in export_training_summary: {exc}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== Exception in export_training_summary ===")
        print(f"Error: {exc}")
        traceback.print_exc()
        print(f"=== End Error ===")
        return jsonify({
            "success": False,
            "error": f"导出训练摘要失败: {str(exc)}"
        }), 500


@training_history_bp.route("/episode/<training_id>/<int:episode_num>", methods=["GET"])
def get_episode_data(training_id: str, episode_num: int):
    """
    获取指定训练中特定episode的详细数据

    路径参数:
        training_id: 训练会话ID
        episode_num: episode编号

    返回:
        JSON响应包含episode的详细数据
    """
    try:
        # 通过训练器的history_manager获取
        if hasattr(training_manager, '_trainer') and training_manager._trainer:
            trainer = training_manager._trainer
            if hasattr(trainer, 'history_manager'):
                episode_data = trainer.history_manager.get_episode_data(training_id, episode_num)

                if episode_data is None:
                    return jsonify({
                        "success": False,
                        "error": f"Episode {episode_num} 在训练 {training_id} 中不存在"
                    }), 404

                return jsonify({
                    "success": True,
                    "episode_data": episode_data
                })

        return jsonify({
            "success": False,
            "error": "当前训练器不支持episode数据查询"
        }), 501
    except Exception as exc:
        current_app.logger.error(f"Exception in get_episode_data: {exc}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== Exception in get_episode_data ===")
        print(f"Error: {exc}")
        traceback.print_exc()
        print(f"=== End Error ===")
        return jsonify({
            "success": False,
            "error": f"获取episode数据失败: {str(exc)}"
        }), 500


@training_history_bp.route("/stats/<training_id>", methods=["GET"])
def get_training_statistics(training_id: str):
    """
    获取指定训练的统计信息摘要

    路径参数:
        training_id: 训练会话ID

    返回:
        JSON响应包含训练的统计摘要
    """
    try:
        training_info = training_manager.get_training_history(training_id)

        if "error" in training_info:
            return jsonify({
                "success": False,
                "error": training_info["error"]
            }), 404

        metadata = training_info.get("metadata", {})

        # 构建统计摘要
        stats_summary = {
            "training_id": training_id,
            "status": metadata.get("status", "unknown"),
            "start_datetime": metadata.get("start_datetime"),
            "end_datetime": metadata.get("end_datetime"),
            "duration_seconds": metadata.get("duration_seconds"),
            "episodes_completed": metadata.get("episodes_completed", 0),
            "total_steps": metadata.get("total_steps", 0),
            "best_reward": metadata.get("best_reward"),
            "mesh_name": metadata.get("mesh_name"),
            "description": metadata.get("description"),
            "episode_count": training_info.get("episode_count", 0)
        }

        # 如果有最终统计信息，添加更多细节
        final_stats = metadata.get("final_stats", {})
        if final_stats:
            stats_summary.update({
                "final_episode_rewards": final_stats.get("episode_rewards", [])[-10:] if final_stats.get(
                    "episode_rewards") else [],  # 最后10个episode的奖励
                "training_time": final_stats.get("training_time", 0),
                "evaluation_rewards": final_stats.get("evaluation_rewards", [])
            })

        return jsonify({
            "success": True,
            "statistics": stats_summary
        })
    except Exception as exc:
        current_app.logger.error(f"Exception in get_training_statistics: {exc}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== Exception in get_training_statistics ===")
        print(f"Error: {exc}")
        traceback.print_exc()
        print(f"=== End Error ===")
        return jsonify({
            "success": False,
            "error": f"获取训练统计信息失败: {str(exc)}"
        }), 500


@training_history_bp.route("/search", methods=["GET"])
def search_training_history():
    """
    搜索训练历史记录

    查询参数:
        mesh_name: 按mesh名称筛选
        status: 按状态筛选 (running, completed, stopped, error)
        start_date: 开始日期 (YYYY-MM-DD格式)
        end_date: 结束日期 (YYYY-MM-DD格式)
        limit: 返回记录数量限制，默认10

    返回:
        JSON响应包含筛选后的训练记录
    """
    try:
        # 获取查询参数
        mesh_name = request.args.get("mesh_name")
        status = request.args.get("status")
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        limit = int(request.args.get("limit", 10))

        # 获取所有训练历史
        all_trainings = training_manager.list_all_training_history()

        # 应用筛选条件
        filtered_trainings = []

        for training in all_trainings:
            metadata = training.get("metadata", {})

            # 按mesh名称筛选
            if mesh_name and metadata.get("mesh_name") != mesh_name:
                continue

            # 按状态筛选
            if status and metadata.get("status") != status:
                continue

            # 按日期筛选 (简化实现，实际可能需要更复杂的日期解析)
            if start_date or end_date:
                training_date = metadata.get("start_datetime", "")
                if start_date and training_date < start_date:
                    continue
                if end_date and training_date > end_date:
                    continue

            filtered_trainings.append(training)

            # 限制返回数量
            if len(filtered_trainings) >= limit:
                break

        return jsonify({
            "success": True,
            "trainings": filtered_trainings,
            "count": len(filtered_trainings),
            "total_available": len(all_trainings),
            "filters_applied": {
                "mesh_name": mesh_name,
                "status": status,
                "start_date": start_date,
                "end_date": end_date,
                "limit": limit
            }
        })
    except Exception as exc:
        current_app.logger.error(f"Exception in search_training_history: {exc}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== Exception in search_training_history ===")
        print(f"Error: {exc}")
        traceback.print_exc()
        print(f"=== End Error ===")
        return jsonify({
            "success": False,
            "error": f"搜索训练历史失败: {str(exc)}",
            "trainings": [],
            "count": 0
        }), 500


@training_history_bp.route("/health", methods=["GET"])
def history_health_check():
    """
    历史管理API健康检查

    返回:
        JSON响应表示历史管理服务状态
    """
    try:
        current_training_id = training_manager.get_current_training_id()
        history_count = len(training_manager.list_all_training_history())

        return jsonify({
            "status": "healthy",
            "service": "training-history-api",
            "current_training_active": current_training_id is not None,
            "current_training_id": current_training_id,
            "total_history_count": history_count,
            "timestamp": __import__("time").time()
        })
    except Exception as exc:
        current_app.logger.error(f"Exception in history_health_check: {exc}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== Exception in history_health_check ===")
        print(f"Error: {exc}")
        traceback.print_exc()
        print(f"=== End Error ===")
        return jsonify({
            "status": "unhealthy",
            "service": "training-history-api",
            "error": str(exc),
            "timestamp": __import__("time").time()
        }), 500
