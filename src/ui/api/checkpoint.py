"""
Checkpoint管理API端点

提供checkpoint相关的REST API接口，包括列表查询、详细信息获取等功能。
"""

import traceback
import time
from flask import Blueprint, jsonify, request, current_app

from src.utils.checkpoint_manager import get_checkpoint_manager

checkpoint_bp = Blueprint("checkpoint", __name__, url_prefix="/checkpoint")


@checkpoint_bp.route("/list", methods=["GET"])
def list_checkpoints():
    """
    获取所有可用的checkpoint列表

    GET /checkpoint/list

    Returns:
        JSON响应包含checkpoint列表
    """
    try:
        checkpoint_manager = get_checkpoint_manager()
        checkpoints = checkpoint_manager.list_available_checkpoints()

        current_app.logger.info(f"查询到 {len(checkpoints)} 个checkpoints")

        return jsonify({
            "checkpoints": checkpoints,
            "count": len(checkpoints),
            "success": True
        }), 200

    except Exception as e:
        error_msg = f"获取checkpoint列表失败: {str(e)}"
        current_app.logger.error(f"Checkpoint列表查询异常: {error_msg}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== Checkpoint列表查询异常 ===")
        print(f"Error: {e}")
        traceback.print_exc()
        print(f"=== End Error ===")

        return jsonify({
            "error": error_msg,
            "checkpoints": [],
            "count": 0,
            "success": False
        }), 500


@checkpoint_bp.route("/info/<checkpoint_name>", methods=["GET"])
def get_checkpoint_info(checkpoint_name: str):
    """
    获取指定checkpoint的详细信息

    GET /checkpoint/info/{checkpoint_name}

    Args:
        checkpoint_name: checkpoint名称（不包含.pth扩展名）

    Returns:
        JSON响应包含checkpoint的详细信息
    """
    try:
        checkpoint_manager = get_checkpoint_manager()
        checkpoint_info = checkpoint_manager.get_checkpoint_info(checkpoint_name)

        if checkpoint_info is None:
            return jsonify({
                "error": f"Checkpoint不存在: {checkpoint_name}",
                "success": False
            }), 404

        current_app.logger.info(f"获取checkpoint信息: {checkpoint_name}")

        return jsonify({
            "checkpoint_info": checkpoint_info,
            "success": True
        }), 200

    except Exception as e:
        error_msg = f"获取checkpoint信息失败: {str(e)}"
        current_app.logger.error(f"Checkpoint信息查询异常: {error_msg}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== Checkpoint信息查询异常 ===")
        print(f"Error: {e}")
        traceback.print_exc()
        print(f"=== End Error ===")

        return jsonify({
            "error": error_msg,
            "success": False
        }), 500


@checkpoint_bp.route("/validate/<checkpoint_name>", methods=["GET"])
def validate_checkpoint(checkpoint_name: str):
    """
    验证指定checkpoint是否有效

    GET /checkpoint/validate/{checkpoint_name}

    Args:
        checkpoint_name: checkpoint名称（不包含.pth扩展名）

    Returns:
        JSON响应包含验证结果
    """
    try:
        checkpoint_manager = get_checkpoint_manager()
        is_valid = checkpoint_manager.validate_checkpoint(checkpoint_name)

        current_app.logger.info(f"验证checkpoint: {checkpoint_name}, 结果: {is_valid}")

        return jsonify({
            "checkpoint_name": checkpoint_name,
            "is_valid": is_valid,
            "success": True
        }), 200

    except Exception as e:
        error_msg = f"验证checkpoint失败: {str(e)}"
        current_app.logger.error(f"Checkpoint验证异常: {error_msg}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== Checkpoint验证异常 ===")
        print(f"Error: {e}")
        traceback.print_exc()
        print(f"=== End Error ===")

        return jsonify({
            "error": error_msg,
            "checkpoint_name": checkpoint_name,
            "is_valid": False,
            "success": False
        }), 500


@checkpoint_bp.route("/delete/<checkpoint_name>", methods=["DELETE"])
def delete_checkpoint(checkpoint_name: str):
    """
    删除指定的checkpoint

    DELETE /checkpoint/delete/{checkpoint_name}

    Args:
        checkpoint_name: checkpoint名称（不包含.pth扩展名）

    Returns:
        JSON响应包含删除结果
    """
    try:
        checkpoint_manager = get_checkpoint_manager()

        # 检查checkpoint是否存在
        if not checkpoint_manager.validate_checkpoint(checkpoint_name):
            return jsonify({
                "error": f"Checkpoint不存在或无效: {checkpoint_name}",
                "success": False
            }), 404

        # 删除checkpoint
        success = checkpoint_manager.delete_checkpoint(checkpoint_name)

        if success:
            current_app.logger.info(f"成功删除checkpoint: {checkpoint_name}")
            return jsonify({
                "message": f"Checkpoint已删除: {checkpoint_name}",
                "checkpoint_name": checkpoint_name,
                "success": True
            }), 200
        else:
            return jsonify({
                "error": f"删除checkpoint失败: {checkpoint_name}",
                "checkpoint_name": checkpoint_name,
                "success": False
            }), 500

    except Exception as e:
        error_msg = f"删除checkpoint失败: {str(e)}"
        current_app.logger.error(f"Checkpoint删除异常: {error_msg}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== Checkpoint删除异常 ===")
        print(f"Error: {e}")
        traceback.print_exc()
        print(f"=== End Error ===")

        return jsonify({
            "error": error_msg,
            "checkpoint_name": checkpoint_name,
            "success": False
        }), 500


@checkpoint_bp.route("/copy", methods=["POST"])
def copy_checkpoint_from_history():
    """
    从历史训练目录复制checkpoint到checkpoints目录

    POST /checkpoint/copy
    Content-Type: application/json

    Body:
        {
            "training_id": "sac_20250115_143022_simple_square",
            "checkpoint_name": "my_checkpoint"  // 可选，默认使用training_id
        }

    Returns:
        JSON响应包含复制结果
    """
    try:
        data = request.get_json() or {}
        training_id = data.get("training_id")
        checkpoint_name = data.get("checkpoint_name")

        if not training_id:
            return jsonify({
                "error": "training_id参数是必需的",
                "success": False
            }), 400

        checkpoint_manager = get_checkpoint_manager()

        # 执行复制操作
        success = checkpoint_manager.copy_checkpoint_from_history(training_id, checkpoint_name)

        if success:
            final_checkpoint_name = checkpoint_name or training_id
            current_app.logger.info(f"成功复制checkpoint: {training_id} -> {final_checkpoint_name}")
            return jsonify({
                "message": f"Checkpoint已复制: {final_checkpoint_name}",
                "training_id": training_id,
                "checkpoint_name": final_checkpoint_name,
                "success": True
            }), 200
        else:
            return jsonify({
                "error": f"复制checkpoint失败: {training_id}",
                "training_id": training_id,
                "success": False
            }), 500

    except Exception as e:
        error_msg = f"复制checkpoint失败: {str(e)}"
        current_app.logger.error(f"Checkpoint复制异常: {error_msg}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== Checkpoint复制异常 ===")
        print(f"Error: {e}")
        traceback.print_exc()
        print(f"=== End Error ===")

        return jsonify({
            "error": error_msg,
            "success": False
        }), 500


@checkpoint_bp.route("/health", methods=["GET"])
def health_check():
    """
    Checkpoint服务健康检查

    GET /checkpoint/health

    Returns:
        JSON响应表示服务健康状态
    """
    try:
        checkpoint_manager = get_checkpoint_manager()
        checkpoints = checkpoint_manager.list_available_checkpoints()

        return jsonify({
            "status": "healthy",
            "service": "checkpoint-api",
            "checkpoint_count": len(checkpoints),
            "timestamp": time.time()
        }), 200

    except Exception as e:
        error_msg = f"健康检查失败: {str(e)}"
        current_app.logger.error(f"Checkpoint健康检查异常: {error_msg}")
        current_app.logger.error(traceback.format_exc())
        print(f"=== Checkpoint健康检查异常 ===")
        print(f"Error: {e}")
        traceback.print_exc()
        print(f"=== End Error ===")

        return jsonify({
            "status": "unhealthy",
            "service": "checkpoint-api",
            "error": error_msg,
            "timestamp": time.time()
        }), 500


# 错误处理
@checkpoint_bp.errorhandler(404)
def not_found(error):
    """处理404错误"""
    return jsonify({
        "error": "Endpoint not found",
        "success": False
    }), 404


@checkpoint_bp.errorhandler(405)
def method_not_allowed(error):
    """处理405错误"""
    return jsonify({
        "error": "Method not allowed",
        "success": False
    }), 405


@checkpoint_bp.errorhandler(500)
def internal_error(error):
    """处理500错误"""
    return jsonify({
        "error": "Internal server error",
        "success": False
    }), 500
