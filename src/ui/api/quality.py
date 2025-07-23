import traceback
from flask import Blueprint, jsonify, request, current_app
from src.quality.quality import QualityManager

quality_bp = Blueprint("quality", __name__, url_prefix="/quality")

quality_manager = QualityManager()


@quality_bp.route("/methods", methods=["GET"])
def get_quality_methods():
    """
    获取所有可用的质量测量方法
    
    返回:
        JSON响应包含质量方法名称列表和详细信息
    """
    try:
        methods = quality_manager.get_available_methods()
        method_info = quality_manager.get_method_info()
        
        return jsonify({
            "methods": methods,
            "method_info": method_info,
            "count": len(methods),
            "success": True
        })
        
    except Exception as exc:
        current_app.logger.error(f"Exception in get_quality_methods: {exc}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            "error": f"获取质量方法失败: {str(exc)}",
            "methods": [],
            "method_info": {},
            "count": 0,
            "success": False
        }), 500


@quality_bp.route("/calculate", methods=["POST"])
def calculate_quality():
    """
    计算给定四边形顶点的质量得分
    
    请求体:
        {
            "vertices": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
            "method": "method_name"
        }
        
    返回:
        JSON响应包含质量得分
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "请求数据为空",
                "success": False
            }), 400
            
        vertices = data.get("vertices")
        method = data.get("method")
        
        if not vertices:
            return jsonify({
                "error": "缺少顶点数据",
                "success": False
            }), 400
            
        if not method:
            return jsonify({
                "error": "缺少方法名称",
                "success": False
            }), 400
            
        if len(vertices) != 4:
            return jsonify({
                "error": "顶点数量必须为4个",
                "success": False
            }), 400
            
        for i, vertex in enumerate(vertices):
            if not isinstance(vertex, list) or len(vertex) != 2:
                return jsonify({
                    "error": f"顶点{i+1}格式错误，应为[x, y]格式",
                    "success": False
                }), 400
                
        vertices_tuples = [(float(v[0]), float(v[1])) for v in vertices]
        
        # Handle optional parameters (like gamma for hybrid method)
        gamma = data.get("gamma", 1.0)  # Default gamma for hybrid method
        
        quality_score = quality_manager.calculate_quality(method, vertices_tuples, gamma=gamma)
        
        return jsonify({
            "quality_score": quality_score,
            "method": method,
            "vertices": vertices,
            "success": True
        })
        
    except ValueError as exc:
        current_app.logger.error(f"ValueError in calculate_quality: {exc}")
        return jsonify({
            "error": str(exc),
            "success": False
        }), 400
        
    except Exception as exc:
        current_app.logger.error(f"Exception in calculate_quality: {exc}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            "error": f"质量计算失败: {str(exc)}",
            "success": False
        }), 500


@quality_bp.route("/health", methods=["GET"])
def health_check():
    """
    健康检查端点
    
    返回:
        JSON响应表示服务状态
    """
    try:
        methods_count = len(quality_manager.get_available_methods())
        return jsonify({
            "status": "healthy",
            "service": "quality-api",
            "available_methods": methods_count,
            "timestamp": __import__("time").time()
        })
    except Exception as exc:
        current_app.logger.error(f"Exception in health_check: {exc}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            "status": "unhealthy",
            "service": "quality-api",
            "error": str(exc),
            "timestamp": __import__("time").time()
        }), 500