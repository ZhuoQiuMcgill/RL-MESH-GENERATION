"""
Geometry visualization API blueprint
提供坐标规范化和几何处理功能
"""
import logging
from flask import Blueprint, request, jsonify
from src.utils.angle import normalize_coordinates_cartesian, euclidean_distance

logger = logging.getLogger(__name__)

geometry_bp = Blueprint("geometry", __name__, url_prefix="/geometry")


@geometry_bp.route("/normalize", methods=["POST"])
def normalize_coordinates_endpoint():
    """
    接收一组有序坐标，进行归一化处理后返回
    
    Request Body:
    {
        "coordinates": [[x1, y1], [x2, y2], ..., [xn, yn]]  // n为奇数
    }
    
    Response:
    {
        "status": "success",
        "original_coordinates": [[x1, y1], ...],
        "normalized_coordinates": [[r1, theta1], [r2, theta2], ...],
        "ref_vertex_index": int,  // 中间点索引
        "scale_factor": float
    }
    """
    try:
        data = request.get_json()
        
        if not data or "coordinates" not in data:
            return jsonify({
                "status": "error",
                "message": "Missing 'coordinates' in request body"
            }), 400
        
        coordinates = data["coordinates"]
        
        # 验证输入
        if not isinstance(coordinates, list) or len(coordinates) == 0:
            return jsonify({
                "status": "error",
                "message": "Coordinates must be a non-empty list"
            }), 400
        
        # 检查是否为奇数个点
        if len(coordinates) % 2 == 0:
            return jsonify({
                "status": "error",
                "message": f"Number of points must be odd, got {len(coordinates)}"
            }), 400
        
        # 验证坐标格式
        for i, coord in enumerate(coordinates):
            if not isinstance(coord, list) or len(coord) != 2:
                return jsonify({
                    "status": "error",
                    "message": f"Invalid coordinate format at index {i}. Expected [x, y]"
                }), 400
            try:
                float(coord[0])
                float(coord[1])
            except (ValueError, TypeError):
                return jsonify({
                    "status": "error",
                    "message": f"Coordinate values must be numbers at index {i}"
                }), 400
        
        # 转换为浮点数坐标
        coordinates = [[float(x), float(y)] for x, y in coordinates]
        
        # 计算参考点索引（中间点）
        ref_vertex_index = len(coordinates) // 2
        ref_vertex = coordinates[ref_vertex_index]
        
        # 右邻居点（ref_vertex的前一个点）
        right_neighbor_index = ref_vertex_index - 1
        right_neighbor_vertex = coordinates[right_neighbor_index]
        
        # 计算scale_factor：1除以ref_vertex左右各两条边的平均长度
        edges_for_scale = []
        
        # ref_vertex左边的两条边
        if ref_vertex_index >= 2:
            # 边: coordinates[ref_vertex_index-2] -> coordinates[ref_vertex_index-1]
            edge1_length = euclidean_distance(
                coordinates[ref_vertex_index-2], 
                coordinates[ref_vertex_index-1]
            )
            edges_for_scale.append(edge1_length)
        
        if ref_vertex_index >= 1:
            # 边: coordinates[ref_vertex_index-1] -> coordinates[ref_vertex_index]
            edge2_length = euclidean_distance(
                coordinates[ref_vertex_index-1], 
                coordinates[ref_vertex_index]
            )
            edges_for_scale.append(edge2_length)
        
        # ref_vertex右边的两条边
        if ref_vertex_index < len(coordinates) - 1:
            # 边: coordinates[ref_vertex_index] -> coordinates[ref_vertex_index+1]
            edge3_length = euclidean_distance(
                coordinates[ref_vertex_index], 
                coordinates[ref_vertex_index+1]
            )
            edges_for_scale.append(edge3_length)
        
        if ref_vertex_index < len(coordinates) - 2:
            # 边: coordinates[ref_vertex_index+1] -> coordinates[ref_vertex_index+2]
            edge4_length = euclidean_distance(
                coordinates[ref_vertex_index+1], 
                coordinates[ref_vertex_index+2]
            )
            edges_for_scale.append(edge4_length)
        
        if not edges_for_scale:
            return jsonify({
                "status": "error",
                "message": "Unable to calculate scale factor: insufficient edges"
            }), 400
        
        # 计算平均边长
        average_edge_length = sum(edges_for_scale) / len(edges_for_scale)
        scale_factor = 1.0 / average_edge_length if average_edge_length > 0 else 1.0
        
        logger.info(f"Processing {len(coordinates)} coordinates")
        logger.info(f"Ref vertex index: {ref_vertex_index}, vertex: {ref_vertex}")
        logger.info(f"Right neighbor index: {right_neighbor_index}, vertex: {right_neighbor_vertex}")
        logger.info(f"Scale factor: {scale_factor} (avg edge length: {average_edge_length})")
        
        # 调用normalize_coordinates函数
        normalized_coords = normalize_coordinates_cartesian(
            vertices=coordinates,
            ref_vertex=tuple(ref_vertex),
            right_neighbor_vertex=tuple(right_neighbor_vertex),
            scale_factor=scale_factor
        )
        
        logger.info(f"Normalization completed successfully")
        
        return jsonify({
            "status": "success",
            "original_coordinates": coordinates,
            "normalized_coordinates": normalized_coords,
            "ref_vertex_index": ref_vertex_index,
            "right_neighbor_index": right_neighbor_index,
            "scale_factor": scale_factor,
            "average_edge_length": average_edge_length,
            "edges_used_for_scale": len(edges_for_scale)
        })
        
    except Exception as e:
        logger.error(f"Error in normalize_coordinates_endpoint: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Internal server error: {str(e)}"
        }), 500


@geometry_bp.route("/health", methods=["GET"])
def health_check():
    """健康检查端点"""
    return jsonify({
        "status": "success",
        "message": "Geometry API is running",
        "endpoints": [
            "/geometry/normalize - POST - Normalize coordinates",
            "/geometry/health - GET - Health check"
        ]
    })