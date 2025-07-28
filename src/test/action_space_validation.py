"""
Action Space验证测试模块
用于测试当前action space设置的合理性和有效性
完全使用src中的现有代码和类
"""
import math
import numpy as np
import random
import copy
from typing import Dict, List, Tuple, Optional

from src.utils.importer import MeshImporter
from src.geometry import Mesh, Boundary
from src.rl.action.action_manager import ActionManager
from src.rl.action.type1 import ActionType1
from src.utils.angle import decode_coordinate, valid_element_angle


class ActionSpaceValidator:
    """Action Space验证器"""
    
    def __init__(self, mesh_name: str = "Square", subfolder: str = "mesh"):
        """
        初始化验证器
        
        Args:
            mesh_name: 测试用的mesh文件名
            subfolder: mesh文件所在的子文件夹
        """
        self.mesh_name = mesh_name
        self.subfolder = subfolder
        self.importer = MeshImporter()
        self.action_manager = ActionManager(alpha=2, n=2, max_steps=1000)
        self.type1_action = ActionType1()
        
        # 加载初始边界
        self.initial_boundary = None
        self._load_boundary()
        
        # 获取action space信息
        self.action_space = self.action_manager.get_action_space()
        
    def _load_boundary(self):
        """加载边界文件"""
        try:
            self.initial_boundary = self.importer.load_boundary_by_name(
                self.mesh_name, self.subfolder
            )
            print(f"✅ 成功加载boundary文件: {self.mesh_name}.txt")
            print(f"   边界顶点数: {self.initial_boundary.size()}")
            vertices = self.initial_boundary.get_vertices()
            print(f"   顶点坐标示例: {vertices[:3]}...")
        except Exception as e:
            print(f"❌ 加载boundary文件失败: {e}")
            raise
    
    def get_action_space_info(self) -> Dict:
        """获取action space信息"""
        return {
            "type_logit_range": (self.action_space.low[0], self.action_space.high[0]),
            "r_coord_range": (self.action_space.low[1], self.action_space.high[1]),
            "theta_coord_range": (self.action_space.low[2], self.action_space.high[2]),
            "shape": self.action_space.shape,
            "dtype": self.action_space.dtype
        }
    
    def print_action_space_info(self):
        """打印action space信息"""
        info = self.get_action_space_info()
        print(f"✅ Action Space信息:")
        print(f"   type_logit范围: [{info['type_logit_range'][0]:.1f}, {info['type_logit_range'][1]:.1f}]")
        print(f"   r_coord范围: [{info['r_coord_range'][0]:.1f}, {info['r_coord_range'][1]:.1f}]")
        print(f"   theta_coord范围: [{info['theta_coord_range'][0]:.1f}, {info['theta_coord_range'][1]:.1f}]")
        print(f"   theta范围(度): [{math.degrees(info['theta_coord_range'][0]):.1f}°, {math.degrees(info['theta_coord_range'][1]):.1f}°]")
        print(f"   形状: {info['shape']}, 数据类型: {info['dtype']}")
    
    def test_type1_action_validity(
        self, 
        r_coord: float, 
        theta_coord: float, 
        reference_vertex_idx: Optional[int] = None
    ) -> Tuple[bool, str, Optional[Tuple[float, float]]]:
        """
        测试单个type1动作的有效性
        
        Args:
            r_coord: 径向坐标
            theta_coord: 角度坐标
            reference_vertex_idx: 参考顶点索引，如果为None则使用默认值
            
        Returns:
            (is_valid, failure_reason, decoded_coords)
        """
        try:
            # 创建边界和网格的深拷贝
            test_boundary = copy.deepcopy(self.initial_boundary)
            test_mesh = Mesh(test_boundary)
            
            # 获取参考顶点索引
            if reference_vertex_idx is None:
                reference_vertex_idx = test_boundary.get_ref_vertex()
            
            # 构造action向量，固定type_logit为0确保使用type1
            action = np.array([0.0, r_coord, theta_coord])
            
            # 使用ActionManager的decode方法
            action_name, action_instance, new_coords, ref_idx = self.action_manager.decode_action(
                action, test_boundary, reference_vertex_idx
            )
            
            # 验证确实使用了type1
            if action_name != "type1":
                return False, f"unexpected_action_type: {action_name}", None
            
            # 检查动作有效性
            is_valid = self.action_manager.is_valid(
                test_boundary, ref_idx, action_instance, action_name, new_coords
            )
            
            decoded_coords = new_coords[0] if new_coords else None
            
            if is_valid:
                return True, "valid", decoded_coords
            else:
                # 详细分析失败原因
                failure_reason = self._analyze_failure_reason(
                    test_boundary, ref_idx, action_instance, new_coords
                )
                return False, failure_reason, decoded_coords
                
        except Exception as e:
            return False, f"error: {str(e)}", None
    
    def _analyze_failure_reason(
        self, 
        boundary: Boundary, 
        ref_idx: int, 
        action_instance: ActionType1, 
        new_coords: List[Tuple[float, float]]
    ) -> str:
        """分析动作失效的具体原因"""
        try:
            if not new_coords:
                return "no_coords"
            
            quadrilateral = action_instance.get_element(boundary, ref_idx, new_coords[0])
            v0, v3, v2, v1 = quadrilateral
            
            # 检查角度是否有效
            if not valid_element_angle(quadrilateral):
                return "angle_invalid"
            
            # 检查新顶点是否在边界内
            if not boundary.vertex_inside_boundary(v2):
                return "outside_boundary"
            
            # 检查边是否在边界内
            edge_V1_V2 = (v1, v2)
            edge_V2_V3 = (v2, v3)
            
            if not boundary.edge_inside_boundary(edge_V1_V2):
                return "edge_outside"
            if not boundary.edge_inside_boundary(edge_V2_V3):
                return "edge_outside"
            
            # 检查边是否相交
            if boundary.edge_cross(edge_V1_V2):
                return "edge_cross"
            if boundary.edge_cross(edge_V2_V3):
                return "edge_cross"
            
            return "other_error"
            
        except Exception as e:
            return f"analysis_error: {str(e)}"
    
    def run_random_validation_test(self, num_tests: int = 1000, seed: int = 42) -> Dict:
        """
        运行随机验证测试
        
        Args:
            num_tests: 测试次数
            seed: 随机种子
            
        Returns:
            测试结果字典
        """
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        print(f"🔄 开始进行{num_tests}次随机Type1动作测试...")
        
        # 获取action space范围
        r_min, r_max = self.action_space.low[1], self.action_space.high[1]
        theta_min, theta_max = self.action_space.low[2], self.action_space.high[2]
        
        # 统计变量
        valid_count = 0
        failure_reasons = {
            "angle_invalid": 0,
            "outside_boundary": 0,
            "edge_cross": 0,
            "edge_outside": 0,
            "error": 0,
            "other": 0
        }
        
        valid_coords = []
        invalid_coords = []
        
        # 执行测试
        for i in range(num_tests):
            # 随机生成r和theta
            r_coord = random.uniform(r_min, r_max)
            theta_coord = random.uniform(theta_min, theta_max)
            
            # 测试有效性
            is_valid, reason, coords = self.test_type1_action_validity(r_coord, theta_coord)
            
            if is_valid:
                valid_count += 1
                if coords:
                    valid_coords.append(coords)
            else:
                # 归类失败原因
                if reason.startswith("error") or reason.startswith("analysis_error"):
                    failure_reasons["error"] += 1
                elif reason in failure_reasons:
                    failure_reasons[reason] += 1
                else:
                    failure_reasons["other"] += 1
                
                if coords:
                    invalid_coords.append(coords)
            
            # 显示进度
            if (i + 1) % (num_tests // 5) == 0:
                progress = (i + 1) / num_tests * 100
                current_success_rate = valid_count / (i + 1) * 100
                print(f"   进度: {i+1}/{num_tests} ({progress:.0f}%), 当前有效率: {current_success_rate:.1f}%")
        
        # 返回结果
        return {
            "total_tests": num_tests,
            "valid_count": valid_count,
            "invalid_count": num_tests - valid_count,
            "success_rate": valid_count / num_tests,
            "failure_reasons": failure_reasons,
            "valid_coords": valid_coords,
            "invalid_coords": invalid_coords,
            "action_space_info": self.get_action_space_info()
        }
    
    def analyze_coordinate_decoding(self, test_cases: Optional[List[Tuple[float, float]]] = None):
        """
        分析坐标解码过程
        
        Args:
            test_cases: 测试用例列表，格式为[(r, theta), ...]
        """
        print(f"\n{'='*50}")
        print("🔍 分析坐标解码过程")
        print(f"{'='*50}")
        
        # 获取参考点信息
        ref_idx = self.initial_boundary.get_ref_vertex()
        ref_v = self.initial_boundary.get_vertex_by_index(ref_idx)
        right_neighbor_v = self.initial_boundary.get_vertex_by_index(ref_idx - 1)
        base_len = self.initial_boundary.get_avg_neighbor_length(ref_idx, 2)
        scale_factor = 1.0 / base_len if base_len > 0 else 1.0
        
        print(f"参考点信息:")
        print(f"  参考顶点索引: {ref_idx}")
        print(f"  参考顶点坐标: {ref_v}")
        print(f"  右邻居坐标: {right_neighbor_v}")
        print(f"  平均邻居长度: {base_len:.3f}")
        print(f"  缩放因子: {scale_factor:.6f}")
        
        # 默认测试用例
        if test_cases is None:
            test_cases = [
                (0.1, 0.1),    # 小值
                (0.5, 0.5),    # 中等值
                (1.0, 1.0),    # 大值
                (1.5, 1.5),    # action space上限
                (0.5, 0.0),    # 沿参考方向
                (0.5, -1.0),   # 负角度
            ]
        
        print(f"\n测试坐标解码:")
        for i, (r, theta) in enumerate(test_cases):
            print(f"\n测试{i+1}: r={r:.1f}, theta={theta:.3f}rad ({math.degrees(theta):.1f}°)")
            
            try:
                # 解码坐标
                new_x, new_y = decode_coordinate(ref_v, right_neighbor_v, scale_factor, r, theta)
                print(f"  解码后全局坐标: ({new_x:.2f}, {new_y:.2f})")
                
                # 测试这个坐标的有效性
                is_valid, reason, _ = self.test_type1_action_validity(r, theta)
                print(f"  动作有效性: {'有效' if is_valid else '无效'}")
                if not is_valid:
                    print(f"  失败原因: {reason}")
                    
            except Exception as e:
                print(f"  错误: {str(e)}")
    
    def print_validation_results(self, results: Dict):
        """打印验证结果"""
        print(f"\n{'='*50}")
        print(f"测试结果汇总")
        print(f"{'='*50}")
        
        total = results["total_tests"]
        valid = results["valid_count"]
        invalid = results["invalid_count"]
        success_rate = results["success_rate"]
        
        print(f"✅ 总测试数: {total}")
        print(f"✅ 有效动作: {valid} ({success_rate*100:.1f}%)")
        print(f"❌ 无效动作: {invalid} ({(1-success_rate)*100:.1f}%)")
        
        if invalid > 0:
            print(f"\n📊 无效动作原因分析:")
            failure_reasons = results["failure_reasons"]
            reason_names = {
                "angle_invalid": "四边形角度无效",
                "outside_boundary": "新顶点在边界外",
                "edge_cross": "边相交", 
                "edge_outside": "边在边界外",
                "error": "处理错误",
                "other": "其他原因"
            }
            
            for reason, count in failure_reasons.items():
                if count > 0:
                    percentage = count / invalid * 100
                    print(f"   {reason_names.get(reason, reason)}: {count} ({percentage:.1f}%)")
        
        # 评估action space合理性
        print(f"\n📈 Action Space合理性评估:")
        if success_rate >= 0.3:
            print(f"   ✅ 成功率{success_rate*100:.1f}% - Action Space范围较为合理")
        elif success_rate >= 0.1:
            print(f"   ⚠️  成功率{success_rate*100:.1f}% - Action Space范围可能过大")
        else:
            print(f"   ❌ 成功率{success_rate*100:.1f}% - Action Space范围明显不合理")
        
        # 分析主要失败原因并给出建议
        if invalid > 0:
            failure_reasons = results["failure_reasons"]
            max_failure_reason = max(failure_reasons.items(), key=lambda x: x[1])
            if max_failure_reason[1] / invalid > 0.5:
                suggestions = {
                    "angle_invalid": "建议减小r坐标的最大值，避免生成过大的四边形",
                    "outside_boundary": "建议减小r坐标的范围，确保新顶点在边界内",
                    "edge_cross": "建议限制theta坐标范围，避免生成交叉的边",
                    "edge_outside": "建议同时调整r和theta范围",
                    "error": "需要进一步调试代码逻辑",
                    "other": "需要详细分析其他失败原因"
                }
                
                reason_names = {
                    "angle_invalid": "四边形角度无效",
                    "outside_boundary": "新顶点在边界外", 
                    "edge_cross": "边相交",
                    "edge_outside": "边在边界外",
                    "error": "处理错误",
                    "other": "其他原因"
                }
                
                main_reason = max_failure_reason[0]
                print(f"   💡 主要失败原因: {reason_names.get(main_reason, main_reason)}")
                print(f"   💡 建议: {suggestions.get(main_reason, '需要进一步分析')}")
        
        # Action Space范围分析
        action_info = results["action_space_info"]
        theta_range = action_info["theta_coord_range"]
        theta_coverage = (theta_range[1] - theta_range[0]) / (2 * math.pi) * 100
        
        print(f"\n🎯 Action Space范围分析:")
        print(f"   theta覆盖率: {theta_coverage:.1f}% (完整圆周)")
        if theta_coverage < 50:
            print(f"   ⚠️  theta范围过小，建议扩展到[-π, π]")
        
        # r范围分析
        r_max = action_info["r_coord_range"][1]
        # 根据边界获取参考距离
        ref_idx = self.initial_boundary.get_ref_vertex()
        base_len = self.initial_boundary.get_avg_neighbor_length(ref_idx, 2)
        scale_factor = 1.0 / base_len if base_len > 0 else 1.0
        actual_max_distance = r_max / scale_factor
        
        print(f"   r_max对应实际距离: {actual_max_distance:.2f}")
        print(f"   相对于平均边长: {actual_max_distance/base_len:.2f}倍")
        
        if actual_max_distance > base_len * 2:
            print(f"   ⚠️  r_max可能过大，建议限制在{base_len*1.5/scale_factor:.1f}以内")
    
    def run_comprehensive_test(self, num_tests: int = 1000, seed: int = 42):
        """运行综合测试"""
        print("=== Action Space综合验证测试 ===")
        
        # 打印基本信息
        self.print_action_space_info()
        
        # 分析坐标解码
        self.analyze_coordinate_decoding()
        
        # 运行随机验证测试
        results = self.run_random_validation_test(num_tests, seed)
        
        # 打印结果
        self.print_validation_results(results)
        
        return results


def run_validation_test(mesh_name: str = "Square", num_tests: int = 1000):
    """
    运行验证测试的便捷函数
    
    Args:
        mesh_name: 测试用的mesh文件名
        num_tests: 测试次数
    """
    try:
        validator = ActionSpaceValidator(mesh_name)
        results = validator.run_comprehensive_test(num_tests)
        return results
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        return None


if __name__ == "__main__":
    # 直接运行测试
    run_validation_test()