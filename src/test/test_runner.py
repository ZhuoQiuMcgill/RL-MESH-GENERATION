"""
测试运行器
用于统一管理和运行各种测试
"""
import sys
import os
from typing import Dict, List, Optional

from .action_space_validation import ActionSpaceValidator, run_validation_test


class TestRunner:
    """测试运行器类"""
    
    def __init__(self):
        """初始化测试运行器"""
        self.available_tests = {
            "action_space_validation": {
                "description": "Action Space有效性验证测试",
                "function": self._run_action_space_validation,
                "default_params": {"mesh_name": "Square", "num_tests": 1000}
            }
        }
    
    def list_available_tests(self):
        """列出所有可用的测试"""
        print("📋 可用测试列表:")
        print("=" * 50)
        for test_name, test_info in self.available_tests.items():
            print(f"  {test_name}:")
            print(f"    描述: {test_info['description']}")
            print(f"    默认参数: {test_info['default_params']}")
            print()
    
    def run_test(self, test_name: str, **kwargs) -> Optional[Dict]:
        """
        运行指定的测试
        
        Args:
            test_name: 测试名称
            **kwargs: 测试参数
            
        Returns:
            测试结果字典，如果测试失败则返回None
        """
        if test_name not in self.available_tests:
            print(f"❌ 未找到测试: {test_name}")
            print("可用测试:", list(self.available_tests.keys()))
            return None
        
        test_info = self.available_tests[test_name]
        test_function = test_info["function"]
        
        # 合并默认参数和用户参数
        params = test_info["default_params"].copy()
        params.update(kwargs)
        
        print(f"🚀 运行测试: {test_name}")
        print(f"📝 描述: {test_info['description']}")
        print(f"⚙️  参数: {params}")
        print("=" * 50)
        
        try:
            result = test_function(**params)
            if result is not None:
                print(f"✅ 测试 {test_name} 完成")
            else:
                print(f"❌ 测试 {test_name} 失败")
            return result
        except Exception as e:
            print(f"❌ 测试 {test_name} 执行异常: {e}")
            return None
    
    def run_all_tests(self, **global_kwargs):
        """
        运行所有测试
        
        Args:
            **global_kwargs: 应用于所有测试的全局参数
        """
        print("🏃 运行所有测试...")
        print("=" * 50)
        
        all_results = {}
        
        for test_name in self.available_tests.keys():
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = self.run_test(test_name, **global_kwargs)
            all_results[test_name] = result
        
        # 汇总结果
        print(f"\n{'='*50}")
        print("📊 所有测试结果汇总")
        print(f"{'='*50}")
        
        success_count = 0
        total_count = len(all_results)
        
        for test_name, result in all_results.items():
            status = "✅ 成功" if result is not None else "❌ 失败"
            print(f"  {test_name}: {status}")
            if result is not None:
                success_count += 1
        
        print(f"\n总计: {success_count}/{total_count} 个测试成功")
        
        return all_results
    
    def _run_action_space_validation(self, mesh_name: str = "Square", num_tests: int = 1000):
        """运行Action Space验证测试"""
        return run_validation_test(mesh_name, num_tests)


def main():
    """主函数，用于命令行运行"""
    runner = TestRunner()
    
    if len(sys.argv) == 1:
        # 没有参数，列出可用测试并运行默认测试
        runner.list_available_tests()
        print("运行默认测试...")
        result = runner.run_test("action_space_validation")
        
    elif len(sys.argv) == 2:
        test_name = sys.argv[1]
        if test_name == "list":
            runner.list_available_tests()
        elif test_name == "all":
            runner.run_all_tests()
        else:
            # 运行指定测试
            result = runner.run_test(test_name)
    
    else:
        print("用法:")
        print("  python -m src.test.test_runner              # 运行默认测试")
        print("  python -m src.test.test_runner list         # 列出可用测试")
        print("  python -m src.test.test_runner all          # 运行所有测试")
        print("  python -m src.test.test_runner <test_name>  # 运行指定测试")


if __name__ == "__main__":
    main()