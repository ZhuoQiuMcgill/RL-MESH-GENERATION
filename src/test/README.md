# Test Module

本模块包含项目中各个组件的功能验证测试。

## 模块结构

```
src/test/
├── __init__.py                    # 模块初始化文件
├── action_space_validation.py     # Action Space验证测试
├── test_runner.py                 # 测试运行器
└── README.md                      # 本文档
```

## 可用测试

### 1. Action Space验证测试 (`action_space_validation.py`)

验证当前Action Space设置的合理性和有效性，包括：

- **坐标解码验证**: 测试`normalize_coordinates`和`decode_coordinate`函数的正确性
- **Type1动作有效性测试**: 随机生成1000个Type1动作，统计有效率
- **失败原因分析**: 详细分析无效动作的具体原因
- **Action Space合理性评估**: 基于测试结果给出改进建议

#### 主要功能

- `ActionSpaceValidator`: 核心验证器类
- `run_validation_test()`: 便捷测试函数

#### 使用方法

```python
# 方法1: 使用便捷函数
from src.test import run_validation_test
results = run_validation_test(mesh_name="Square", num_tests=1000)

# 方法2: 使用验证器类
from src.test import ActionSpaceValidator
validator = ActionSpaceValidator("Square")
results = validator.run_comprehensive_test(num_tests=1000)
```

### 2. 测试运行器 (`test_runner.py`)

统一管理和运行各种测试的工具。

#### 功能特性

- 列出所有可用测试
- 运行单个或所有测试
- 统一的参数管理和结果汇总

#### 使用方法

```python
from src.test import TestRunner

runner = TestRunner()

# 列出可用测试
runner.list_available_tests()

# 运行单个测试
results = runner.run_test("action_space_validation", mesh_name="Square", num_tests=1000)

# 运行所有测试
all_results = runner.run_all_tests(num_tests=500)
```

## 运行测试

由于存在循环导入问题，建议使用以下方式运行测试：

### 方法1: 创建独立脚本

```python
# test_script.py
import sys
import os
import importlib.util

# 直接加载测试模块
spec = importlib.util.spec_from_file_location(
    "action_space_validation", 
    "src/test/action_space_validation.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# 运行测试
results = module.run_validation_test("Square", 1000)
```

### 方法2: 在项目根目录直接运行

```python
# 在Python REPL中
import sys
sys.path.append('src')

# 延迟导入
try:
    from test.action_space_validation import run_validation_test
    results = run_validation_test("Square", 1000)
except ImportError:
    print("存在循环导入，请使用方法1")
```

## 测试结果解读

### Action Space验证测试结果

典型的测试输出包括：

1. **基本信息**
   - Action Space范围
   - 边界顶点信息
   - 参考点和缩放因子

2. **坐标解码分析**
   - 特定坐标的解码结果
   - 每个测试点的有效性

3. **随机测试统计**
   - 总测试数和有效率
   - 失败原因分布
   - 主要问题和改进建议

4. **Action Space分析**
   - theta覆盖率分析
   - r范围合理性评估

### 当前测试结果显示的问题

**测试发现的主要问题**：
- ✅ `normalize_coordinates`和`decode_coordinate`函数完全正确
- ❌ Action Space范围设置不合理
- ❌ 1000次测试中有效率为0.0%
- ❌ 所有动作都因"四边形角度无效"而失败

**根本原因**：
- theta范围过小，只覆盖47.7%的完整圆周
- 当前theta范围`[-1.5, 1.5]`应该扩展到`[-π, π]`
- 生成的四边形严重变形，违反几何约束

**建议修正**：
```python
# 当前（有问题）
return spaces.Box(low=np.array([-1, 0, -1.5]), high=np.array([1, 1.5, 1.5]))

# 建议修正为
return spaces.Box(low=np.array([-1, 0, -np.pi]), high=np.array([1, R_max, np.pi]))
```

## 扩展测试

可以轻松添加新的测试模块：

1. 在`src/test/`目录下创建新的测试文件
2. 在`__init__.py`中导入新的测试类/函数
3. 在`test_runner.py`中注册新测试

示例：
```python
# src/test/new_test.py
def run_new_test():
    # 测试逻辑
    return results

# 在test_runner.py中添加
self.available_tests["new_test"] = {
    "description": "新测试的描述",
    "function": self._run_new_test,
    "default_params": {}
}
```

## 注意事项

1. **循环导入问题**: 当前项目存在循环导入，建议使用独立脚本运行测试
2. **测试数据**: 所有测试都使用`data/mesh/`目录下的真实边界文件
3. **测试完整性**: 测试完全使用src中的现有代码，确保结果的真实性
4. **性能考虑**: 大量测试（如1000次）可能需要一些时间，可以调整`num_tests`参数

## 未来改进

1. 解决循环导入问题，实现更好的模块化
2. 添加更多类型的测试（如边界验证、网格质量测试等）
3. 添加可视化功能，显示测试点的分布
4. 集成到CI/CD流程中，自动化测试执行