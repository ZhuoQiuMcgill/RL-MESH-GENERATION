from abc import ABC, abstractmethod


class ActionType(ABC):
    """动作类型的抽象基类"""

    @abstractmethod
    def execute(self, mesh, boundary, **kwargs):
        """
        执行具体的几何操作来修改网格和边界。
        返回更新后的 (mesh, boundary, generated_element)。
        """
        pass

    @abstractmethod
    def is_valid(self, boundary, **kwargs):
        """
        检查动作在当前边界下是否有效。
        """
        pass
