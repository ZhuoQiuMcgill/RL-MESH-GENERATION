from action import ActionType


class ActionType0(ActionType):
    """
    实现Type 0动作：不增加新顶点，直接连接边界上的四个点形成一个四边形。
    对应论文中的 Figure 5(a)。
    """

    def execute(self, mesh, boundary, reference_vertex_V0_idx):
        # TODO: 实现Type 0动作的逻辑。
        # 1. 以参考点 V0 为起点，找到它在边界上的两个邻居点 V1 和 V3。
        # 2. V2 是 V1 在边界上的邻居点 (V1的下一个点)。
        # 3. 在 mesh 对象中，使用这四个顶点 (V0, V1, V2, V3) 创建一个新的四边形元素。
        # 4. 更新 boundary 对象：
        #    - 移除旧的边界边 (V0-V1), (V1-V2), (V2-V3)。
        #    - 添加新的内部边 (V0-V3) 作为新的边界。
        # 5. 返回更新后的 (mesh, boundary) 和新生成的元素 (generated_element)。
        pass

    def is_valid(self, boundary, reference_vertex_V0_idx):
        # TODO: 检查Type 0动作的有效性。
        # 1. 检查新生成的对角线 (V0-V3) 是否与边界的其他部分相交。
        # 2. 检查生成的四边形是否是凸的，并且没有翻转。
        # 3. 返回 True 或 False。
        pass
