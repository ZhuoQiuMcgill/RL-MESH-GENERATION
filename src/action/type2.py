from action import ActionType


class ActionType2(ActionType):
    """
    实现Type 2动作：增加两个新顶点V2和V3，形成一个四边形。
    对应论文中的 Figure 5(c)。
    """

    def execute(self, mesh, boundary, reference_vertex_V0_idx, new_vertex_V2_coords, new_vertex_V3_coords):
        # TODO: 实现Type 2动作的逻辑。
        # 1. 以参考点 V0 为起点，找到它在边界上的邻居 V1。
        # 2. 在 mesh 中添加由 new_vertex_V2_coords 和 new_vertex_V3_coords 定义的新顶点 V2 和 V3。
        # 3. 使用 (V0, V1, V2, V3) 创建一个新的四边形元素并添加到 mesh。
        # 4. 更新 boundary 对象：
        #    - 移除旧的边界边 (V0-V1)。
        #    - 添加新的边界边 (V1-V2), (V2-V3), (V3-V0)。
        # 5. 返回更新后的 (mesh, boundary) 和新生成的元素 (generated_element)。
        # 论文中提到，此类型动作仅在特殊情况下使用，可以作为扩展功能实现。 [cite: 141]
        pass

    def is_valid(self, boundary, new_vertex_V2_coords, new_vertex_V3_coords):
        # TODO: 检查Type 2动作的有效性。
        # 1. 检查新顶点 V2 和 V3 是否都位于当前几何域的内部。
        # 2. 检查新生成的边 (V1-V2), (V2-V3), (V3-V0) 是否与边界的其他部分相交。
        # 3. 返回 True 或 False。
        pass
