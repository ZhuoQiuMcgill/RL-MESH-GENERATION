"""
网格类定义

提供网格的构建、维护和查询功能
"""
import copy


class Mesh:
    def __init__(self, boundary):
        self.boundary = boundary
        self.mesh = self.init_mesh()

    # ──────────────────────── core helpers ────────────────────────
    def init_mesh(self):
        """Build the initial vertex-adjacency dictionary from the boundary."""
        vertices = self.boundary.get_vertices()
        n = len(vertices)

        adjacency = {}
        for i, v in enumerate(vertices):
            adjacency[v] = [vertices[i - 1], vertices[(i + 1) % n]]
        return adjacency

    # ──────────────────────── mesh mutations ───────────────────────
    def add_vertex(self, vertex):
        if vertex in self.mesh:
            raise ValueError(f"Vertex {vertex} already present in the mesh.")
        self.mesh[vertex] = []

    def add_edge(self, v1, v2):
        if v1 == v2:
            raise ValueError("Cannot create a self-loop edge.")
        for v in (v1, v2):
            if v not in self.mesh:
                raise ValueError(f"Vertex {v} not found in the mesh.")
        if v2 in self.mesh[v1]:
            raise ValueError(f"Edge ({v1} ↔ {v2}) already exists.")

        self.mesh[v1].append(v2)
        self.mesh[v2].append(v1)

    # ──────────────────────── new requested API ────────────────────
    def get_mesh(self):
        """
        Return a deep copy of the current adjacency dictionary, so external
        callers cannot mutate the internal state accidentally.
        """
        return copy.deepcopy(self.mesh)

    def get_adjacency_dict(self):
        """
        返回网格的邻接关系字典，格式适合前端可视化

        Returns:
            Dict[str, List[List[float]]]: 字符串化的顶点坐标到邻接顶点列表的映射
        """
        adjacency_dict = {}
        for vertex, neighbors in self.mesh.items():
            # 将顶点坐标转换为字符串格式 "[x,y]"
            vertex_key = f"[{vertex[0]},{vertex[1]}]"
            # 邻接顶点保持为坐标列表格式
            adjacency_dict[vertex_key] = [[neighbor[0], neighbor[1]] for neighbor in neighbors]

        return adjacency_dict

    # ──────────────────────── other utility methods ────────────────
    def get_vertices(self):
        """返回网格中所有顶点的列表"""
        return list(self.mesh.keys())

    def get_vertex_count(self):
        """返回网格中顶点的数量"""
        return len(self.mesh)

    def get_edge_count(self):
        """返回网格中边的数量（每条边计算一次）"""
        total_degree = sum(len(neighbors) for neighbors in self.mesh.values())
        return total_degree // 2  # 每条边被计算了两次

    def has_vertex(self, vertex):
        """检查顶点是否在网格中"""
        return vertex in self.mesh

    def get_neighbors(self, vertex):
        """获取指定顶点的所有邻接顶点"""
        if vertex not in self.mesh:
            raise ValueError(f"Vertex {vertex} not found in the mesh.")
        return copy.deepcopy(self.mesh[vertex])

    def __str__(self):
        """返回网格的字符串表示"""
        return f"Mesh(vertices={self.get_vertex_count()}, edges={self.get_edge_count()})"

    def __repr__(self):
        """返回网格的详细字符串表示"""
        return f"Mesh(vertices={self.get_vertex_count()}, edges={self.get_edge_count()}, adjacency={dict(self.mesh)})"
