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
