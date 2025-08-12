"""
Mesh class definition

Provides mesh construction, maintenance and query functionality
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
        Return the mesh adjacency relationship dictionary, formatted for frontend visualization

        Returns:
            Dict[str, List[List[float]]]: Mapping from stringified vertex coordinates to adjacent vertex lists
        """
        adjacency_dict = {}
        for vertex, neighbors in self.mesh.items():
            # Convert vertex coordinates to string format "[x,y]"
            vertex_key = f"[{vertex[0]},{vertex[1]}]"
            # Keep adjacent vertices in coordinate list format
            adjacency_dict[vertex_key] = [[neighbor[0], neighbor[1]] for neighbor in neighbors]

        return adjacency_dict

    # ──────────────────────── other utility methods ────────────────
    def get_vertices(self):
        """Return a list of all vertices in the mesh"""
        return list(self.mesh.keys())

    def get_vertex_count(self):
        """Return the number of vertices in the mesh"""
        return len(self.mesh)

    def get_edge_count(self):
        """Return the number of edges in the mesh (each edge counted once)"""
        total_degree = sum(len(neighbors) for neighbors in self.mesh.values())
        return total_degree // 2  # Each edge is counted twice

    def has_vertex(self, vertex):
        """Check if a vertex exists in the mesh"""
        return vertex in self.mesh

    def get_neighbors(self, vertex):
        """Get all adjacent vertices of the specified vertex"""
        if vertex not in self.mesh:
            raise ValueError(f"Vertex {vertex} not found in the mesh.")
        return copy.deepcopy(self.mesh[vertex])

    def __str__(self):
        """Return string representation of the mesh"""
        return f"Mesh(vertices={self.get_vertex_count()}, edges={self.get_edge_count()})"

    def __repr__(self):
        """Return detailed string representation of the mesh"""
        return f"Mesh(vertices={self.get_vertex_count()}, edges={self.get_edge_count()}, adjacency={dict(self.mesh)})"
