class Mesh:
    def __init__(self, boundary):
        self.boundary = boundary
        self.mesh = self.init_mesh()

    def init_mesh(self):
        """
        Build the initial vertex-adjacency dictionary from the boundary.
        Each boundary vertex is adjacent to its clockwise predecessor
        and successor.
        """
        vertices = self.boundary.get_vertices()
        n = len(vertices)
        adjacency = {}

        for i, v in enumerate(vertices):
            prev_v = vertices[i - 1]          # wrap-around works in Python
            next_v = vertices[(i + 1) % n]
            adjacency[v] = [prev_v, next_v]

        return adjacency

    # ────────────────────────────────────────────────────────────────
    # New functionality
    # ────────────────────────────────────────────────────────────────
    def add_vertex(self, vertex):
        """
        Add an isolated vertex to the mesh.

        Parameters
        ----------
        vertex : tuple[float, float]
            2-D coordinate of the new vertex.

        Raises
        ------
        ValueError if the vertex already exists.
        """
        if vertex in self.mesh:
            raise ValueError(f"Vertex {vertex} already present in the mesh.")
        self.mesh[vertex] = []    # start with no neighbours

    def add_edge(self, v1, v2):
        """
        Insert an (undirected) edge between two existing vertices.

        Parameters
        ----------
        v1, v2 : tuple[float, float]
            End-points of the edge.

        Raises
        ------
        ValueError if the vertices are identical, missing, or already connected.
        """
        if v1 == v2:
            raise ValueError("Cannot create a self-loop edge.")

        # Both end-points must be known
        for v in (v1, v2):
            if v not in self.mesh:
                raise ValueError(f"Vertex {v} not found in the mesh.")

        # Avoid duplicate edges
        if v2 in self.mesh[v1]:
            raise ValueError(f"Edge ({v1} ↔ {v2}) already exists.")

        # Add the connection symmetrically
        self.mesh[v1].append(v2)
        self.mesh[v2].append(v1)
