from src.utils.angle import euclidean_distance, get_interior_angle
import math
import inspect
from typing import List, Tuple, Dict, Callable

Point = Tuple[float, float]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _validate(element: List[Point]) -> bool:
    return element is not None and len(element) == 4


def _sub(a: Point, b: Point) -> Point:
    return a[0] - b[0], a[1] - b[1]


def _norm(v: Point) -> float:
    return math.hypot(v[0], v[1])


def _cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _edge_lengths(element: List[Point]) -> List[float]:
    return [euclidean_distance(element[i], element[(i + 1) % 4]) for i in range(4)]


def _angles(element: List[Point]) -> List[float]:
    return [
        get_interior_angle(element[(i - 1) % 4], element[i], element[(i + 1) % 4])
        for i in range(4)
    ]


def quality_robust(element: List[Point]) -> float:
    if not _validate(element):
        return 0.0
    edges = _edge_lengths(element)
    diag1 = euclidean_distance(element[0], element[2])
    diag2 = euclidean_distance(element[1], element[3])
    max_diag = max(diag1, diag2)
    min_edge = min(edges)
    q_edge = (math.sqrt(2) * min_edge) / max_diag if max_diag > 0 else 0.0
    angs = _angles(element)
    min_angle = min(angs)
    max_angle = max(angs)
    q_angle = min_angle / max_angle if max_angle > 0 else 0.0
    return _clamp(math.sqrt(q_edge * q_angle))


def quality_default(element: List[Point]) -> float:
    if not _validate(element):
        return 0.0
    edges = _edge_lengths(element)
    aspect = max(edges) / min(edges) if min(edges) > 0 else float('inf')
    angs = _angles(element)
    max_error = max(abs(a - math.pi / 2) for a in angs)
    denom = aspect + max_error
    return _clamp(1.0 / denom) if denom > 0 else 0.0


def quality_s_jacobian(element: List[Point]) -> float:
    if not _validate(element):
        return 0.0
    v0, v1, v2, v3 = element
    l0 = _sub(v1, v0)
    l1 = _sub(v2, v1)
    l2 = _sub(v3, v2)
    l3 = _sub(v0, v3)
    j0 = _cross(l3, l0) / (_norm(l3) * _norm(l0) or 1.0)
    j1 = _cross(l0, l1) / (_norm(l0) * _norm(l1) or 1.0)
    j2 = _cross(l1, l2) / (_norm(l1) * _norm(l2) or 1.0)
    j3 = _cross(l2, l3) / (_norm(l2) * _norm(l3) or 1.0)
    q = min(j0, j1, j2, j3)
    return _clamp(q if q > 0 else 0.0)


class QualityManager:
    def __init__(self):
        self._quality_methods: Dict[str, Callable] = {}
        self._discover_quality_methods()
    
    def _discover_quality_methods(self):
        current_module = inspect.getmodule(self)
        for name, func in inspect.getmembers(current_module, inspect.isfunction):
            if name.startswith('quality_') and not name.startswith('_'):
                method_name = name.replace('quality_', '')
                self._quality_methods[method_name] = func
    
    def get_available_methods(self) -> List[str]:
        return list(self._quality_methods.keys())
    
    def calculate_quality(self, method_name: str, vertices: List[Point]) -> float:
        if method_name not in self._quality_methods:
            raise ValueError(f"Quality method '{method_name}' not found")
        return self._quality_methods[method_name](vertices)
    
    def get_method_info(self) -> Dict[str, Dict[str, str]]:
        info = {}
        for method_name, func in self._quality_methods.items():
            doc = func.__doc__ or "No description available"
            info[method_name] = {
                'description': doc.strip().split('\n')[0] if doc else "Quality measurement method",
                'full_name': f'quality_{method_name}'
            }
        return info
