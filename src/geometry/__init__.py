from .boundary import Boundary
from .mesh import Mesh
from .reference_point_selectors import RLReferencePointSelector, RandomReferencePointSelector, MinAngleReferenceSelector

__all__ = [
    'Boundary',
    'Mesh',

    'RLReferencePointSelector',
]

AVALIABLE_REFERENCE_POINT_SELECTORS = {
    "AVG INTERIOR ANGLE SELECTOR": RLReferencePointSelector(),
    "RANDOM SELECTOR": RLReferencePointSelector(),
    "MIN INTERIOR ANGLE SELECTOR": MinAngleReferenceSelector(),
}

__version__ = '1.5.0'
__author__ = 'ZhuoQiuMcgill'
