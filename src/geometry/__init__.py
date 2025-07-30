from .boundary import Boundary
from .mesh import Mesh
from .reference_point_selectors import RLReferencePointSelector, RandomReferencePointSelector

__all__ = [
    'Boundary',
    'Mesh',

    'RLReferencePointSelector',
]

AVALIABLE_REFERENCE_POINT_SELECTORS = {
    "AVG INTERIOR ANGLE SELECTOR": RLReferencePointSelector(),
    "RANDOM SELECTOR": RLReferencePointSelector(),
}

__version__ = '1.5.0'
__author__ = 'ZhuoQiuMcgill'
