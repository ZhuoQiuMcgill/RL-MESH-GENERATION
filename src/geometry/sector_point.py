from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SectorPoint:
    """
    Container for a candidate boundary vertex used in fan-sector selection.

    Fields
    ------
    vertex : Tuple[float, float]
        Original boundary vertex coordinates (global frame).
    dist : float
        Euclidean distance from the reference vertex v0 to this candidate.
    angle_abs : float
        Absolute polar angle (0..2π) of vector (v - v0) in the global frame.
    delta_oriented : float
        Oriented angular distance from the right-neighbor direction (vr1).
        It is measured along the SHORT arc orientation (CCW = +, CW = -) and
        should lie in (0, span) for valid candidates inside the short-arc domain.
    sector_index : Optional[int]
        Index of the sector [0..g-1] this point belongs to. None if not inside
        any open sector (e.g., on/near boundaries numerically).
    center_delta : Optional[float]
        Absolute angular distance between this point's angle and the center
        angle of the sector it belongs to. Used for tie-breaking when distances
        are equal within numerical tolerance.
    in_radius : bool
        Whether the point satisfies the radius cap.
    in_short_arc : bool
        Whether the point is strictly inside the short-arc domain (open).
    """

    vertex: Tuple[float, float]
    dist: float
    angle_abs: float
    delta_oriented: float
    sector_index: Optional[int] = None
    center_delta: Optional[float] = None
    in_radius: bool = True
    in_short_arc: bool = True

    def better_than(self, other: Optional["SectorPoint"], eps: float = 1e-12) -> bool:
        """Return True if this point is preferred over 'other' in the same sector.

        Preference order:
        1) Smaller distance.
        2) If distance ties (|d1 - d2| <= eps), smaller center_delta.
        """
        if other is None:
            return True
        if self.dist < other.dist - eps:
            return True
        if abs(self.dist - other.dist) <= eps:
            # When center_delta is None, treat it as +inf (should not happen if
            # membership is computed correctly, but keep defensive behavior).
            c1 = float("inf") if self.center_delta is None else self.center_delta
            c2 = float("inf") if other.center_delta is None else other.center_delta
            return c1 < c2 - eps
        return False

    def as_dict(self) -> dict:
        """Convenience for logging or debugging."""
        return {
            "vertex": (float(self.vertex[0]), float(self.vertex[1])),
            "dist": float(self.dist),
            "angle_abs": float(self.angle_abs),
            "delta_oriented": float(self.delta_oriented),
            "sector_index": None if self.sector_index is None else int(self.sector_index),
            "center_delta": None if self.center_delta is None else float(self.center_delta),
            "in_radius": bool(self.in_radius),
            "in_short_arc": bool(self.in_short_arc),
        }
