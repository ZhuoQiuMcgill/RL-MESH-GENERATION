import numpy as np


def ray_segment_intersection(ray_origin: np.ndarray, ray_direction: np.ndarray,
                             segment_start: np.ndarray, segment_end: np.ndarray):
    """Calculate intersection point between ray and line segment"""
    # Segment direction vector
    segment_vec = segment_end - segment_start

    # Check for parallelism
    cross_product = ray_direction[0] * segment_vec[1] - ray_direction[1] * segment_vec[0]
    if abs(cross_product) < 1e-10:
        return None

    # Calculate parameters
    to_segment_start = segment_start - ray_origin
    t = (to_segment_start[0] * segment_vec[1] - to_segment_start[1] * segment_vec[0]) / cross_product
    s = (to_segment_start[0] * ray_direction[1] - to_segment_start[1] * ray_direction[0]) / cross_product

    # Check if intersection point is on ray and segment
    if t >= 0 and 0 <= s <= 1:
        intersection = ray_origin + t * ray_direction
        return intersection

    return None


def orientation(p, q, r) -> int:
    """
    Find orientation of ordered triplet (p, q, r).
    Returns:
    0 --> p, q and r are collinear
    1 --> Clockwise
    2 --> Counterclockwise
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - \
          (q[0] - p[0]) * (r[1] - q[1])

    if abs(val) < 1e-10: return 0  # Collinear
    return 1 if val > 0 else 2  # Clockwise or Counterclockwise


def point_on_line_segment(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> bool:
    """
    Check if a point lies on a line segment

    Args:
        point: The point to check
        line_start: Start point of the line segment
        line_end: End point of the line segment

    Returns:
        bool: True if the point is on the line segment, False otherwise
    """
    # Use vector cross product to determine if point is on line segment
    # If point is on segment, cross product should be 0, and point should be within segment range

    # Vectors
    v1 = point - line_start
    v2 = line_end - line_start

    # Cross product (scalar in 2D)
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]

    # If cross product is not 0 (considering floating point precision), point is not on the line
    if abs(cross_product) > 1e-10:
        return False

    # Check if point is within segment range
    dot_product = np.dot(v1, v2)
    squared_length = np.dot(v2, v2)

    if squared_length == 0:  # Segment length is 0
        return np.allclose(point, line_start)

    param = dot_product / squared_length
    return 0 <= param <= 1


def line_segments_intersect(p1, q1, p2, q2) -> bool:
    """
    A robust function to check if line segment 'p1q1' and 'p2q2' intersect.
    This handles all general, collinear, and touching cases.
    """
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case: segments cross each other
    if o1 != o2 and o3 != o4:
        return True

    # Special Cases for collinear points
    # p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if o1 == 0 and point_on_line_segment(np.array(p2), np.array(p1), np.array(q1)):
        return True

    # p1, q1 and q2 are collinear and q2 lies on segment p1q1
    if o2 == 0 and point_on_line_segment(np.array(q2), np.array(p1), np.array(q1)):
        return True

    # p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if o3 == 0 and point_on_line_segment(np.array(p1), np.array(p2), np.array(q2)):
        return True

    # p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if o4 == 0 and point_on_line_segment(np.array(q1), np.array(p2), np.array(q2)):
        return True

    return False  # Doesn't fall in any of the above cases


def segments_overlap_interior(a, b, c, d, eps: float = 1e-8) -> bool:
    """
    Segments a-b and c-d are assumed colinear.
    Return True iff they overlap by more than a single point.
    """
    # Project onto the dominant axis to measure one-dim overlap
    if abs(a[0] - b[0]) >= abs(a[1] - b[1]):
        s1 = sorted([a[0], b[0]])
        s2 = sorted([c[0], d[0]])
    else:
        s1 = sorted([a[1], b[1]])
        s2 = sorted([c[1], d[1]])

    overlap_len = min(s1[1], s2[1]) - max(s1[0], s2[0])
    return overlap_len > eps  # positive-length ⇒ real overlap


def point_to_segment_distance(point, seg_start, seg_end):
    p = np.asarray(point, dtype=float)
    start_vec = np.asarray(seg_start, dtype=float)
    end_vec = np.asarray(seg_end, dtype=float)
    segment = end_vec - start_vec

    if np.allclose(segment, 0.0):  # degenerate segment
        return float(np.linalg.norm(p - start_vec))

    t = np.dot(p - start_vec, segment) / np.dot(segment, segment)
    t = np.clip(t, 0.0, 1.0)  # clamp to segment
    projection = start_vec + t * segment
    return float(np.linalg.norm(p - projection))
