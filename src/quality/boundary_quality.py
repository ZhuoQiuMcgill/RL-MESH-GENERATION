def calculate_angle_quality(angle1, angle2, M_angle):
    """
    Calculate angle quality based on three angles.

    Args:
        angle1: First angle
        angle2: Second angle
        M_angle: Maximum angle

    Returns:
        float: Calculated angle quality
    """
    return min(angle1, angle2, M_angle) / M_angle