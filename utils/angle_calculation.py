import numpy as np

def calculate_angle(point1, point2, point3):
    """
    Compute the angle formed by three points: point1 - point2 - point3.
    The angle is measured at point2.
    
    Parameters
    ----------
    point1, point2, point3 : dict
        Each point must contain keys 'x' and 'y' (normalized coordinates from MediaPipe).
    
    Returns
    -------
    float
        Angle in degrees.
    """
    
    # Create vectors from point2 to point1 and point2 to point3
    vector1 = np.array([point1['x'] - point2['x'], point1['y'] - point2['y']])
    vector2 = np.array([point3['x'] - point2['x'], point3['y'] - point2['y']])
    
    # Compute cosine of the angle using dot product
    cosine = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    
    # Convert cosine value to angle in degrees (clip to avoid numerical errors)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    
    return angle


def calculate_angles(landmarks):
    """
    Compute all required body joint angles from MediaPipe pose landmarks.
    
    Parameters
    ----------
    landmarks : list of dict
        A list of landmarks where each element contains keys 'x', 'y'.
        Indices follow MediaPipe Pose format (0–32).
    
    Returns
    -------
    np.ndarray
        Array of computed joint angles.
    """
    
    angles = []
    
    # Example angle calculations below.
    # Modify according to your complete angle definitions from angles.ipynb.
    
    # Left elbow angle: shoulder → elbow → wrist
    left_elbow = calculate_angle(
        landmarks[11],  # Left shoulder
        landmarks[13],  # Left elbow
        landmarks[15]   # Left wrist
    )
    angles.append(left_elbow)
    
    # Right elbow angle: shoulder → elbow → wrist
    right_elbow = calculate_angle(
        landmarks[12],  # Right shoulder
        landmarks[14],  # Right elbow
        landmarks[16]   # Right wrist
    )
    angles.append(right_elbow)
    
    # TODO: Add all remaining angles (knees, hips, shoulders, torso, etc.)
    # Example:
    # left_knee = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
    # angles.append(left_knee)
    
    return np.array(angles)
