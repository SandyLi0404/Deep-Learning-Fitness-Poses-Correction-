# angle_calculation.py - Updated version matching new_version_feedback_image.ipynb

import math
import mediapipe as mp

mp_pose = mp.solutions.pose


def calculate_angle(landmark1, landmark2, landmark3):
    """
    Compute the angle formed by three landmarks.
    Uses atan2 for proper angle calculation in [0, 360) range.
    
    Parameters
    ----------
    landmark1, landmark2, landmark3 : tuple
        Each landmark is (x, y, z) from MediaPipe.
    
    Returns
    -------
    float
        Angle in degrees [0, 360).
    """
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(
        math.atan2(y3 - y2, x3 - x2) -
        math.atan2(y1 - y2, x1 - x2)
    )
    if angle < 0:
        angle += 360
    return angle


def angles_finder(landmarks):
    """
    Compute all 12 required body joint angles from MediaPipe pose landmarks.
    
    Parameters
    ----------
    landmarks : list of tuples
        List of (x, y, z) landmark coordinates in pixel space.
        Should contain 33 landmarks following MediaPipe Pose format.
    
    Returns
    -------
    dict
        Dictionary containing all 12 angle measurements.
    """
    P = mp_pose.PoseLandmark

    # Elbow angles
    left_elbow_angle = calculate_angle(
        landmarks[P.LEFT_SHOULDER.value],
        landmarks[P.LEFT_ELBOW.value],
        landmarks[P.LEFT_WRIST.value]
    )
    right_elbow_angle = calculate_angle(
        landmarks[P.RIGHT_SHOULDER.value],
        landmarks[P.RIGHT_ELBOW.value],
        landmarks[P.RIGHT_WRIST.value]
    )

    # Shoulder angles
    left_shoulder_angle = calculate_angle(
        landmarks[P.LEFT_ELBOW.value],
        landmarks[P.LEFT_SHOULDER.value],
        landmarks[P.LEFT_HIP.value]
    )
    right_shoulder_angle = calculate_angle(
        landmarks[P.RIGHT_HIP.value],
        landmarks[P.RIGHT_SHOULDER.value],
        landmarks[P.RIGHT_ELBOW.value]
    )

    # Knee angles
    left_knee_angle = calculate_angle(
        landmarks[P.LEFT_HIP.value],
        landmarks[P.LEFT_KNEE.value],
        landmarks[P.LEFT_ANKLE.value]
    )
    right_knee_angle = calculate_angle(
        landmarks[P.RIGHT_HIP.value],
        landmarks[P.RIGHT_KNEE.value],
        landmarks[P.RIGHT_ANKLE.value]
    )

    # Hand angle (across shoulders)
    hand_angle = calculate_angle(
        landmarks[P.LEFT_ELBOW.value],
        landmarks[P.RIGHT_SHOULDER.value],
        landmarks[P.RIGHT_ELBOW.value]
    )

    # Hip angles
    left_hip_angle = calculate_angle(
        landmarks[P.LEFT_SHOULDER.value],
        landmarks[P.LEFT_HIP.value],
        landmarks[P.LEFT_KNEE.value]
    )
    right_hip_angle = calculate_angle(
        landmarks[P.RIGHT_SHOULDER.value],
        landmarks[P.RIGHT_HIP.value],
        landmarks[P.RIGHT_KNEE.value]
    )

    # Neck angle
    neck_angle_uk = calculate_angle(
        landmarks[P.NOSE.value],
        landmarks[P.LEFT_SHOULDER.value],
        landmarks[P.RIGHT_SHOULDER.value]
    )

    # Wrist angles (body alignment)
    left_wrist_angle_bk = calculate_angle(
        landmarks[P.LEFT_WRIST.value],
        landmarks[P.LEFT_HIP.value],
        landmarks[P.LEFT_ANKLE.value]
    )
    right_wrist_angle_bk = calculate_angle(
        landmarks[P.RIGHT_WRIST.value],
        landmarks[P.RIGHT_HIP.value],
        landmarks[P.RIGHT_ANKLE.value]
    )

    return {
        "left_elbow_angle": left_elbow_angle,
        "right_elbow_angle": right_elbow_angle,
        "left_shoulder_angle": left_shoulder_angle,
        "right_shoulder_angle": right_shoulder_angle,
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "hand_angle": hand_angle,
        "left_hip_angle": left_hip_angle,
        "right_hip_angle": right_hip_angle,
        "neck_angle_uk": neck_angle_uk,
        "left_wrist_angle_bk": left_wrist_angle_bk,
        "right_wrist_angle_bk": right_wrist_angle_bk,
    }


def circular_diff(a, b):
    """
    Calculate the circular difference between two angles in degrees.
    Result is in [-180, 180].
    
    Parameters
    ----------
    a, b : float
        Angles in degrees.
    
    Returns
    -------
    float
        Signed difference in degrees.
    """
    return (a - b + 180) % 360 - 180