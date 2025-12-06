import math
import mediapipe as mp

mp_pose = mp.solutions.pose


# Compute the angle formed by three landmarks
# Returns angle in degrees [0, 360)

def calculate_angle(landmark1, landmark2, landmark3):
    
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(
        math.atan2(y3 - y2, x3 - x2) -
        math.atan2(y1 - y2, x1 - x2)
    )
    
    # Ensure result is between 0 and 360
    if angle < 0:
        angle += 360
        
    return angle

# Calculate all relevant joint angles from the full list of 33 pose landmarks
def angles_finder(landmarks):
   
    P = mp_pose.PoseLandmark
    
    return {
        "left_elbow_angle": calculate_angle(
            landmarks[P.LEFT_SHOULDER.value], 
            landmarks[P.LEFT_ELBOW.value], 
            landmarks[P.LEFT_WRIST.value]
        ),
        "right_elbow_angle": calculate_angle(
            landmarks[P.RIGHT_SHOULDER.value], 
            landmarks[P.RIGHT_ELBOW.value], 
            landmarks[P.RIGHT_WRIST.value]
        ),
        "left_shoulder_angle": calculate_angle(
            landmarks[P.LEFT_ELBOW.value], 
            landmarks[P.LEFT_SHOULDER.value], 
            landmarks[P.LEFT_HIP.value]
        ),
        "right_shoulder_angle": calculate_angle(
            landmarks[P.RIGHT_HIP.value], 
            landmarks[P.RIGHT_SHOULDER.value], 
            landmarks[P.RIGHT_ELBOW.value]
        ),
        "left_knee_angle": calculate_angle(
            landmarks[P.LEFT_HIP.value], 
            landmarks[P.LEFT_KNEE.value], 
            landmarks[P.LEFT_ANKLE.value]
        ),
        "right_knee_angle": calculate_angle(
            landmarks[P.RIGHT_HIP.value], 
            landmarks[P.RIGHT_KNEE.value], 
            landmarks[P.RIGHT_ANKLE.value]
        ),
        "hand_angle": calculate_angle(
            landmarks[P.LEFT_ELBOW.value], 
            landmarks[P.RIGHT_SHOULDER.value], 
            landmarks[P.RIGHT_ELBOW.value]
        ),
        "left_hip_angle": calculate_angle(
            landmarks[P.LEFT_SHOULDER.value], 
            landmarks[P.LEFT_HIP.value], 
            landmarks[P.LEFT_KNEE.value]
        ),
        "right_hip_angle": calculate_angle(
            landmarks[P.RIGHT_SHOULDER.value], 
            landmarks[P.RIGHT_HIP.value], 
            landmarks[P.RIGHT_KNEE.value]
        ),
        "neck_angle_uk": calculate_angle(
            landmarks[P.NOSE.value], 
            landmarks[P.LEFT_SHOULDER.value], 
            landmarks[P.RIGHT_SHOULDER.value]
        ),
        "left_wrist_angle_bk": calculate_angle(
            landmarks[P.LEFT_WRIST.value], 
            landmarks[P.LEFT_HIP.value], 
            landmarks[P.LEFT_ANKLE.value]
        ),
        "right_wrist_angle_bk": calculate_angle(
            landmarks[P.RIGHT_WRIST.value], 
            landmarks[P.RIGHT_HIP.value], 
            landmarks[P.RIGHT_ANKLE.value]
        ),
    }