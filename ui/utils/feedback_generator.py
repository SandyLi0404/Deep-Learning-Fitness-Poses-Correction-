import numpy as np

# friendly names for display
FRIENDLY_NAMES = {
    "left_hip_angle": "Left Hip",
    "right_hip_angle": "Right Hip",
    "left_knee_angle": "Left Knee",
    "right_knee_angle": "Right Knee",
    "left_elbow_angle": "Left Elbow",
    "right_elbow_angle": "Right Elbow",
    "neck_angle_uk": "Neck",
    "left_shoulder_angle": "Left Shoulder",
    "right_shoulder_angle": "Right Shoulder",
    "left_wrist_angle_bk": "Left Wrist",
    "right_wrist_angle_bk": "Right Wrist",
    "hand_angle": "Arms Level",
}

# Here we defines which joints are relevant for each pose
POSE_ANGLE_WHITELIST = {
    "downdog": [
        "left_knee_angle", "right_knee_angle", 
        "left_hip_angle", "right_hip_angle", 
        "left_shoulder_angle", "right_shoulder_angle", 
        "left_elbow_angle", "right_elbow_angle",
        "neck_angle_uk"
    ],
    "plank": [
        "left_hip_angle", "right_hip_angle", 
        "left_knee_angle", "right_knee_angle", 
        "left_elbow_angle", "right_elbow_angle",
        "left_shoulder_angle", "right_shoulder_angle",
        "neck_angle_uk"
    ],
    "goddess": [
        "left_knee_angle", "right_knee_angle", 
        "left_hip_angle", "right_hip_angle",
        "left_shoulder_angle", "right_shoulder_angle",
        "left_elbow_angle", "right_elbow_angle",
        "neck_angle_uk"
    ],
    "tree": [
        "left_hip_angle", "right_hip_angle", 
        "left_knee_angle", "right_knee_angle",
        "left_shoulder_angle", "right_shoulder_angle",
        "neck_angle_uk"
    ], 
    "warrior2": [
        "left_knee_angle", "right_knee_angle",
        "left_hip_angle", "right_hip_angle",
        "left_shoulder_angle", "right_shoulder_angle", 
        "left_elbow_angle", "right_elbow_angle",
        "hand_angle", "neck_angle_uk"
    ] 
}

# Converts 0-360 angle to 0-180 for human readability.
def normalize_to_180(angle):
   
    if angle > 180:
        return 360 - angle
    return angle

# Calculates minimal difference on a circle
def circular_diff(a, b):
   
    diff = (a - b + 180) % 360 - 180
    return diff

def pretty_name(name):
    return FRIENDLY_NAMES.get(name, name.replace("_", " ").title())

def classify_severity(diff, threshold):
    ratio = abs(diff) / max(threshold, 1e-6)
    if ratio < 1.2: return "mild"
    elif ratio < 1.8: return "moderate"
    else: return "severe"

# Generates text based on difference direction
def describe_angle_feedback(pose, angle_name, diff, side):
    
    # Simply tell them to adjust based on direction
    # Since we are using model stats, 'positive' diff usually means 'larger/more open than average'
    if diff > 0:
        return f"Decrease angle of {side} {pretty_name(angle_name)}."
    else:
        return f"Increase angle of {side} {pretty_name(angle_name)}."

# Generates feedback using MODEL STATISTICS but displaying user-friendly angles
def generate_feedback(angle_dict, pose_name, angle_stats, feature_cols, std_factor=1.0, min_deg=20.0):
   
    pose_lower = pose_name.lower()
    
    # 1. Get Statistical Data from Model (angle_stats)
    # We prioritize the data loaded from the .pkl file
    stats = angle_stats.get(pose_name) # Try exact match first
    if not stats:
        
        # Fallback case-insensitive search
        for key in angle_stats.keys():
            if key.lower() == pose_lower:
                stats = angle_stats[key]
                break
    
    if not stats:
        return f"Error: No model statistics found for pose '{pose_name}'."

    mean_angles = stats["mean"]
    std_angles = stats["std"]

    whitelist = POSE_ANGLE_WHITELIST.get(pose_lower, [])
    
    correction_lines = []
    correct_lines = []
    
    # 2. Iterate through features
    for i, name in enumerate(feature_cols):
        if name not in whitelist:
            continue
            
        # Raw Values (0-360) - Used for Calculation
        raw_user_val = angle_dict.get(name, 0.0)
        raw_model_mean = mean_angles[i]
        model_std = std_angles[i]
        
        # Display Values (0-180) - Used for Reading
        display_user_val = normalize_to_180(raw_user_val)
        display_model_val = normalize_to_180(raw_model_mean)
        
        # 3. Calculate Difference (using Circular Math on RAW values)
        diff = circular_diff(raw_user_val, raw_model_mean)
        
        # Threshold: Model Std * Factor (but at least min_deg)
        threshold = max(min_deg, std_factor * model_std)
        
        # 4. Check Pass/Fail
        is_correct = False
        if abs(diff) <= threshold:
            is_correct = True
            
        # 5. Generate Output
        p_name = pretty_name(name)
        severity = classify_severity(diff, threshold)
        side = "left" if "left_" in name else "right" if "right_" in name else ""
        
        if is_correct:
            line = f"- ✅ {p_name}: {display_user_val:.1f}° (Avg: {display_model_val:.1f}°)"
            correct_lines.append(line)
        else:
            text = describe_angle_feedback(pose_lower, name, diff, side)
            # Show display values, but use calculated diff severity
            line = (
                f"- ⚠️ ({severity}) {text} "
                f"({p_name}: {display_user_val:.1f}°, Typical: {display_model_val:.1f}°, Diff: {abs(diff):.1f}°)"
            )
            correction_lines.append(line)

    # Score calculation
    total_checked = len(correct_lines) + len(correction_lines)
    if total_checked == 0:
        score = 0.0
    else:
        score = max(0.0, 100.0 * len(correct_lines) / total_checked)
    
    # Construct Final Output Text
    result = f" Feedback Analysis for {pose_name.title()}\n"
    result += f" Overall Alignment Score: {score:.1f} / 100\n"
    result += f"(Checked {total_checked} joints against model statistics)\n\n"
    
    if correction_lines:
        result += " Corrections Needed:\n"
        result += "\n".join(correction_lines)
        result += "\n\n"
    
    if correct_lines:
        result += " ✅ Good Alignment:\n"
        result += "\n".join(correct_lines)
        result += "\n\n"
        
    if not correction_lines and total_checked > 0:
        result += "\n✨ Perfect form! Matches all typical expert alignments."
        
    return result