# feedback_generator.py - Fixed version with proper pose name handling

import numpy as np


# Pose-specific angle whitelists (lowercase keys)
POSE_ANGLE_WHITELIST = {
    "downdog": [
        "left_shoulder_angle", "right_shoulder_angle",
        "left_hip_angle", "right_hip_angle",
        "left_knee_angle", "right_knee_angle",
        "left_wrist_angle_bk", "right_wrist_angle_bk",
        "neck_angle_uk",
    ],
    "plank": [
        "left_shoulder_angle", "right_shoulder_angle",
        "left_hip_angle", "right_hip_angle",
        "left_knee_angle", "right_knee_angle",
        "left_elbow_angle", "right_elbow_angle",
        "neck_angle_uk",
    ],
    "goddess": [
        "left_hip_angle", "right_hip_angle",
        "left_knee_angle", "right_knee_angle",
        "left_shoulder_angle", "right_shoulder_angle",
    ],
    "tree": [
        "left_hip_angle", "right_hip_angle",
        "left_knee_angle", "right_knee_angle",
        "left_shoulder_angle", "right_shoulder_angle",
    ],
    "warrior2": [
        "left_hip_angle", "right_hip_angle",
        "left_knee_angle", "right_knee_angle",
        "left_shoulder_angle", "right_shoulder_angle",
        "hand_angle",
    ],
}


# Friendly names for display
FRIENDLY_NAMES = {
    "left_hip_angle": "Left Hip",
    "right_hip_angle": "Right Hip",
    "left_knee_angle": "Left Knee",
    "right_knee_angle": "Right Knee",
    "left_elbow_angle": "Left Elbow",
    "right_elbow_angle": "Right Elbow",
    "neck_angle_uk": "Neck Alignment",
    "left_shoulder_angle": "Left Shoulder",
    "right_shoulder_angle": "Right Shoulder",
    "left_wrist_angle_bk": "Left Wrist Line",
    "right_wrist_angle_bk": "Right Wrist Line",
    "hand_angle": "Arm Level",
}


def circular_diff(a, b):
    """Calculate circular difference between two angles."""
    return (a - b + 180) % 360 - 180


def pretty_name(name):
    """Convert internal name to friendly display name."""
    return FRIENDLY_NAMES.get(name, name.replace("_", " ").title())


def classify_severity(diff, threshold):
    """Classify deviation severity."""
    ratio = abs(diff) / max(threshold, 1e-6)
    if ratio < 1.2:
        return "mild"
    elif ratio < 1.8:
        return "moderate"
    else:
        return "severe"


def describe_angle_feedback(pose, angle_name, diff, side):
    """Generate pose-specific coaching feedback."""
    more = diff > 0
    side_word = f"{side} " if side else ""

    # Downdog & Plank
    if pose in ["downdog", "plank"]:
        if "hip_angle" in angle_name:
            return f"{'Lower' if more else 'Lift'} your {side_word}hip slightly for better alignment."
        if "knee_angle" in angle_name:
            return f"{'Straighten' if more else 'Soften'} your {side_word}knee a bit more."
        if "shoulder_angle" in angle_name:
            return f"{'Push the floor away and open' if more else 'Stack'} your {side_word}shoulder more."
        if "wrist_angle_bk" in angle_name:
            return f"Adjust your {side_word}arm line from shoulder to wrist."
        if "elbow_angle" in angle_name:
            return f"{'Straighten' if more else 'Slightly bend'} your {side_word}elbow."
        if "neck_angle" in angle_name:
            return "Lengthen the back of your neck" if more else "Gently relax your neck."

    # Goddess & Warrior2
    if pose in ["goddess", "warrior2"]:
        if "knee_angle" in angle_name:
            return f"{'Bend' if more else 'Straighten'} your {side_word}knee slightly."
        if "hip_angle" in angle_name:
            return f"{'Draw your hip under' if more else 'Open your hip more to the side'} for {side_word}alignment."
        if "shoulder_angle" in angle_name:
            return f"Adjust your {side_word}shoulder opening."
        if "hand_angle" in angle_name:
            return "Level your arms at shoulder height."

    # Tree
    if pose == "tree":
        if "knee_angle" in angle_name:
            return f"{'Press your knee out' if more else 'Keep your knee from collapsing'} for {side_word}stability."
        if "hip_angle" in angle_name:
            return f"{'Square' if more else 'Open'} your {side_word}hip slightly."
        if "shoulder_angle" in angle_name:
            return f"Balance your {side_word}shoulder alignment."

    return f"Adjust your {pretty_name(angle_name)} alignment."


def generate_feedback(angle_dict, pose_name, angle_stats, feature_cols, std_factor=1.0, min_deg=20.0):
    """
    Generate detailed feedback text for the predicted pose.
    
    Parameters
    ----------
    angle_dict : dict
        Dictionary of computed angles from the user's image.
    pose_name : str
        Predicted pose name (can be "Plank" or "plank").
    angle_stats : dict
        Statistical data for each pose (keys are lowercase).
    feature_cols : list
        Ordered list of angle feature names.
    std_factor : float
        Multiplier on standard deviation for flagging abnormal angles.
    min_deg : float
        Minimum absolute angle difference to flag.
    
    Returns
    -------
    str
        Formatted feedback text.
    """
    # Normalize pose name to lowercase for lookups
    pose_lower = pose_name.lower()
    
    # Debug: print available keys
    print(f"Looking for pose: '{pose_lower}' in angle_stats keys: {list(angle_stats.keys())}")
    
    # Get statistical data (try multiple variations)
    stats = None
    for key in [pose_lower, pose_name, pose_name.capitalize(), pose_name.upper()]:
        if key in angle_stats:
            stats = angle_stats[key]
            print(f"✓ Found stats with key: '{key}'")
            break
    
    if stats is None:
        return f"⚠️ No statistical data available for pose: {pose_name}\n\nAvailable poses: {', '.join(angle_stats.keys())}"
    
    mean_angles = np.array(stats["mean"])
    std_angles = np.array(stats["std"])
    
    # Get feature values
    feats = [angle_dict.get(name, 0.0) for name in feature_cols]
    
    # Determine which angles to check
    whitelist = POSE_ANGLE_WHITELIST.get(pose_lower, feature_cols)
    
    feedback_lines = []
    violations = []
    good_angles = []  # Track angles within tolerance
    
    # Check each angle
    for name, value, mean, std in zip(feature_cols, feats, mean_angles, std_angles):
        if name not in whitelist:
            continue
        
        diff = circular_diff(value, mean)
        threshold = max(min_deg, std_factor * std)
        
        if abs(diff) < threshold:
            # Angle is GOOD - still record it
            good_angles.append((name, value, mean, diff))
            continue
        
        # Angle needs attention - flag it
        severity = classify_severity(diff, threshold)
        side = "left" if "left_" in name else "right" if "right_" in name else ""
        direction = "more" if diff > 0 else "less"
        
        violations.append((name, diff, threshold, severity))
        
        # Generate feedback text
        text = describe_angle_feedback(pose_lower, name, diff, side)
        if text is None:
            text = (
                f"Your {pretty_name(name).lower()} is about "
                f"{abs(diff):.1f}° {direction} than typical."
            )
        
        # Format like notebook: "- (severity) text (details)"
        feedback_lines.append(
            f"   - ({severity}) {text} "
            f"(your {pretty_name(name).lower()}: {value:.1f}°, "
            f"typical: {mean:.1f}°; "
            f"≈{abs(diff):.1f}° {direction}, "
            f"threshold ≈{threshold:.1f}°)"
        )
    
    # Calculate score
    total_checked = len([n for n in feature_cols if n in whitelist])
    bad = len(violations)
    good = total_checked - bad
    score = max(0.0, 100.0 * good / max(total_checked, 1))
    
    # Sort violations for top-3
    violations_sorted = sorted(
        violations,
        key=lambda v: abs(v[1]) / max(v[2], 1e-6),
        reverse=True
    )
    top3 = violations_sorted[:3]
    
    # Build final feedback string - match notebook format
    result = f"Feedback Analysis \n"
    result += f"(Flagging joints where |difference| > max({std_factor} * std, {min_deg}°))\n\n"
    
    # Show corrections if any
    if feedback_lines:
        result += " ⚠️ Corrections Needed: \n\n"
        result += "\n\n".join(feedback_lines)
        result += "\n\n"
    
    # Score and summary
    result += f" Overall alignment score: {score:.1f} / 100 \n"
    result += f"Joints within tolerance: {good}/{total_checked}\n"
    result += f"Joints needing attention: {bad}/{total_checked}\n\n"
    
    # Top-3 focus areas
    if top3:
        result += "🎯 Focus first on these joints: \n"
        for name, diff, threshold, severity in top3:
            direction = "more" if diff > 0 else "less"
            result += (
                f"   * ({severity}) {pretty_name(name).lower()} "
                f"(≈{abs(diff):.1f}° {direction}, threshold ≈{threshold:.1f}°)\n"
            )
        result += "\n"
    
    # Encouraging message
    if bad == 0:
        result += "🎉 All checked joint angles look good! \n\n"
    elif good > 0:
        result += f"💪 Nice work! {good} joints are already within the target range.\n\n"
    
    # Show good angles with their actual deviations
    if good_angles:
        result += "✅ Angles Within Tolerance Range: \n\n"
        for name, value, mean, diff in good_angles:
            result += (
                f"   • {pretty_name(name)}: {value:.1f}° "
                f"(typical: {mean:.1f}°, ≈{abs(diff):.1f}° {'more' if diff > 0 else 'less'})\n"
            )
    
    return result